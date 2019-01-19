#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <thread>

#include "header/Hitable.h"
#include "header/Camera.h"
#include "header/GLWindow.h"
#include "curand_kernel.h"
#include "MathHelper.h"
using namespace std;


Vec3 cam_rotation(0, 0, 0),camera_lookat(0,0,0);
Camera cam(Vec3(-2, 1, 1), Vec3(0, 0, -1), Vec3(0, 1, 0), 90, float(ImageWidth) / float(ImageHeight));
bool Use_IPR=true;
float * h_pixeldataF;
curandState *d_rng_states = nullptr;

const int object_count = 4;
const int material_count = 4;
float p1[HITABLE_PARAMTER_COUNT] = { 0,0,-1,0.5,0 };
float p2[HITABLE_PARAMTER_COUNT] = { 0,-100.5,-1,100,1 };
float p3[HITABLE_PARAMTER_COUNT] = { -1,0,-1,0.5,2 };
float p4[HITABLE_PARAMTER_COUNT] = { 1,0,-1,0.5,3 };
GPUHitable w[object_count] = { GPUHitable(p1),GPUHitable(p2),GPUHitable(p3),GPUHitable(p4) };


float m1[MATERIAL_PARAMTER_COUNT] = { 3,0,0,0.5f,0,0 };
float m2[MATERIAL_PARAMTER_COUNT] = { 1,1,1,0.5,0,0 };
float m3[MATERIAL_PARAMTER_COUNT] = { 0,0,1,0.1,0,0 };
float m4[MATERIAL_PARAMTER_COUNT] = { 0,1,0,0,0,0 };
Material m[material_count] = { Material(DIELECTIRC,m1),Material(LAMBERTIAN,m2) ,Material(METAL,m3) ,Material(LAMBERTIAN,m4) };


void InitData()
{
	h_pixeldataF = new float[ImageWidth*ImageHeight * 4];
	for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)h_pixeldataF[i] = 0;

	PixelLength = ImageHeight * ImageWidth * 4;
	PixelData = new GLbyte[PixelLength];
	for (auto i = 0; i < PixelLength; i++)
		PixelData[i] = static_cast<GLbyte>(int(0));
}

__device__ bool HitTest(GPUHitable* list, int size, Ray r, float t_min, float t_max,  HitRecord& rec, Material* materials)
{
	HitRecord temp_rec;
	auto hit_anything = false;
	double closest_so_far = t_max;
	for (auto i=0;i<size;i++)
	{
		if (list[i].Hit(r, t_min, closest_so_far, temp_rec, materials)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

__global__
void IPRSampler(int d_width, int d_height, int worldsize, int seed, int SPP,int MST,GPUHitable* d_world, float * d_pixeldata, Camera* d_camera, curandState *const rngStates, Material* materials)
{
	// Determine thread ID
	 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto tid2 = blockIdx.y * blockDim.y + threadIdx.y;

	curand_init(seed+tid+tid2*d_width,0, 0, &rngStates[tid]);			//初始化随机数
	Vec3 color(0, 0, 0);

	const int x = blockIdx.x * 16 + threadIdx.x,y = blockIdx.y * 16 + threadIdx.y;
	//**********  Debug Specific Pixel **********
	//int x = 256, y = 256;
	//float u = float(x) / float(512);
	//float v = float(y) / float(512);

	for (auto j = 0; j < SPP; j++) {
		const auto u = float(x + curand_uniform(&rngStates[tid])) / float(d_width),
			v = float(y + curand_uniform(&rngStates[tid])) / float(d_height);
		Ray ray(d_camera->Origin(), d_camera->LowerLeftCorner() + u * d_camera->Horizontal() + v * d_camera->Vertical() - d_camera->Origin());
		Vec3 c(0, 0, 0);
		Vec3 factor(1, 1, 1);
		for (auto i = 0; i < MST; i++)
		{
			HitRecord rec;
			if (HitTest(d_world, worldsize, ray, 0.001, 99999, rec, materials))
			{
				// random in unit sphere
				Vec3 random_in_unit_sphere;
				do random_in_unit_sphere = 2.0*Vec3(curand_uniform(&rngStates[tid]), curand_uniform(&rngStates[tid]), curand_uniform(&rngStates[tid])) - Vec3(1, 1, 1);
				while (random_in_unit_sphere.squared_length() >= 1.0);

				Ray scattered;
				Vec3 attenuation;
				if (i < MST&&rec.mat_ptr->scatter(ray, rec, attenuation, scattered, random_in_unit_sphere, curand_uniform(&rngStates[tid]))) {
					//auto target = rec.p + rec.normal + random_in_unit_sphere;
					factor *= attenuation;
					ray = scattered;
				}
				//****** 超过最大反射次数，返回黑色 ******
				else
				{
					c = Vec3(0, 0, 0);
					break;
				}
			}
			else
			{
				const auto t = 0.5*(unit_vector(ray.Direction()).y() + 1);
				c = factor * ((1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0));
				break;
			}
		}
		color += c;
	}
	//color /= SPP;

	//SetColor
	const auto i = d_width * 4 * y + x * 4;
	d_pixeldata[i] += color.r() ;
	d_pixeldata[i + 1] += color.g() ;
	d_pixeldata[i + 2] += color.b() ;
	d_pixeldata[i + 3] += SPP;
}


namespace IPR
{
	int * d_Width = 0;
	int * d_Height = 0;
	float * d_pixeldata;
	GPUHitable * d_world_gpu;
	Material * d_materials;
	Camera * d_camera;
	dim3 grid(ImageWidth / BlockSize, ImageHeight / BlockSize), block(BlockSize, BlockSize);
	//dim3 grid(1),block(1);		
	void IPR_Init()
	{
		//******  分配地址 ****** 
		cudaMalloc(reinterpret_cast<void**>(&d_Width), sizeof(int));
		cudaMalloc(reinterpret_cast<void**>(&d_Height), sizeof(int));
		cudaMalloc(reinterpret_cast<void**>(&d_world_gpu), sizeof(GPUHitable) * object_count);
		cudaMalloc(reinterpret_cast<void**>(&d_materials), sizeof(Material)*material_count);
		cudaMalloc(reinterpret_cast<void**>(&d_pixeldata), ImageWidth * ImageHeight * 4 * sizeof(float));
		cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(Camera));
		if (current_spp == 0) {
			if (d_rng_states != nullptr)cudaFree(d_rng_states);
			cudaMalloc(reinterpret_cast<void **>(&d_rng_states), grid.x * block.x * sizeof(curandState));
		}

		//****** 内存复制 host->Device ******
		cudaMemcpy(d_Width, &ImageWidth, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Height, &ImageHeight, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_world_gpu, &w, sizeof(GPUHitable) * object_count, cudaMemcpyHostToDevice);
		cudaMemcpy(d_materials, &m, sizeof(Material) *material_count, cudaMemcpyHostToDevice);

		
	}
	void IPR_Dispose()
	{
		//****** 释放显存 **********
		cudaFree(d_Width);
		cudaFree(d_Height);
		cudaFree(d_pixeldata);
		cudaFree(d_world_gpu);
		cudaFree(d_materials);
		cudaFree(d_camera);

	}
	cudaError_t IPRRender()
	{
		const int random_seed = drand48() * 1000;
		const auto cuda_status = cudaSetDevice(0);

		//更新数据

		cudaMemcpy(d_camera, &cam, sizeof(Camera), cudaMemcpyHostToDevice);
		cudaMemcpy(d_pixeldata, h_pixeldataF, ImageWidth * ImageHeight * 4 * sizeof(float), cudaMemcpyHostToDevice);
		//******分配线程 ******
		IPRSampler << <grid, block >> > (
			ImageWidth, ImageHeight,
			object_count,
			random_seed,
			SPP,
			MAX_SCATTER_TIME,
			d_world_gpu,
			d_pixeldata,
			d_camera,
			d_rng_states,
			d_materials);

		//****** 复制内存 Device->host ******
		cudaMemcpy(h_pixeldataF, d_pixeldata, ImageWidth * ImageHeight * 4 * sizeof(float), cudaMemcpyDeviceToHost);

		//****** 转换缓冲数据 ****** TODO 可以优化
		for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)
			PixelData[i] = (Range(sqrt(h_pixeldataF[i] / (current_spp == 0 ? SPP : current_spp + SPP))) * 255);
		current_spp += SPP;

		return cuda_status;
	}
}
void ReSetIPR()
{
	if(!Use_IPR)return;
	current_spp = 0; 
	for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)h_pixeldataF[i] = 0;
}
void Render()
{
	if(!Use_IPR&&current_spp!=0)return;
	IPR::IPRRender();
}

void OnMouseMove(int x,int y)
{
	cam_rotation[0] += x / 57.3;
	cam_rotation[1] -= y / 57.30f;
	if (cam_rotation[1] > 3.14 / 2)cam_rotation[1] = 3.14 / 2;
	if (cam_rotation[1] < -3.14 / 2)cam_rotation[1] = -3.14 / 2;

	if (cam_rotation[0] > 2 * 3.1415)cam_rotation[0] = 0;
	if (cam_rotation[0] < 0)cam_rotation[0] = 2 * 3.1415;

	camera_lookat = unit_vector(Vec3(cos(cam_rotation[0]), sin(cam_rotation[1]), sin(cam_rotation[0])));
	cam.Update(cam.Origin(), cam.Origin() + camera_lookat, Vec3(0, 1, 0), 90, float(ImageWidth) / float(ImageHeight));
}

void OnKeyDown()
{
	auto newpos = Vec3(0, 0, 0);
	if (GLWindow::keyDown['w'])newpos += camera_lookat * 0.05f;
	if (GLWindow::keyDown['s'])newpos -= camera_lookat * 0.05f;
	cam.Update(cam.Origin()+ newpos, newpos + cam.Origin() + camera_lookat, Vec3(0, 1, 0), 90, float(ImageWidth) / float(ImageHeight));
}

int main(int argc, char* argv[])
{
	InitData();
	IPR::IPR_Init();
	if(Use_IPR)SPP = IPR_SPP; 
	GLWindow::InitWindow(argc, argv, GLUT_DOUBLE | GLUT_RGBA, 100, 100, ImageWidth, ImageHeight, "ALightGPU");
	IPR::IPR_Dispose();
	return 0;
}


