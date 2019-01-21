#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <thread>

#include "Hitable.h"
#include "Camera.h"
#include "GLWindow.h"
#include "curand_kernel.h"
#include "MathHelper.h"
#include "BVH.h"
#include "Sphere.h"
#include <list>
#include <vector>
#include "device.h"
#include "Renderer.h"

using namespace std;
using namespace Renderer;

void ReSetIPR()
{
	if(!Use_IPR)return;
	current_spp = 0; 
	for (auto i = 0; i < ImageWidth*ImageHeight * 4; i++)h_pixeldataF[i] = 0;
}

void Render()
{
	if(!Use_IPR&&current_spp!=0)return;

	Renderer::IPRRender();
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
	// Renderer::InitData();
	// cout << " Vec3: " << sizeof(Vec3) << "  AABB: " << sizeof(AABB) << "  NVH: " << sizeof(BVHNode) << "  Hitable: " << sizeof(Hitable) << endl;
	//
	//
	// return 0;
	Renderer::InitData();

	Renderer::Init();
	if(Renderer::Use_IPR)SPP = IPR_SPP; 
	GLWindow::InitWindow(argc, argv, GLUT_DOUBLE | GLUT_RGBA, 100, 100, ImageWidth, ImageHeight, "ALightGPU");
	Renderer::IPR_Dispose();
	return 0;
}


