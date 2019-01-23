
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

texture<float, cudaTextureType2DLayered> tex;
__global__ void transformKernel(float* output,
                                int width, int height,
                                float theta)
{
	// Calculate normalized texture coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / static_cast<float>(width);
	float v = y / static_cast<float>(height);

	// Transform coordinates
	u -= 0.5f;
	v -= 0.5f;
	float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
	float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

	// Read from texture and write to global memory
	output[y * width + x] = tex2DLayered(tex, tu, tv,0);
	printf("%d,%d [%f,%f]  :%f,%f,%f\n", x, y,tu,tv, tex2DLayered(tex, tu, tv, 0), tex2DLayered(tex, tu, tv, 1), tex2DLayered(tex, tu, tv, 2));
}

// Host code
int main()
{
	int width, height, depth;
	const auto tex_data = stbi_load("D:/Codes/Projects/Academic/ComputerGraphic/ALightGPU/x64/Release/wall.jpg",
		&width, &height, &depth, 0);
	const auto size = width * height * depth;
	float* h_data = new float[size];
	printf("Hello %d,%d,%d",width,height,depth);
	//for (auto i = 0; i < size; i++)h_data[i] = tex_data[i] / 255.0;
	for (auto i=0;i<size;i++)h_data[i] = tex_data[i] /255.0;
	


	// Allocate device memory for result
	float *dData = NULL;
	cudaMalloc((void **)&dData, size);

	// Allocate array and copy image data
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *cuArray;
	cudaMallocArray(&cuArray,
		&channelDesc,
		width,
		height);
	cudaMemcpyToArray(cuArray,
		0,
		0,
		h_data,
		size,
		cudaMemcpyHostToDevice);

	// Set texture parameters
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = true;    // access with normalized texture coordinates

	// Bind the array to the texture
	cudaBindTextureToArray(tex, cuArray, channelDesc);




	// Invoke kernel
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
	             (height + dimBlock.y - 1) / dimBlock.y);
	transformKernel << <dimGrid, dimBlock >> > (dData, width, height,
	                                            0.5f);

	// Destroy texture object

	// Free device memory
	cudaFreeArray(cuArray);

	return 0;
}