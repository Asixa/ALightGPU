#include "Float2Byte.h"

__host__ __device__ float Range(float a, float Small, float Big)
{
	if (a < Small)a = Small;
	else if (a > Big)a = Big;
	return a;
}
__global__ void Float2Byte(int width, int sampled, int spp, float* in, GLbyte* out)
{
	const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
	const auto index = width * 4 * y + x * 4;
	
	
	for (auto i = 0; i < 4; i++)out[index + i] = Range((in[index + i] / sampled), 0, 1) * 255;
}