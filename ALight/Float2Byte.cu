#include "Float2Byte.h"
#include "Defines.h"

__host__ __device__ float Range(float a, float Small, float Big)
{
	if (a < Small)a = Small;
	else if (a > Big)a = Big;
	return a;
}
__global__ void Float2Byte(bool quick,int width, int sampled, int spp, float* in, GLbyte* out)
{
	const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
	const auto index= width * 4 * y + x * 4;
	if(quick)
	{
		if((x % PREVIEW_PIXEL_SIZE == 0&& y % PREVIEW_PIXEL_SIZE == 0))
			for (auto i = 0; i < 4; i++)out[index + i] = Range((in[index + i] / sampled), 0, 1) * 255;
		else
		{
			const auto a = width * 4 * (y - y % PREVIEW_PIXEL_SIZE) + (x - x % PREVIEW_PIXEL_SIZE) * 4;
			for (auto i = 0; i < 4; i++)out[index + i] = out[a + i];
		}
	}
	else
	{
		for (auto i = 0; i < 4; i++)out[index + i] = Range((in[index + i] / sampled), 0, 1) * 255;
	}
}