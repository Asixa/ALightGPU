#pragma once
#include <device_launch_parameters.h>
#include <GL/glew.h>
__host__ __device__ float Range(float a, float Small = 0, float Big = 1);
__global__ void Float2Byte(int width, int sampled, int spp, float* in, GLbyte* out);