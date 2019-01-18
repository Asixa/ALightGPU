#pragma once
#include <Windows.h>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "vec3.h"


#define M_PI 3.1415926
const int  ImageWidth = 512,ImageHeight = 512;
const int SPP = 1024,MAX_SCATTER_TIME = 8;
GLint    PixelLength;
GLbyte* PixelData;
const GLint rgbwidth = ImageWidth * 4;
const int BlockSize=16;
Vec3* col;


__device__ static unsigned long long seed = 4;

// Ëæ»úÊý 
 inline float drand48()
{
	const long long  m = 0x100000000LL, ra = 0x5DEECE66DLL;
	seed = (ra * seed + 0xB16) & 0xFFFFFFFFFFFFLL;
	const unsigned int x = seed >> 16;
	return static_cast<float>(static_cast<double>(x) / static_cast<double>(m));
}

 __device__ inline float drand()
{
	const long long  m = 0x100000000LL, ra = 0x5DEECE66DLL;
	seed = (ra * seed + 0xB16) & 0xFFFFFFFFFFFFLL;
	const unsigned int x = seed >> 16;
	return static_cast<float>(static_cast<double>(x) / static_cast<double>(m));
}

 __device__ inline float drand(unsigned long long* s)
 {
	 *s = ((0x5DEECE66DLL * *s + 0xB16) & 0xFFFFFFFFFFFFLL);
	 return static_cast<float>(static_cast<double>(*s >> 16) / static_cast<double>(0x100000000LL));
 }