#pragma once
#include <Windows.h>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "vec3.h"
#define MATERIAL_PARAMTER_COUNT 6
#define HITABLE_PARAMTER_COUNT 5

#define TEXTURE_COUNT 3

// #define LAMBERTIAN 1
// #define METAL 2
// #define DIELECTIRC 3
#define M_PI 3.1415926
const int  ImageWidth = 1280,ImageHeight = 720;
int SPP = 1024,MAX_SCATTER_TIME = 8;
const int IPR_SPP = 4; int current_spp = 0;
GLint    PixelLength;
GLbyte* PixelData;
const GLint rgbwidth = ImageWidth * 4;
const int BlockSize=16;
Vec3* col;


//texture<unsigned char, 2, cudaReadModeElementType> tex;

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

 /*
  *	
compute_30,sm_30
compute_35,sm_35
compute_37,sm_37
compute_50,sm_50
compute_52,sm_52
compute_60,sm_60
compute_61,sm_61
compute_70,sm_70
compute_75,sm_75
  */


