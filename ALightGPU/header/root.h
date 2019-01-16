#pragma once
#include <Windows.h>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "vec3.h"
const GLint  ImageWidth = 512;
const GLint  ImageHeight = 512;
const int SPP = 64;
GLint    PixelLength;
GLbyte* PixelData;
const GLint SamplingRate = 1000;
const GLint rgbwidth = ImageWidth * 4;
Vec3* col;


static unsigned long long seed = 4;

// Ëæ»úÊý 
inline float drand48()
{
	const long long  m = 0x100000000LL, ra = 0x5DEECE66DLL;
	seed = (ra * seed + 0xB16) & 0xFFFFFFFFFFFFLL;
	const unsigned int x = seed >> 16;
	return static_cast<float>(static_cast<double>(x) / static_cast<double>(m));
}