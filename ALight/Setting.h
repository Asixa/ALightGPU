#pragma once

enum RenderMode
{
	Raster_GL,
	RT_CPU,
	RT_GPU,
	RT_GPU_IPR,
};

namespace Setting
{
	static RenderMode render_mode = RT_GPU_IPR;
	static float FOV = 75;
	static bool IPR = true;
	static int SPP = 1;
	static int argc;
	static char* argv;
	static  int width = 1024/2, height = 1024/2;
	const int BlockSize = 16;
}