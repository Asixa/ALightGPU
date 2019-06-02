#pragma once

// static unsigned long long seed = 4;
// float drand48()
// {
// 	const long long  m = 0x100000000LL, ra = 0x5DEECE66DLL;
// 	seed = (ra * seed + 0xB16) & 0xFFFFFFFFFFFFLL;
// 	const unsigned int x = seed >> 16;
// 	return static_cast<float>(static_cast<double>(x) / static_cast<double>(m));
// }