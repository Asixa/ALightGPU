#pragma once
#include <cstdlib>

float drand()
{
	return rand() / (0x7fff + 1.0);
}
