#pragma once
float Range(float a, float Small = 0, float Big = 1)
{
	if (a < Small)a = Small;
	else if (a > Big)a = Big;
	return a;
}