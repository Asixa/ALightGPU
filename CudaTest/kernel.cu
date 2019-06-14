
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


class T
{
public:
	float v;
	T(float a) :v(a){}
};
T& operator-(T x, const T v)
{
	return  T(x.v - v.v);
}
void operator-=(T& x, const T v)
{
	x=x - v;
}

T T::operator-(void)
{
	return  T(-v.v);
}
int main()
{
	T a(1),b(2);
	(a -= b);
	printf("%f,%f,%f\n", -T(5), b.v,(a-b).v);


}
