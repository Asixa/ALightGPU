#pragma once
#include "float3Extension.h"
#include <vector_functions.hpp>
#include <istream>


	class ONB
	{
	public:
		__device__ ONB() {};

		__device__ ONB(const float3& a, const float3& b, const float3& c)
		{
			U = a; V = b; W = c;
		}

		__device__ void initFromU(const float3& u);
		__device__ void initFromV(const float3& v);
		__device__ void initFromW(const float3& w);

		__device__ void set(const float3& a, const float3& b, const float3& c)
		{
			U = a; V = b; W = c;
		}

		// Calculate the ONB from two vectors
		// The first one is the Fixed vector (it is just normalized)
		// The second is normalized and its direction can be adjusted
		__device__ void  initFromUV(const float3& u, const float3& v);
		__device__ void  initFromVU(const float3& v, const float3& u);

		__device__ void  initFromUW(const float3& u, const float3& w);
		__device__ void  initFromWU(const float3& w, const float3& u);

		__device__ void  initFromVW(const float3& v, const float3& w);
		__device__ void  initFromWV(const float3& w, const float3& v);

		__device__ friend std::istream& operator>>(std::istream& is, ONB& t);
		__device__ friend std::ostream& operator<<(std::ostream& os, const ONB& t);
		__device__ friend bool  operator==(const ONB& o1, const ONB& o2);

		// Get a component from the ONB basis
		__device__ float3 u() const { return U; }
		__device__ float3 v() const { return V; }
		__device__ float3 w() const { return W; }

		float3 U, V, W;
	};
