// #include "RayMarchSampler.h"
// #include <crt/host_defines.h>
// #include "float3Extension.h"
// #include "math.h"
// #include <vector_functions.hpp>
// #include "float4x4.h"
//
// // #include "Ray.h"
// // #include "Camera.h"
// // #include "float3Extension.h"
// // #include <curand_discrete2.h>
// // #include <device_launch_parameters.h>
// // #include <curand_kernel.h>
// // #include <cstdio>
// // #include "RayHit.h"
// // #include "Objects.h"
// // #include "float3x3.h"
// // #include "Float2Byte.h"
// // #include "float2Extension.h"
// namespace RayMatch
// {
// #define EPSILON 0.001f
// #define MAX_DST  200
// #define MAX_STEP_COUNT  250
//
// 	struct Data
// 	{
// 		float power;
// 		float darkness;
// 		float blackAndWhite;
// 		float3 colourAMix;
// 		float3 colourBMix;
// 		float4x4 _CameraToWorld;
// 		float4x4 _CameraInverseProjection;
// 		float3 _LightDirection;
//
// 	};
//
// 	struct Ray {
// 		float3 origin;
// 		float3 direction;
// 	};
// 	Ray CreateRay(float3 origin, float3 direction) {
// 		Ray ray;
// 		ray.origin = origin;
// 		ray.direction = direction;
// 		return ray;
// 	}
// 	Ray CreateCameraRay(float2 uv) {
// 		float3 origin = mul(_CameraToWorld, float4(0, 0, 0, 1)).xyz;
// 		float3 direction = mul(_CameraInverseProjection, float4(uv, 0, 1)).xyz;
// 		direction = mul(_CameraToWorld, float4(direction, 0)).xyz;
// 		direction = normalize(direction);
// 		return CreateRay(origin, direction);
// 	}
//
// 	float2 SceneInfo(float3 position,Data data) {
// 		auto z = position;
// 		float dr = 1.0;
// 		float r = 0.0;
// 		auto iterations = 0;
//
// 		for (auto i = 0; i < 15; i++) {
// 			iterations = i;
// 			r = length (z);
//
// 			if (r > 2) {
// 				break;
// 			}
//
// 			// convert to polar coordinates
// 			float theta = acos(z.z / r);
// 			float phi = atan2(z.y, z.x);
// 			dr = pow(r, data.power - 1.0) * data.power * dr + 1.0;
//
// 			// scale and rotate the point
// 			const float zr = pow(r, data.power);
// 			theta = theta * data.power;
// 			phi = phi * data.power;
//
// 			// convert back to cartesian coordinates
// 			z = zr * make_float3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
// 			z =z+ position;
// 		}
// 		float dst = 0.5 * log(r) * r / dr;
// 		return make_float2(iterations, dst * 1);
// 	}
//
// 	float3 EstimateNormal(float3 p,Data data) {
// 		float x = SceneInfo(make_float3(p.x + EPSILON, p.y, p.z), data).y - SceneInfo(make_float3(p.x - EPSILON, p.y, p.z), data).y;
// 		float y = SceneInfo(make_float3(p.x, p.y + EPSILON, p.z), data).y - SceneInfo(make_float3(p.x, p.y - EPSILON, p.z), data).y;
// 		float z = SceneInfo(make_float3(p.x, p.y, p.z + EPSILON), data).y - SceneInfo(make_float3(p.x, p.y, p.z - EPSILON), data).y;
// 		return normalize(make_float3(x, y, z));
// 	}
//
// 	__global__ void RayMatchSampler()
// 	{
// 		int width, height;
// 		float2 uv = id.xy / make_float2(width, height);
//
// 		auto data = Data();
//
// 		// Background gradient
// 		float4 result = lerp(make_float4(51, 3, 20, 1), make_float4(16, 6, 28, 1), uv.y) / 255.0;
//
// 		// Raymarching:
// 		Ray ray = CreateCameraRay(uv * 2 - 1);
// 		float rayDst = 0;
// 		int marchSteps = 0;
//
// 		while (rayDst < MAX_DST && marchSteps < MAX_STEP_COUNT) {
// 			marchSteps++;
// 			float2 sceneInfo = SceneInfo(ray.origin,data);
// 			float dst = sceneInfo.y;
//
// 			// Ray has hit a surface
// 			if (dst <= EPSILON) {
// 				float escapeIterations = sceneInfo.x;
// 				float3 normal = EstimateNormal(ray.origin - ray.direction * EPSILON * 2);
//
// 				float colourA = saturate(dot(normal * .5 + .5, -data._LightDirection));
// 				float colourB = saturate(escapeIterations / 16.0);
// 				float3 colourMix = saturate(colourA * data.colourAMix + colourB * data.colourBMix);
//
// 				result = make_float4(colourMix.x,colourMix.y, colourMix.z, 1);
// 				break;
// 			}
// 			ray.origin = ray.origin+ ray.direction * dst;
// 			rayDst += dst;
// 		}
//
// 		float rim = marchSteps / data.darkness;
// 		//Destination[id.xy] = lerp(result, 1, blackAndWhite) * rim;
// 	}
//
// }