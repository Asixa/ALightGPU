#pragma once
#include "vec3.h"
#include "ray.h"
#include <valarray>
#include "Material.h"
#include "AABB.h"

inline void GetSphereUv(const Vec3& p, float& u, float& v) {
	const auto phi = atan2(p.z(), p.x());
	const auto theta = asin(p.y());
	u = 1 - (phi + M_PI) / (2 * M_PI);
	v = (theta + M_PI / 2) / M_PI;
}
struct  DeviceData;
namespace Instance
{
	const int BVH = 2,SPHERE=1,TRIANGLE=3;
}
class Hitable {
public:
	virtual ~Hitable() = default;
	
	__device__ virtual bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec, DeviceData* data) const = 0;
	__host__  virtual bool BoundingBox(float t0, float t1, AABB& box) const = 0;
	__host__  virtual int Size() const = 0;
	__device__  virtual int Debug() const = 0;
	__host__ virtual Hitable* GPUPointer() { return nullptr; }
	__host__ virtual int count() { return 0; }
	__host__ virtual void SetChildId() { }
	//__device__ __host__  virtual int type() { return 0;}
	int id; int type;
}; 
struct  DeviceData
{
	Hitable** world;
	Material* materials;
	cudaTextureObject_t texs[TEXTURE_COUNT];
};
