#pragma once
#include "hitable.h"

class BVHNode : public Hitable {
public:
	BVHNode() {}
	__device__  BVHNode(Hitable **l, int n, float time0, float time1);
	__device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
	__device__  bool BoundingBox(float t0, float t1, AABB& b) const override;
	Hitable *Left;
	Hitable *Right;
	AABB Box;
};


__device__  inline bool BVHNode::BoundingBox(float t0, float t1, AABB& b) const {
	b = Box;
	return true;
}

__device__ bool BVHNode::Hit(const Ray& r, const float t_min, const float t_max, HitRecord& rec) const {
	if (Box.Hit(r, t_min, t_max)) {
		HitRecord left_rec, right_rec;
		const auto hit_left = Left->Hit(r, t_min, t_max, left_rec);
		const auto hit_right = Right->Hit(r, t_min, t_max, right_rec);
		if (hit_left && hit_right) {
			if (left_rec.t < right_rec.t)
				rec = left_rec;
			else
				rec = right_rec;
			return true;
		}
		if (hit_left) {
			rec = left_rec;
			return true;
		}
		if (hit_right) {
			rec = right_rec;
			return true;
		}
		return false;
	}
	else return false;
}


__device__  inline int BoxXCompare(const void * a, const void * b) {
	AABB box_left, box_right;
	Hitable *ah = *(Hitable**)a;
	Hitable *bh = *(Hitable**)b;
	if (!ah->BoundingBox(0, 0, box_left) || !bh->BoundingBox(0, 0, box_right))
		std::cerr << "no bounding box in bvh_node constructor\n";
	if (box_left._min.x() - box_right._min.x() < 0.0)
		return -1;
	else
		return 1;
}

__device__  inline int BoxYCompare(const void * a, const void * b)
{
	AABB box_left, box_right;
	Hitable *ah = *(Hitable**)a;
	Hitable *bh = *(Hitable**)b;
	if (!ah->BoundingBox(0, 0, box_left) || !bh->BoundingBox(0, 0, box_right))
		std::cerr << "no bounding box in bvh_node constructor\n";
	if (box_left._min.y() - box_right._min.y() < 0.0)
		return -1;
	else
		return 1;
}

__device__  inline int BoxZCompare(const void * a, const void * b)
{
	AABB box_left, box_right;
	Hitable *ah = *(Hitable**)a;
	Hitable *bh = *(Hitable**)b;
	if (!ah->BoundingBox(0, 0, box_left) || !bh->BoundingBox(0, 0, box_right))
		std::cerr << "no bounding box in bvh_node constructor\n";
	if (box_left._min.z() - box_right._min.z() < 0.0)
		return -1;
	else
		return 1;
}


__device__  inline BVHNode::BVHNode(Hitable **l, int n, float time0, float time1) {
	int axis = int(3 * drand48());
	if (axis == 0)
		qsort(l, n, sizeof(Hitable *), BoxXCompare);
	else if (axis == 1)
		qsort(l, n, sizeof(Hitable *), BoxYCompare);
	else
		qsort(l, n, sizeof(Hitable *), BoxZCompare);
	if (n == 1) {
		Left = Right = l[0];
	}
	else if (n == 2) {
		Left = l[0];
		Right = l[1];
	}
	else {
		Left = new BVHNode(l, n / 2, time0, time1);
		Right = new BVHNode(l + n / 2, n - n / 2, time0, time1);
	}
	AABB box_left, box_right;
	if (!Left->BoundingBox(time0, time1, box_left) || !Right->BoundingBox(time0, time1, box_right))
		std::cerr << "no bounding box in bvh_node constructor\n";
	Box =SurroundingBox(box_left, box_right);
}
