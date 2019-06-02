#pragma once
#include "hitable.h"
#include <algorithm>
#include <vector>


class BVHNode : public Hitable {
public:
	BVHNode(){type = Instance::BVH; }
	BVHNode(Hitable **l, int n, float time0, float time1);
	 __device__ BVHNode(Hitable* data)
	{
		 id = data->id;
		type = Instance::BVH;
		const auto mirror = static_cast<BVHNode*>(data);
		left_id = mirror->left_id;
		right_id= mirror->right_id;
		Box = mirror->Box;
	}
	 __device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec, DeviceData* data) const override;
	__host__  bool BoundingBox(float t0, float t1, AABB& b) const override;
	int Size() const override;
	__device__ int Debug() const override;
	Hitable* GPUPointer() override;
	void SetChildId() override;
	int count() override;
	//__device__ __host__  int type() override;

	int left_id, right_id;
	AABB Box;
	Hitable *Left;
    Hitable *Right;

};


__device__  inline bool BVHNode::BoundingBox(float t0, float t1, AABB& b) const {
	b = Box;
	return true;
}

inline int BVHNode::Size() const
{
	return sizeof(BVHNode);
}

__device__ inline int BVHNode::Debug() const
{
	return 233;
}

inline Hitable* BVHNode::GPUPointer()
{
	
	printf("开始创建BVH的GPU指针\n");




	Hitable* pointer;
	cudaMalloc(&pointer, sizeof(BVHNode));
	cudaMemcpy(pointer, this, sizeof(this), cudaMemcpyHostToDevice);

	return pointer;
}

inline void BVHNode::SetChildId()
{
	left_id = Left->id;
	right_id = Right->id;
}

inline int BVHNode::count()
{
	//printf("left type %d %s\n ", Left->type, typeid(Left).name());
	return Left->count() + Right->count() + 1;
}

// __device__ __host__  inline int BVHNode::type()
// {
// 	return 2;
// }


__device__ bool BVHNode::Hit(const Ray& r, const float t_min, const float t_max, HitRecord& rec,DeviceData* data) const
{
	// printf("检测hit");
	 return false;
	if (Box.Hit(r, t_min, t_max))
	{
		HitRecord left_rec, right_rec;
		const auto hit_left = data->world[left_id]->Hit(r, t_min, t_max, left_rec, data);
		const auto hit_right = data->world[right_id]->Hit(r, t_min, t_max, right_rec, data);
		if (hit_left && hit_right)
		{
			if (left_rec.t < right_rec.t)
				rec = left_rec;
			else
				rec = right_rec;
			return true;
		}
		if (hit_left)
		{
			rec = left_rec;
			return true;
		}
		if (hit_right)
		{
			rec = right_rec;
			return true;
		}
		return false;
	}
	else return false;
}


inline int BoxXCompare(const void * a, const void * b) {
	AABB box_left, box_right;
	Hitable *ah = *(Hitable**)a;
	Hitable *bh = *(Hitable**)b;
	if (!ah->BoundingBox(0, 0, box_left) || !bh->BoundingBox(0, 0, box_right))
		printf("no bounding box in bvh_node constructor\n");
	if (box_left._min.x() - box_right._min.x() < 0.0)
		return -1;
	else
		return 1;
}

inline int BoxYCompare(const void * a, const void * b)
{
	AABB box_left, box_right;
	Hitable *ah = *(Hitable**)a;
	Hitable *bh = *(Hitable**)b;
	if (!ah->BoundingBox(0, 0, box_left) || !bh->BoundingBox(0, 0, box_right))
		printf("no bounding box in bvh_node constructor\n");
	if (box_left._min.y() - box_right._min.y() < 0.0)
		return -1;
	else
		return 1;
}

inline int BoxZCompare(const void * a, const void * b)
{
	AABB box_left, box_right;
	Hitable *ah = *(Hitable**)a;
	Hitable *bh = *(Hitable**)b;
	if (!ah->BoundingBox(0, 0, box_left) || !bh->BoundingBox(0, 0, box_right))
		printf("no bounding box in bvh_node constructor\n");
	if (box_left._min.z() - box_right._min.z() < 0.0)
		return -1;
	else
		return 1;
}


inline BVHNode::BVHNode(Hitable **l, int n, float time0, float time1) {
	type = Instance::BVH;
	const auto axis = int(3 * drand48());
	if (axis == 0)
		qsort(l, n, sizeof(Hitable *), BoxXCompare);
	else if (axis == 1)
		qsort(l, n, sizeof(Hitable *), BoxYCompare);
	else
		qsort(l, n, sizeof(Hitable *), BoxZCompare);
	if (n == 1)
	{
		Left = Right = l[0];
	}
	else if (n == 2)
	{
		Left = l[0];
		Right = l[1];
	}
	else
	{
		Left = new BVHNode(l, n / 2, time0, time1);
		Right = new BVHNode(l + n / 2, n - n / 2, time0, time1);
	}
	AABB box_left, box_right;
	if (!Left->BoundingBox(time0, time1, box_left) || !Right->BoundingBox(time0, time1, box_right))
		printf("no bounding box in bvh_node constructor\n");
	Box =SurroundingBox(box_left, box_right);
}




// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
inline unsigned int ExpandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
inline unsigned int Morton3D(float x, float y, float z)
{
	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	const auto xx = ExpandBits(static_cast<unsigned int>(x));
	const auto yy = ExpandBits(static_cast<unsigned int>(y));
	const auto zz = ExpandBits(static_cast<unsigned int>(z));
	return xx * 4 + yy * 2 + zz;
}
/**
 * a wrapper to calculate morton code from
 * the position of an object inside the
 * unit cube.
 */
inline unsigned int Morton3D(Vec3 pos)
{
	return Morton3D(pos[0], pos[1], pos[2]);
}

bool mortonCompare(Hitable* p1, Hitable* p2)
{
	return p1->MortonCode < p2->MortonCode;
}

void constructBVH(BVHNode* root)
{
	if (root->range == 1) return;

	int gamma = findSplitPosition(root->start, root->start + root->range - 1);

	if (gamma == -1) return;

	int lchildSpan = gamma - root->start + 1;
	BBox lchildBBox = generate_bounding_box(root->start, lchildSpan);
	BVHNode* lchild = new BVHNode(lchildBBox, root->start, lchildSpan);

	int rchildSpan = root->range - lchildSpan;
	BBox rchildBBox = generate_bounding_box(gamma + 1, rchildSpan);
	BVHNode* rchild = new BVHNode(rchildBBox, gamma + 1, rchildSpan);

	root->Left = lchild;
	root->Right = rchild;

	constructBVH(root->Left);
	constructBVH(root->r);
}

inline BVHNode CreateBVH(std::vector<Hitable*>& primitives, int n)
{
	BBox bb;
	for (size_t i = 0; i < primitives.size(); ++i) {
		bb.expand(primitives[i]->get_bbox());
	}
	auto root = new BVHNode(bb, 0, primitives.size());

	for (size_t i = 0; i < primitives.size(); ++i) {
		const unsigned int morton_code = 
			Morton3D(bb.getUnitcubePosOf(primitives[i]->get_bbox().centroid()));
		primitives[i]->MortonCode = morton_code;
	}
	std::sort(primitives.begin(), primitives.end(), mortonCompare);
	constructBVH(root);
}

