#pragma once
#include "vec3.h"
#include "ray.h"

__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

class AABB {
public:
	__host__ __device__ AABB(){}
	__host__ AABB(const Vec3& a, const Vec3& b);
	__device__ bool Hit(const Ray& r, float tmin, float tmax) const;
	Vec3 _min;
	Vec3 _max;
};

inline AABB::AABB(const Vec3& a, const Vec3& b)
{
	_min = a; _max = b;
}

inline bool AABB::Hit(const Ray& r, float tmin, float tmax) const
{
	for (auto a = 0; a < 3; a++) {
		const auto t0 = ffmin((_min[a] - r.Origin()[a]) / r.Direction()[a],
			(_max[a] - r.Origin()[a]) / r.Direction()[a]);
		const auto t1 = ffmax((_min[a] - r.Origin()[a]) / r.Direction()[a],
			(_max[a] - r.Origin()[a]) / r.Direction()[a]);
		tmin = ffmax(t0, tmin);
		tmax = ffmin(t1, tmax);
		if (tmax <= tmin)
			return false;
	}
	return true;
}

inline AABB SurroundingBox(AABB box0, AABB box1) {
	const Vec3 _small(fmin(box0._min.x(), box1._min.x()),
		fmin(box0._min.y(), box1._min.y()),
		fmin(box0._min.z(), box1._min.z()));
	const Vec3 big(fmax(box0._max.x(), box1._max.x()),
		fmax(box0._max.y(), box1._max.y()),
		fmax(box0._max.z(), box1._max.z()));
	return AABB(_small, big);
}




struct BBox {

	Vec3 _max;	    ///< min corner of the bounding box
	Vec3 _min;	    ///< max corner of the bounding box
	Vec3 extent;  ///< extent of the bounding box (min -> max)

	/**
	 * Constructor.
	 * The default constructor creates a new bounding box which contains no
	 * points.
	 */
	BBox() {
		_max = Vec3(-INF_D, -INF_D, -INF_D);
		_min = Vec3(INF_D, INF_D, INF_D);
		extent = _max - _min;
	}

	/**
	 * Constructor.
	 * Creates a bounding box that includes a single point.
	 */
	BBox(const Vec3& p) : _min(p), _max(p) { extent = _max - _min; }

	/**
	 * Constructor.
	 * Creates a bounding box with given bounds.
	 * \param min the min corner
	 * \param max the max corner
	 */
	BBox(const Vec3& min, const Vec3& max) :
		_min(min), _max(max) {
		extent = max - min;
	}

	/**
	 * Constructor.
	 * Creates a bounding box with given bounds (component wise).
	 */
	BBox(const double minX, const double minY, const double minZ,
		const double maxX, const double maxY, const double maxZ) {
		_min = Vec3(minX, minY, minZ);
		_max = Vec3(maxX, maxY, maxZ);
		extent = _max - _min;
	}

	/**
	 * Expand the bounding box to include another (union).
	 * If the given bounding box is contained within *this*, nothing happens.
	 * Otherwise *this* is expanded to the minimum volume that contains the
	 * given input.
	 * \param bbox the bounding box to be included
	 */
	__host__ __device__
		void expand(const BBox& bbox) {
		_min[0] = ffmin(_min.x(), bbox._min.x());
		_min[1] = ffmin(_min.y(), bbox._min.y());
		_min[2] = ffmin(_min.z(), bbox._min.z());
		_max[0] = ffmax(_max.x(), bbox._max.x());
		_max[1] = fmax(_max.y(), bbox._max.y());
		_max[2] = ffmax(_max.z(), bbox._max.z());
		extent[0] = _max.x - _min.x;
		extent[1] = _max.y - _min.y;
		extent[2] = _max.z - _min.z;
	}

	/**
	 * Expand the bounding box to include a new point in space.
	 * If the given point is already inside *this*, nothing happens.
	 * Otherwise *this* is expanded to a minimum volume that contains the given
	 * point.
	 * \param p the point to be included
	 */
	void expand(const Vec3& p) {
		_min[0] = ffmin(_min.x(), p.x());
		_min[1] = ffmin(_min.y(), p.y());
		_min[2] = ffmin(_min.z(), p.z());
		_max[0] = ffmax(_max.x(), p.x());
		_max[1] = ffmax(_max.y(), p.y());
		_max[2] = ffmax(_max.z(), p.z());
		extent = _max - _min;
	}

	Vec3 centroid() const {
		return (_min + _max) / 2;
	}

	/**
	 * Compute the surface area of the bounding box.
	 * \return surface area of the bounding box.
	 */
	float surface_area() const {
		if (empty()) return 0.0;
		return 2 * (extent.x() * extent.z() +
			extent.x() * extent.y() +
			extent.y() * extent.z());
	}

	/**
	 * Compute the maximum dimension of the bounding box (x, y, or z).
	 * \return 0 if max dimension is x,
	 *         1 if max dimension is y,
	 *         2 if max dimension is z
	 * TODO: replace with enum (or #define)
	 *  - sure but please make sure indexing with the returned value still works
	 */
	uint8_t max_dimension() const {
		uint8_t d = 0;
		if (extent.y > extent.x) d = 1;
		if (extent.z > extent.y) d = 2;
		return d;
	}

	/**
	 * Check if bounding box is empty.
	 * Bounding box that has no size is considered empty. Note that since
	 * bounding box are used for objects with positive volumes, a bounding
	 * box of zero size (empty, or contains a single vertex) are considered
	 * empty.
	 */
	bool empty() const {
		return _min.x() > _max.x() || _min.y() > _max.y() || _min.z() > _max.z();
	}

	/**
	 * Ray - bbox intersection.
	 * Intersects ray with bounding box, does not store shading information.
	 * \param r the ray to intersect with
	 * \param t0 lower bound of intersection time
	 * \param t1 upper bound of intersection time
	 */
	bool intersect(const Ray& r, double& t0, double& t1) const;

	/**
	 * Draw box wireframe with OpenGL.
	 * \param c color of the wireframe
	 */
	// void draw(Color c) const;

	/**
	 * Calculate and return an object's
	 * normalized position in the unit
	 * cube defined by this BBox. if the
	 * object is not inside of the BBox, its
	 * position will be clamped into the BBox.
	 *
	 * \param pos the position to be evaluated
	 * \return the normalized position in the unit
	 * cube, with x,y,z ranging from [0,1]
	 */
	Vec3 getUnitcubePosOf(Vec3 pos)
	{
		const auto o2pos = pos - _min;
		if (!extent.isZero())
		{
			const auto normalized_pos = o2pos / extent;
			return normalized_pos;
		}
		else
		{
			return Vec3();
		}
	}

};

std::ostream& operator<<(std::ostream& os, const BBox& b);

