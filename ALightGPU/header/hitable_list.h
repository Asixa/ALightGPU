#pragma once
#include "Hitable.h"

class HitableList : public Hitable {
public:
	HitableList() {}
	HitableList(Hitable **l, int n) { list = l; list_size = n; }
	virtual bool Hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;
	Hitable **list;
	int list_size;
};

inline bool HitableList::Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
	HitRecord temp_rec;
	auto hit_anything = false;
	double closest_so_far = t_max;
	for (int i = 0; i < list_size; i++) {
		if (list[i]->Hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}