#include "ONB.h"
#include "MathHelper.h"


#define ONB_EPSILON 0.01f 
	void ONB::initFromU(const float3& u) {
		const auto n = make_float3(1.0f, 0.0f, 0.0f);
		const auto m = make_float3(0.0f, 1.0f, 0.0f);
		U = normalize(u);
		V = cross(U, n);
		if (length(V) < ONB_EPSILON)
			V = cross(U, m);
		W = cross(U, V);
	}

	void  ONB::initFromV(const float3& v) {
		const auto n = make_float3(1.0f, 0.0f, 0.0f);
		const auto m = make_float3(0.0f, 1.0f, 0.0f);

		V = normalize(v);
		U = cross(V, n);
		if (squaredLength(U) < ONB_EPSILON)
			U = cross(V, m);
		W = cross(U, V);
	}

	void  ONB::initFromW(const float3& w) {
		const auto n = make_float3(1.0f, 0.0f, 0.0f);
		const auto m = make_float3(0.0f, 1.0f, 0.0f);

		W = normalize(w);
		U = cross(W, n);
		if (length(U) < ONB_EPSILON)
			U = cross(W, m);
		V = cross(W, U);
	}

	void ONB::initFromUV(const float3& u, const float3& v) {
		U = normalize(u);
		W = normalize(cross(u, v));
		V = cross(W, U);
	}

	void ONB::initFromVU(const float3& v, const float3& u) {
		V = normalize(v);
		W = normalize(cross(u, v));
		U = cross(V, W);
	}

	void ONB::initFromUW(const float3& u, const float3& w) {
		U = normalize(u);
		V = normalize(cross(w, u));
		W = cross(U, V);
	}

	void  ONB::initFromWU(const float3& w, const float3& u) {
		W = normalize(w);
		V = normalize(cross(w, u));
		U = cross(V, W);
	}

	void  ONB::initFromVW(const float3& v, const float3& w) {
		V = normalize(v);
		U = normalize(cross(v, w));
		W = cross(U, V);
	}

	void ONB::initFromWV(const float3& w, const float3& v) {
		W = normalize(w);
		U = normalize(cross(v, w));
		V = cross(W, U);
	}

	// bool  operator==(const ONB& o1, const ONB& o2)
	// {
	// 	return(o1.u() == o2.u() && o1.v() == o2.v() && o1.w() == o2.w());
	// }

	// std::istream& operator>>(std::istream & is, ONB & t) {
	// 	float3 new_u, new_v, new_w;
	// 	is >> new_u >> new_v >> new_w;
	// 	t.initFromUV(new_u, new_v);
	//
	// 	return is;
	// }
	//
	// std::ostream& operator<<(std::ostream & os, const ONB & t) {
	// 	os << t.u() << "\n" << t.v() << "\n" << t.w() << "\n";
	// 	return os;
	// }

