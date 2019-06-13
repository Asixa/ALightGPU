#include "ONB.h"


#define ONB_EPSILON 0.01f 
	void ONB::initFromU(const float3& u) {
		const auto n = make_float3(1.0f, 0.0f, 0.0f);
		const auto m = make_float3(0.0f, 1.0f, 0.0f);
		U = Float3::UnitVector(u);
		V = Float3::Cross(U, n);
		if (Float3::Length(V) < ONB_EPSILON)
			V = Float3::Cross(U, m);
		W = Float3::Cross(U, V);
	}

	void  ONB::initFromV(const float3& v) {
		const auto n = make_float3(1.0f, 0.0f, 0.0f);
		const auto m = make_float3(0.0f, 1.0f, 0.0f);

		V = Float3::UnitVector(v);
		U = Float3::Cross(V, n);
		if (Float3::SquaredLength(U) < ONB_EPSILON)
			U = Float3::Cross(V, m);
		W = Float3::Cross(U, V);
	}

	void  ONB::initFromW(const float3& w) {
		const auto n = make_float3(1.0f, 0.0f, 0.0f);
		const auto m = make_float3(0.0f, 1.0f, 0.0f);

		W = Float3::UnitVector(w);
		U = Float3::Cross(W, n);
		if (Float3::Length(U) < ONB_EPSILON)
			U = Float3::Cross(W, m);
		V = Float3::Cross(W, U);
	}

	void ONB::initFromUV(const float3& u, const float3& v) {
		U = Float3::UnitVector(u);
		W = Float3::UnitVector(Float3::Cross(u, v));
		V = Float3::Cross(W, U);
	}

	void ONB::initFromVU(const float3& v, const float3& u) {
		V = Float3::UnitVector(v);
		W = Float3::UnitVector(Float3::Cross(u, v));
		U = Float3::Cross(V, W);
	}

	void ONB::initFromUW(const float3& u, const float3& w) {
		U = Float3::UnitVector(u);
		V = Float3::UnitVector(Float3::Cross(w, u));
		W = Float3::Cross(U, V);
	}

	void  ONB::initFromWU(const float3& w, const float3& u) {
		W = Float3::UnitVector(w);
		V = Float3::UnitVector(Float3::Cross(w, u));
		U = Float3::Cross(V, W);
	}

	void  ONB::initFromVW(const float3& v, const float3& w) {
		V = Float3::UnitVector(v);
		U = Float3::UnitVector(Float3::Cross(v, w));
		W = Float3::Cross(U, V);
	}

	void ONB::initFromWV(const float3& w, const float3& v) {
		W = Float3::UnitVector(w);
		U = Float3::UnitVector(Float3::Cross(v, w));
		V = Float3::Cross(W, U);
	}

	bool  operator==(const ONB& o1, const ONB& o2)
	{
		return(o1.u() == o2.u() && o1.v() == o2.v() && o1.w() == o2.w());
	}

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

