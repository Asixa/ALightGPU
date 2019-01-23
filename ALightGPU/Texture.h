#pragma once
#include "vec3.h"


enum TextureType
{
	Constant,Image
};

namespace Texture {

	class Texture {
		TextureType type;
		int w, h;
		unsigned char data[];
	public:
		Texture(unsigned char *pixels, int a, int b, int length);
		virtual Vec3 Value(float u, float v, const Vec3& p) const = 0;
	};

	inline Texture::Texture(unsigned char *pixels, const int a, const int b, const int length):w(a),h(b)
	{
		memcpy(data, pixels, length * sizeof(unsigned char));
	}

	inline Vec3 Texture::Value(const float u, const float v, const Vec3& p) const
	{
		switch (type)
		{
		case Constant:
			return {data[0] / 255.0, data[1] / 255.0, data[2] / 255.0 };
		case Image:
			int i = (u)*w;
			int j = (1 - v)*h - 0.001;
			if (i < 0) i = 0;
			if (j < 0) j = 0;
			if (i > w - 1) i = w - 1;
			if (j > h - 1) j = h - 1;
			const float r = int(data[3 * i + 3 * w*j]) / 255.0;
			const float g = int(data[3 * i + 3 * w*j + 1]) / 255.0;
			const float b = int(data[3 * i + 3 * w*j + 2]) / 255.0;
			return {r, g, b};
			break;
		default: ;
		}
		return {};
	}
}

