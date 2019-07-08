//***************************
// https://github.com/shiinamiyuki
//*********************************


#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include <atomic>
#include <mutex>
#define M_PI 3.1415926535897932384626433832795

float random(unsigned int* rng) {
	*rng = (1103515245 * (*rng) + 12345);
	return (float)* rng / (float)0xFFFFFFFF;
}


struct Vec {        // Usage: time ./smallpt 5000 && xv image.ppm
	double x, y, z;                  // position, also color (r,g,b)
	Vec(double x_ = 0, double y_ = 0, double z_ = 0) {
		x = x_;
		y = y_;
		z = z_;
	}

	Vec operator+(const Vec& b) const { return Vec(x + b.x, y + b.y, z + b.z); }

	Vec operator-(const Vec& b) const { return Vec(x - b.x, y - b.y, z - b.z); }

	Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }

	Vec mult(const Vec& b) const { return Vec(x * b.x, y * b.y, z * b.z); }

	Vec& norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }

	double dot(const Vec& b) const { return x * b.x + y * b.y + z * b.z; } // cross:
	Vec operator%(Vec& b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};

struct Ray {
	Vec o, d;

	Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

enum Refl_t {
	DIFF, SPEC, REFR
};  // material types, used in radiance()
struct Sphere {
	double rad;       // radius
	Vec p, e, c;      // position, emission, color
	Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
		rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}

	double intersect(const Ray& r) const { // returns distance, 0 if nohit
		Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		double t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
		if (det < 0) return 0; else det = sqrt(det);
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
	}
};

Sphere spheres[] = {//Scene: radius, position, emission, color, material
		Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFF),//Left
		Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF),//Rght
		Sphere(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFF),//Back
		Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFF),//Frnt
		Sphere(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF),//Botm
		Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF),//Top
		Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * .999, SPEC),//Mirr
		Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * .999, REFR),//Glas
		Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(), DIFF) //Lite
};

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

inline double removeNaN(double x) { return std::isnan(x) || x < 0.0 ? 0.0 : x; }

inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

inline bool intersect(const Ray& r, double& t, int& id) {
	double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
	for (int i = int(n); i--;)
		if ((d = spheres[i].intersect(r)) && d < t) {
			t = d;
			id = i;
		}
	return t < inf;
}

struct PrimarySample {
	double value;
	double _backup;
	uint64_t lastModificationIteration;
	uint64_t lastModifiedBackup;

	void backup() {
		_backup = value;
		lastModifiedBackup = lastModificationIteration;
	}

	void restore() {
		value = _backup;
		lastModificationIteration = lastModifiedBackup;
	}
};


const double largeStepProb = 0.25;

struct RadianceRecord {
	int x, y;
	Vec Li;

	RadianceRecord() {
		x = y = 0;
	};
};

struct Sampler {
	unsigned int seed;
	std::vector<PrimarySample> X;
	uint64_t currentIteration = 0;
	bool largeStep = true;
	uint64_t lastLargeStepIteration = 0;
	int w, h;
	RadianceRecord current;

	Sampler(int w, int h, unsigned int seed) : w(w), h(h), seed(seed) {}

	uint32_t sampleIndex = 0;
	uint64_t a = 0, r = 0;

	void startIteration() {
		sampleIndex = 0;
		currentIteration++;
		largeStep = uniform() < largeStepProb;
	}

	double uniform() {
		return random(&seed);
	}

	void mutate(PrimarySample& Xi, int sampleIndex) {
		double s1, s2;
		if (sampleIndex >= 2) {
			s1 = 1.0 / 1024.0, s2 = 1.0 / 64.0;
		}
		else if (sampleIndex == 1) {
			s1 = 1.0 / h, s2 = 0.1;
		}
		else {
			s1 = 1.0 / w, s2 = 0.1;
		}
		if (Xi.lastModificationIteration < lastLargeStepIteration) {
			Xi.value = uniform();
			Xi.lastModificationIteration = lastLargeStepIteration;
		}

		if (largeStep) {
			Xi.backup();
			Xi.value = uniform();
		}
		else {
			int64_t nSmall = currentIteration - Xi.lastModificationIteration;

			auto nSmallMinus = nSmall - 1;
			if (nSmallMinus > 0) {
				auto x = Xi.value;
				while (nSmallMinus > 0) {
					nSmallMinus--;
					x = mutate(x, s1, s2);
				}
				Xi.value = x;
				Xi.lastModificationIteration = currentIteration - 1;
			}
			Xi.backup();
			Xi.value = mutate(Xi.value, s1, s2);
		}

		Xi.lastModificationIteration = currentIteration;
	}

	double next() {
		if (sampleIndex >= X.size()) {
			X.resize(sampleIndex + 1u);
		}
		auto& Xi = X[sampleIndex];
		mutate(Xi, sampleIndex);
		sampleIndex += 1;
		return Xi.value;

	}

	double mutate(double x, double s1, double s2) {
		double r = uniform();
		if (r < 0.5) {
			r = r * 2.0;
			x = x + s2 * exp(-log(s2 / s1) * r);
			if (x > 1.0) x -= 1.0;
		}
		else {
			r = (r - 0.5) * 2.0;
			x = x - s2 * exp(-log(s2 / s1) * r);
			if (x < 0.0) x += 1.0;
		}
		return x;
	}

	void accept() {
		if (largeStep) {
			lastLargeStepIteration = currentIteration;
		}
		a++;
	}

	void reject() {
		for (PrimarySample& Xi : X) {
			if (Xi.lastModificationIteration == currentIteration) {
				Xi.restore();
			}
		}
		r++;
		--currentIteration;
	}
};

Vec radiance(Ray r, Sampler& sampler) {
	double t;                               // distance to intersection
	int id = 0;                               // id of intersected object
	Vec cl(0, 0, 0);   // accumulated color
	Vec cf(1, 1, 1);  // accumulated reflectance
	int depth = 0;
	while (true) {
		double u1 = sampler.next(), u2 = sampler.next(), u3 = sampler.next();
		if (!intersect(r, t, id)) return cl; // if miss, return black
		const Sphere& obj = spheres[id];        // the hit object
		Vec x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
		double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
		cl = cl + cf.mult(obj.e);
		if (++depth > 5) if (u3 < p) f = f * (1 / p); else { return cl; } //R.R.
		cf = cf.mult(f);
		if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection
			double r1 = 2 * M_PI * u1, r2 = u2, r2s = sqrt(r2);
			Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
			Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
			//return obj.e + f.mult(radiance(Ray(x,d),depth,Xi));
			r = Ray(x, d);
			continue;
		}
		else if (obj.refl == SPEC) {           // Ideal SPECULAR reflection
		 //return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi));
			r = Ray(x, r.d - n * 2 * n.dot(r.d));
			continue;
		}
		Ray reflRay(x, r.d - n * 2 * n.dot(r.d));     // Ideal dielectric REFRACTION
		bool into = n.dot(nl) > 0;                // Ray from outside going in?
		double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
		if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) {    // Total internal reflection
			//return obj.e + f.mult(radiance(reflRay,depth,Xi));
			r = reflRay;
			continue;
		}
		Vec tdir = (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
		double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
		double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
		// return obj.e + f.mult(sampler.next()<P ?
		//                       radiance(reflRay,    depth,Xi)*RP:
		//                       radiance(Ray(x,tdir),depth,Xi)*TP);
		if (u1 < P) {
			cf = cf * RP;
			r = reflRay;
		}
		else {
			cf = cf * TP;
			r = Ray(x, tdir);
		}
	}
}

Vec radiance(int x, int y, int w, int h, Sampler& sampler) {
	Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir
	Vec cx = Vec(w * .5135 / h), cy = (cx % cam.d).norm() * .5135;
	double r1 = 2 * sampler.next(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
	double r2 = 2 * sampler.next(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
	Vec d = cx * (((1 + dx) / 2 + x) / w - .5) +
		cy * (((1 + dy) / 2 + y) / h - .5) + cam.d;
	return radiance(Ray(cam.o + d * 140, d.norm()), sampler);
}


RadianceRecord radiance(Sampler& sampler, bool bootstrap) {
	if (!bootstrap)
		sampler.startIteration();
	double x = sampler.next();
	double y = sampler.next();
	RadianceRecord record;
	record.x = std::min<int>(sampler.w - 1, lround(x * sampler.w));
	record.y = std::min<int>(sampler.h - 1, lround(y * sampler.h));
	record.Li = radiance(record.x, record.y, sampler.w, sampler.h, sampler);
	return record;
}

double b;

double ScalarContributionFunction(const Vec& Li) {
	return 0.2126 * Li.x + 0.7152 * Li.y + 0.0722 * Li.z;
}

void RunMarkovChain(Sampler& sampler, RadianceRecord& r1, RadianceRecord& r2) {
	auto r = radiance(sampler, false);
	double accept = std::max(0.0,
		std::min(1.0,
			ScalarContributionFunction(r.Li) /
			ScalarContributionFunction(sampler.current.Li)));
	double weight1 = (accept + (sampler.largeStep ? 1.0 : 0.0))
		/ (ScalarContributionFunction(r.Li) / b + largeStepProb);
	double weight2 = (1 - accept)
		/ (ScalarContributionFunction(sampler.current.Li) / b + largeStepProb);
	r1.x = r.x;
	r1.y = r.y;
	r1.Li = r.Li * weight1;
	r2.x = sampler.current.x;
	r2.y = sampler.current.y;
	r2.Li = sampler.current.Li * weight2;
	if (accept == 1 || sampler.uniform() < accept) {
		sampler.accept();
		sampler.current = r;
	}
	else {
		sampler.reject();
	}
}


uint32_t nBootstrap = 100000;

inline uint64_t floatToBits(double f) {
	uint64_t ui;
	memcpy(&ui, &f, sizeof(double));
	return ui;
}

inline double bitsToFloat(uint64_t ui) {
	double f;
	memcpy(&f, &ui, sizeof(uint64_t));
	return f;
}

class AtomicFloat {
public:
	AtomicFloat(double v = 0) { bits = floatToBits(v); }

	AtomicFloat(const AtomicFloat& rhs) {
		bits.store(rhs.bits.load(std::memory_order_relaxed), std::memory_order_relaxed);
	}

	operator double() const { return bitsToFloat(bits); }

	double operator=(double v) {
		bits = floatToBits(v);
		return v;
	}

	AtomicFloat& operator=(const AtomicFloat& rhs) {
		bits.store(rhs.bits.load(std::memory_order_relaxed), std::memory_order_relaxed);
		return *this;
	}

	void add(double v) {
		uint64_t oldBits = bits, newBits;
		do {
			newBits = floatToBits(bitsToFloat(oldBits) + v);
		} while (!bits.compare_exchange_weak(oldBits, newBits));
	}

	void store(double v) {
		bits.store(floatToBits(v), std::memory_order_relaxed);
	}

private:
	std::atomic<uint64_t> bits;


};

struct AtomicVec {
	AtomicFloat x, y, z;

	void splat(const Vec& c) {
		x.add(c.x);
		y.add(c.y);
		z.add(c.z);
	}
};

int main(int argc, char* argv[]) {
	int w = 1024, h = 768, samps = argc == 2 ? atoi(argv[1]) : 4; // # samples

	samps =512;

	uint32_t nChains = 2048;
	uint32_t nMutations = std::ceil(double(w) * h * samps / nChains);

	std::vector<uint32_t> seeds;
	for (int i = 0; i < nBootstrap; i++) {
		seeds.emplace_back(rand());
	}
	std::vector<double> weights;
	for (int i = 0; i < nBootstrap; i++) {
		Sampler sampler(w, h, seeds[i]);
		weights.emplace_back(ScalarContributionFunction(radiance(sampler, true).Li));
	}
	std::vector<double> cdf;
	cdf.emplace_back(0);
	for (auto& i : weights) {
		cdf.emplace_back(cdf.back() + i);
	}
	b = cdf.back() / nBootstrap;
	printf("nChains = %d, nMutations = %d\nb = %lf\n", nChains, nMutations, b);

	std::vector<AtomicVec> c(w * h);
	std::atomic<uint64_t> totalMutations(0);
	unsigned int mainSeed = rand();
	auto write = [&](const RadianceRecord& record) {
		auto& r = record.Li;
		int i = (h - record.y - 1) * w + record.x;
		c[i].splat(Vec(removeNaN(r.x), removeNaN(r.y), removeNaN(r.z)));
	};
	std::mutex mutex;
	int32_t count = 0;
#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < nChains; i++) {
		double r = random(&mainSeed) * cdf.back();
		int k = 1;
		for (; k <= nBootstrap; k++) {
			if (cdf[k - 1] < r && r <= cdf[k]) {
				break;
			}
		}
		k -= 1;
		Sampler sampler(w, h, seeds[k]);
		sampler.current = radiance(sampler, true); // retrace path
		sampler.seed = rand(); // reseeding
		for (int m = 0; m < nMutations; m++) {
			RadianceRecord r1, r2;
			RunMarkovChain(sampler, r1, r2);
			write(r1);
			write(r2);
			totalMutations++;
		}
		{
			std::lock_guard<std::mutex> lockGuard(mutex);
			count++;
			printf("Done markov chain %d/%d, acceptance rate %lf\n", count, nChains,
				double(sampler.a) / double(sampler.a + sampler.r));
		}
	}
	for (auto& i : c) {
		i.x = i.x * (1.0 / double(samps));
		i.y = i.y * (1.0 / double(samps));
		i.z = i.z * (1.0 / double(samps));
	}
	FILE* f = fopen("image.ppm", "w");         // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for (int i = 0; i < w * h; i++)
		fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}
