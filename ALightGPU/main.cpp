
#include <iostream>
#include "stdlib.h"
#include <thread>

#include "header/Hitable.h"
#include "header/Camera.h"
#include "header/Sphere.h"
#include "header/hitable_list.h"
#include "header/GLWindow.h"
using namespace std;

Hitable *world;	Hitable *list[2]; Camera cam;
void InitData()
{
	PixelLength = ImageHeight*ImageWidth*4;
	PixelData = new GLbyte[PixelLength];
	col = new Vec3[ImageHeight*ImageWidth];
	//InitWindow zero
	for (auto i = 0; i < PixelLength; i++)
		PixelData[i] = static_cast<GLbyte>(int(0));
}
void InitScene()
{

	list[0] = new Sphere(Vec3(0, 0, -1), 0.5);
	list[1] = new Sphere(Vec3(0, -100.5, -1), 100);
	world = new HitableList(list, 2);

}

Vec3 shadeNormal(const Ray& r,Hitable *world)
{
	HitRecord rec;
	if(world->Hit(r,0,99999,rec))
	{
		return  0.5*Vec3(rec.normal.x() + 1, rec.normal.y() + 1, rec.normal.z() + 1);
	}
	else
	{
		Vec3 unit_dir = unit_vector(r.Direction());
		auto t = 0.5*(unit_dir.y() + 1);
		return  (1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
	}
}
Vec3 shade(const Ray& r, Hitable *world)
{
	HitRecord rec;
	if (world->Hit(r, 0, 99999, rec))
	{
		Vec3 target = rec.p + rec.normal + RandomInUnitSphere();
		return  0.5*shade(Ray(rec.p, target - rec.p), world);
	}
	else
	{
		Vec3 unit_dir = unit_vector(r.Direction());
		auto t = 0.5*(unit_dir.y() + 1);
		return  (1.0 - t)*Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
	}
}
void Render()
{
	const Vec3 lower_left_corner(-2.0, -2.0, -1.0),horizontal(4, 0, 0), vertical(0, 4, 0), origin(0, 0, 0);

	for (auto j=ImageHeight-1;j>=0;j--)
		for (auto i=0;i<ImageWidth;i++)
		{
			Vec3 c(0, 0, 0);
			for (int s=0;s<SPP;s++)
			{
				
				float u = float(i + drand48()) / float(ImageWidth);
				float v = float(j + drand48()) / float(ImageHeight);
				Ray r = cam.GetRay(u, v);
				Vec3 p = r.PointAtParameter(2);
				c += shade(r, world);
			}
			//show_progress((ImageHeight - j)*ImageWidth + i, ImageHeight*ImageWidth);
			//cout<<((ImageHeight - j)*ImageWidth + i)<<" / "<<ImageHeight*ImageWidth<<endl;
			c /= SPP;
			SetPixel(i, j, &c);
		}
}




int main(int argc, char* argv[])
{
	
	InitData();
	InitScene();
	thread t(Render);

	InitWindow(argc, argv, GLUT_DOUBLE | GLUT_RGBA, 100, 100, ImageWidth, ImageHeight, "ALightCPP");
	//t.join();
	return 0;
}

