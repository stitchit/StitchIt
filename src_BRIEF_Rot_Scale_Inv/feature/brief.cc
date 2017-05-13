//File: brief.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "brief.hh"
#include <omp.h>
#include <random>
#include <math.h>

#include "lib/debugutils.hh"
#include "lib/timer.hh"

#define DEGREES_TO_RAD 0.0174533
#define RAD_TO_DEG 57.2958

#define SIZE 5

using namespace std;
using namespace pano;

namespace pano {

BRIEF::BRIEF(const ScaleSpace& ss, const vector<SSPoint>& points,
		const BriefPattern& pattern):
	ss(ss), points(points), pattern(pattern) { }

vector<Descriptor> BRIEF::get_descriptor() const
{
	TotalTimer tm("brief descriptor");

    const int half = std::max(SIZE/2,(int)(pattern.s*0.75));
	vector<Descriptor> ret;

    int n = points.size();
    #pragma omp parallel for
    for(int i = 0; i < n ; i++)
	{
        auto& p = points[i];        
        const GaussianPyramid& pyramid = ss.pyramids[p.pyr_id];
        auto& img = pyramid.get(p.scale_id);
        int w = pyramid.w, h = pyramid.h;

		int x = round(p.real_coor.x * w),
				y = round(p.real_coor.y * h);

        if((x <= half) || (y <= half) || (x >= (w - half)) || (y >= (h - half)))
            continue;

		auto desp = calc_descriptor(p);
        #pragma omp critical
		    ret.emplace_back(move(desp));
	}

	return ret;
}


float fast_atan(float y, float x) {
    float absx = fabs(x), absy = fabs(y);
    float m = max(absx, absy);

    if (m < EPS) 
        return -M_PI;
    float a = min(absx, absy) / m;
    float s = a * a;
    float r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
    if (absy > absx)
        r = M_PI_2 - r;
    if (x < 0) r = M_PI - r;
    if (y < 0) r = -r; 
    return r;
}


int calcAngle(int x, int y, const Mat32f& img)// int patchCalc, int **patch)
{
    int size = SIZE;
    int half = size/2;

    float meanX = 0;
    float meanY = 0;
    float pixelSum = 0;
   
    auto pixel = [&](int r, int c)  
    {   
        const float* ptr = img.ptr(r, c); 
        return ptr[0];
        //return (ptr[0]) + (ptr[1]) + (ptr[2]) / 3;
    };  
 
    for(int i = -half; i <= half; i++)
    {   
        for(int j = -half; j <= half; j++)
        {
            int x1 = x + i;
            int y1 = y + j;
    
            float val;    
            val = pixel(y1,x1);

            meanX += val*(i);
            meanY += val*(j);
            pixelSum += val;
        }
    }   

    meanX = meanX / pixelSum;
    meanY = meanY / pixelSum;
    
    float angle = fast_atan(meanY, meanX);
    if(angle < 0)
        angle = angle + 2*M_PI;

    int angle1 = (int(round(angle * RAD_TO_DEG))) % 360;
    return angle1;
}
    

Descriptor BRIEF::calc_descriptor(const SSPoint& p) const
{
    const GaussianPyramid& pyramid = ss.pyramids[p.pyr_id];
    auto& img = pyramid.get(p.scale_id);
    int w = pyramid.w, h = pyramid.h;

	int x = round(p.real_coor.x * w),
			y = round(p.real_coor.y * h);

    if(p.angle == -1)
        p.angle = calcAngle(x,y,img);

	int angle = p.angle;
	
	SampleCoords chosenP = pattern.patterns[angle];
    //SampleCoords chosenP = pattern.patterns[0];	

	const int n = pattern.n;
	
	vector<bool> bits(n, false);
	auto pixel = [&](int r, int c) 
	{
		const float* ptr = img.ptr(r, c);
        return ptr[0];
		//return (ptr[0] + ptr[1] + ptr[2]) / 3;
	};

	REP(i, n) 
	{
		int x1 = x + chosenP.patternX[i].first;
		int x2 = x + chosenP.patternX[i].second;
		int y1 = y + chosenP.patternY[i].first;
		int y2 = y + chosenP.patternY[i].second;
		
		bits[i] = pixel(y1, x1) > pixel(y2, x2);
	}

	Descriptor ret;
	ret.coor = p.real_coor;
	ret.descriptor.resize(n / 32, 0);
	
	#pragma omp parallel for schedule(static)
	for(int i = 0 ; i < n; i+=32)
	{
		int toSet = 0;
		int idx = i/32;
		int last = i + 32;
		for(int j = i; j < last; j++)
		{
			if(bits[j])
			{
				toSet = toSet | (1 << j);
			}
		}
	    ret.descriptor[idx] = toSet;
	}
	return ret;
}

BriefPattern BRIEF::gen_brief_pattern(int s, int n) 
{
	m_assert(s % 2 == 1);
	m_assert(n % 32 == 0);

	BriefPattern ret;
	ret.s = s;
	ret.n = n;	

	int half = s/2;
	
	while(n--)
	{
		int x1 = (rand()%s) - half;
		int x2 = (rand()%s) - half;
		int y1 = (rand()%s) - half;
		int y2 = (rand()%s) - half;
	

		#pragma omp parallel for
		for(int i = 0; i < NUM_ANGLES; i++)
		{
			float ang = i * DEGREES_TO_RAD;
			float cos1 = cos(ang);
			float sin1 = sin(ang);
			int x11 = round(x1*cos1 - y1*sin1); 
			int x21 = round(x2*cos1 - y2*sin1);
			int y11 = round(x1*sin1 + y1*cos1);
			int y21 = round(x2*sin1 + y2*cos1);
		
			ret.patterns[i].patternX.emplace_back(x11, x21);
			ret.patterns[i].patternY.emplace_back(y11, y21);
		}
	}

	return ret;

}
}
