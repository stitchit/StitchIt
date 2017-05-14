//File: sift.cc
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include "sift.hh"
#include <algorithm>
#include "lib/timer.hh"
#include "dog.hh"
using namespace std;
using namespace config;
using namespace pano;

namespace 
{

float fast_sin(float x)
{
    float sin;
    
    if (x < -3.14159265)
        x += 6.28318531;
    else
        if (x >  3.14159265)
            x -= 6.28318531;

    if (x < 0)
        sin = 1.27323954 * x + .405284735 * x * x;
    else
        sin = 1.27323954 * x - 0.405284735 * x * x;

    return sin;
}

float fast_cos(float x)
{
    float cos;

    if (x < -3.14159265)
        x += 6.28318531;
    else
        if (x >  3.14159265)
            x -= 6.28318531;

    if (x >  3.14159265)
        x -= 6.28318531;

    if (x < 0)
        cos = 1.27323954 * x + 0.405284735 * x * x;
    else
        cos = 1.27323954 * x - 0.405284735 * x * x;

    return cos;
}


const int featlen = DESC_HIST_WIDTH * DESC_HIST_WIDTH * DESC_HIST_BIN_NUM;

Descriptor hist_to_descriptor(float* hist) 
{
	Descriptor ret;
	ret.descriptor.resize(featlen);
	memcpy(ret.descriptor.data(), hist, featlen * sizeof(float));

	TotalTimer tm("Hist_to_Descriptor");

    float sum = 0;

	// normalize and thresholding and renormalize
/*
 *  for (auto &i : ret.descriptor) sum += sqr(i);
 *  sum = sqrt(sum);
 *  for (auto &i : ret.descriptor) {
 *    i /= sum;
 *    update_min(i, (float)DESC_NORM_THRESH);
 *  }
 *  // L2 normalize SIFT
 *  sum = 0;
 *  for (auto &i : ret.descriptor) sum += sqr(i);
 *  sum = sqrt(sum);
 *  sum = (float)DESC_INT_FACTOR / sum;
 *  for (auto &i : ret.descriptor) i = i * sum;
 *
 */
	// using RootSIFT: rootsift= sqrt( sift / sum(sift) );
	// L1 normalize SIFT


/*	sum = 0;
	for (auto &i : ret.descriptor) sum += i;
	for (auto &i : ret.descriptor) 
	{
		i /= sum;
		i = std::sqrt(i) * DESC_INT_FACTOR;
	}
	return ret;
*/
	int step = 4;

	int n = ret.descriptor.size();
	//int remaining = n/step;	
	float *ptr = &ret.descriptor[0];

	m_assert(n % 4 == 0);
	for(int i = 0; i < n ; i+=step)
	{
		const __m128 a = _mm_loadu_ps(ptr);
		__m128 shufReg, sumsReg;

        shufReg = _mm_movehdup_ps(a);        // Broadcast elements 3,1 to 2,0
        sumsReg = _mm_add_ps(a, shufReg);
        shufReg = _mm_movehl_ps(shufReg, sumsReg); // High Half -> Low Half
        sumsReg = _mm_add_ps(sumsReg, shufReg);
        sum +=  _mm_cvtss_f32(sumsReg);

		ptr += 4;	
	}

	float inv = 1.0/sum;
	float mulFactor = inv * DESC_INT_FACTOR * DESC_INT_FACTOR;

	ptr = &ret.descriptor[0];
	__m128 mulVec = _mm_set1_ps(mulFactor);
	for(int i = 0; i < n ; i+=step)
    {
        const __m128 a = _mm_loadu_ps(ptr);
        __m128 norm = _mm_mul_ps(a,mulVec);
		
		_mm_store_ps(ptr,norm);
        
		ptr += 4; 
    }

	for(auto &i : ret.descriptor)
		i = std::sqrt(i);

	return ret;
}

void trilinear_interpolate(
		float xbin, float ybin, float hbin,
		float weight, float hist[][DESC_HIST_BIN_NUM]) 
{
	// WARNING: x,y can be -1
	int ybinf = floor(ybin),
	xbinf = floor(xbin),
	hbinf = floor(hbin);
	float ybind = ybin - ybinf,
				xbind = xbin - xbinf,
				hbind = hbin - hbinf;
	REP(dy, 2) 
		if (between(ybinf + dy, 0, DESC_HIST_WIDTH)) 
		{
			float w_y = weight * (dy ? ybind : 1 - ybind);
			REP(dx, 2) 
				if (between(xbinf + dx, 0, DESC_HIST_WIDTH)) 
				{
					float w_x = w_y * (dx ? xbind : 1 - xbind);
					int bin_2d_idx = (ybinf + dy) * DESC_HIST_WIDTH + (xbinf + dx);
					hist[bin_2d_idx][hbinf % DESC_HIST_BIN_NUM] += w_x * (1 - hbind);
					hist[bin_2d_idx][(hbinf + 1) % DESC_HIST_BIN_NUM] += w_x * hbind;
				}
		}
}
}

namespace pano {

SIFT::SIFT(const ScaleSpace& ss,
		const vector<SSPoint>& keypoints):
	ss(ss), points(keypoints)
{ }

std::vector<Descriptor> SIFT::get_descriptor() const {
	TotalTimer tm("sift descriptor");
	vector<Descriptor> ret;
	
	int size = points.size();

	//#pragma omp parallel for schedule(dynamic)
	for(int i = 0 ; i < size; i++)
	//for (auto& p : points) 
	{
		auto& p = points[i];
		auto desp = calc_descriptor(p);

		//#pragma omp critical
		ret.emplace_back(move(desp));
	}
	return ret;
}

Descriptor SIFT::calc_descriptor(const SSPoint& p) const 
{
	const static float pi2 = 2 * M_PI;
	const static float nbin_per_rad = DESC_HIST_BIN_NUM / pi2;
	const static float DESC_HW_2 =  DESC_HIST_WIDTH / 2;

	const GaussianPyramid& pyramid = ss.pyramids[p.pyr_id];
	int w = pyramid.w, h = pyramid.h;
	auto& mag_img = pyramid.get_mag(p.scale_id);
	auto& ort_img = pyramid.get_ort(p.scale_id);

	Coor coor = p.coor;
	float ort = p.dir,
				// size of blurred field of this point in orignal image
				hist_w = p.scale_factor * DESC_HIST_SCALE_FACTOR,
				// sigma is half of window width from lowe
				exp_denom = 2 * sqr(DESC_HIST_WIDTH);
	// radius of gaussian to use
	int radius = round(M_SQRT1_2 * hist_w * (DESC_HIST_WIDTH + 1));

	float hist[DESC_HIST_WIDTH * DESC_HIST_WIDTH][DESC_HIST_BIN_NUM];
	memset(hist, 0, sizeof(hist));

/*	float cosort = fast_cos(ort),
				sinort = fast_sin(ort);
*/


	float cosort = cos(ort),
				sinort = sin(ort);

	
	//#pragma omp parallel for
	for (int xx = -radius; xx <= radius; xx ++) 
	{
		int nowx = coor.x + xx;
		if (!between(nowx, 1, w - 1)) 
			continue;
		for (int yy = -radius; yy <= radius; yy ++) 
		{
			int nowy = coor.y + yy;
			if (!between(nowy, 1, h - 1)) 
				continue;
			if (sqr(xx) + sqr(yy) > sqr(radius)) 
				continue;		// to be circle
			
			// coordinate change, relative to major orientation
			// major orientation become (x, 0)
			float y_rot = (-xx * sinort + yy * cosort) / hist_w,
						x_rot = (xx * cosort + yy * sinort) / hist_w;
			// calculate 2d bin idx (which bin do I fall into)
			// -0.5 to make the center of bin 1st (x=1.5) falls fully into bin 1st
			float ybin = y_rot + DESC_HW_2 - 0.5,
						xbin = x_rot + DESC_HW_2 - 0.5;

			if (!between(ybin, -1, DESC_HIST_WIDTH) ||
					!between(xbin, -1, DESC_HIST_WIDTH)) 
				continue;

			float now_mag = mag_img.at(nowy, nowx),
						now_ort = ort_img.at(nowy, nowx);
			
			// gaussian & magitude weight on histogram
			float weight = expf(-(sqr(x_rot) + sqr(y_rot)) / exp_denom);
			weight = weight * now_mag;

			now_ort -= ort;	// for rotation invariance
			if (now_ort < 0) 
				now_ort += pi2;
			if (now_ort > pi2) 
				now_ort -= pi2;
			
			// bin number in histogram
			float hist_bin = now_ort * nbin_per_rad;

			// all three bin idx are float, do trilinear interpolation
			trilinear_interpolate(
					xbin, ybin, hist_bin, weight, hist);
		}
	}

	// build descriptor from hist

	Descriptor ret = hist_to_descriptor((float*)hist);
	ret.coor = p.real_coor;
	return ret;
}

}
