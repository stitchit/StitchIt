// File: gaussian.hh
// Date: Sat May 04 01:33:12 2013 +0800
// Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#pragma once
#include <memory>
#include <vector>
#include "lib/mat.h"
#include "lib/utils.hh"
#include "lib/timer.hh"
#include "common/common.hh"
#include "omp.h"

namespace pano {

class GaussCache {
	public:
		std::unique_ptr<float, std::default_delete<float[]>> kernel_buf;
		float* kernel;
		int kw;
		GaussCache(float sigma);
};


class GaussianBlur {
	float sigma;
	GaussCache gcache;
	public:
		GaussianBlur(float sigma): sigma(sigma), gcache(sigma) {}
		
		Mat32f blurF(const Mat32f& img) const 
		{
			m_assert(img.channels() == 1);
			TotalTimer tm("gaussianblur");

			const int w = img.width(), h = img.height();
			Mat32f ret(h, w, 1);
			//Mat32f temp(h, w, 1);

			const int kw = gcache.kw;
			const int center = kw / 2;
			float *kernel = gcache.kernel;
		
		//	std::vector<> cur_line_mem(center * 2 + std::max(w, h), 0);
		//	T *cur_line = cur_line_mem.data() + center;

			{

			int chunksize = 32;
			std::vector<float> temp(w*(chunksize+(kw-1)));

			for(int i = 0 ; i < h ; i+=chunksize)
            {
				int limit = std::min(chunksize+center,(h-i-1));

		    	#pragma omp parallel for
				for(int i2 = -center; i2<limit; i2++)
				{
					int rowStart = i+i2;
					if((rowStart < 0) || (rowStart >= (h-center)))
						continue;

					for(int j = 0 ; j < center; j++)
					{
						float out = 0.0;
						for(int k = -center; k <= center; k++)
	                    {
    	                    if((j+k>=0))
           	                    out += img.at(rowStart,j+k) * kernel[k];
							else
								out += img.at(rowStart,0) * kernel[k];
                   	    }
                   	    temp[(i2+center)*w + j] = out;
                   	}

                	for(int j = center; j < w-center; j++)
                	{
                		float out = 0.0;
 						
							const float *ptr = img.ptr(rowStart,j);
							const float *kPtr = kernel;
							int x;

							for(x = -center; x < center; x+=4)
							{
								const __m128 patch = _mm_loadu_ps(ptr);
								const __m128 kern = _mm_load_ps(kPtr);
								const __m128 prod = _mm_mul_ps(patch,kern);
	
								__m128 shufReg, sumsReg;
	
    							shufReg = _mm_movehdup_ps(prod);        // Broadcast elements 3,1 to 2,0
    							sumsReg = _mm_add_ps(prod, shufReg);
    							shufReg = _mm_movehl_ps(shufReg, sumsReg); // High Half -> Low Half
    							sumsReg = _mm_add_ps(sumsReg, shufReg);
    							out +=  _mm_cvtss_f32(sumsReg);

								ptr +=4;
								kPtr+=4;

							}
							
							for(int k = x; k < center; k++)
							{
								out += img.at(rowStart,j+k) * kernel[k];   
							}
                    
                    	temp[(i2+center)*w + j] = out;
                    }

                    for(int j = w-center; j < w; j++)
                    {
                    	float out = 0.0;
                		for(int k = -center; k <= center; k++)
                        {
                            if((j+k)<w)
                                out += img.at(rowStart,j+k) * kernel[k];
							else
								out += img.at(rowStart,w-1) * kernel[k];
                        }
                        temp[(i2+center)*w + j] = out;
					}
				}

				if(i==0)
				{
					#pragma omp parallel for
					for(int j=0; j < center; j++)
						memcpy(&temp[j*w], &temp[center*w], w * sizeof(float));
					/*
						for(int k = 0; k < w; k++)
							temp[j*w + k] = temp[center*w + k];*/
				}

				if(h-i <= chunksize)
				{
					#pragma omp parallel for
					for(int j=(limit); j < (chunksize+center); j++)
						 memcpy(&temp[j*w], &temp[(limit-1)*w], w * sizeof(float));
					/*	for(int k = 0; k < w; k++)
							temp[j*w + k] = temp[(limit-1)*w + k];	*/
				}

				// Vertical Blur
				limit = std::min(chunksize,h-i-1);

				#pragma omp parallel for
				for(int i2 = 0; i2 < limit; i2++)
                {
                	int offsetStart = i2+center;
                    for(int j = 0; j < w; j++)
                    {
                        float out = 0.0;
						for(int k = -center; k <= center; k++)
        	                out += temp[(offsetStart+k)*w + j] * kernel[k];

						float *dest = ret.ptr(i+i2,j);
	                    *dest = out;
                    }
                }

            }

			}
/*
			#pragma omp parallel for
			for(int i = 0 ; i < h ; i++)
			{
				for(int j = 0; j < w; j++)
				{
					float out = 0.0;
					for(int k = -center; k <= center; k++)
					{
						if(((j+k)>=0) && ((j+k) < w))
							out += img.at(i,j+k) * kernel[k];						
					}
					float *dest = temp.ptr(i,j);
					*dest = out;
				}
			}

			#pragma omp parallel for
			for(int i = 0 ; i < h ; i++)
            {
                for(int j = 0; j < w; j++)
                {
                    float out = 0.0;
                    for(int k = -center; k <= center; k++)
                    {
						if(((i+k)>=0) && ((i+k) < h))
                        	out += temp.at(i+k,j) * kernel[k];
                    }

					float *dest = ret.ptr(i,j);
                	*dest = out;
				}
            }
*/
			return ret;
		}


        template <typename T>
		Mat<T> blur(const Mat<T>& img) const {
			m_assert(img.channels() == 1);
			TotalTimer tm("gaussianblur");
			const int w = img.width(), h = img.height();
			Mat<T> ret(h, w, img.channels());

			const int kw = gcache.kw;
			const int center = kw / 2;
			float * kernel = gcache.kernel;

			std::vector<T> cur_line_mem(center * 2 + std::max(w, h), 0);
			T *cur_line = cur_line_mem.data() + center;
		

			// apply to columns
			REP(j, w)
			{
				const T* src = img.ptr(0, j);
				// copy a column of src
				REP(i, h) 
				{
					cur_line[i] = *src;
					src += w;
				}

				// pad the border with border value
				T v0 = cur_line[0];
				for (int i = 1; i <= center; i++)
					cur_line[-i] = v0;
				v0 = cur_line[h - 1];
				for (int i = 0; i < center; i ++)
					cur_line[h + i] = v0;

				T *dest = ret.ptr(0, j);
				REP(i, h) 
				{
					T tmp{0};
					for (int k = -center; k <= center; k ++)
						tmp += cur_line[i + k] * kernel[k];
					*dest = tmp;
					dest += w;
				}
			}

			// apply to rows
			REP(i, h) 
			{
				T *dest = ret.ptr(i);
				memcpy(cur_line, dest, sizeof(T) * w);
				{	// pad the border
					T v0 = cur_line[0];
					for (int j = 1; j <= center; j ++)
						cur_line[-j] = v0;
					v0 = cur_line[w - 1];
					for (int j = 0; j < center; j ++)
						cur_line[center + j] = v0;
				}
				REP(j, w) 
				{
					T tmp{0};
					for (int k = -center; k <= center; k ++)
						tmp += cur_line[j + k] * kernel[k];
					*(dest ++) = tmp;
				}
			}
			return ret;
		}
};

class MultiScaleGaussianBlur {
	std::vector<GaussianBlur> gauss;		// size = nscale - 1
	public:
	MultiScaleGaussianBlur(
			int nscale, float gauss_sigma,
			float scale_factor) {
		
//		#pragma omp parallel for
		REP(k, nscale - 1) 
		{
			gauss.emplace_back(gauss_sigma);
			gauss_sigma *= scale_factor;
		}
	}

	Mat32f blur(const Mat32f& img, int n) const
	{
		return gauss[n - 1].blur(img); 
	}
};

}
