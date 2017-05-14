//File: blender.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "blender.hh"
#include <stdio.h>
#include <nmmintrin.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <mmintrin.h>
#include <iostream>
#include "lib/config.hh"
#include "lib/imgproc.hh"
#include "lib/timer.hh"
using namespace std;
using namespace config;


namespace pano {

void LinearBlender::add_image(
			const Coor& upper_left,
			const Coor& bottom_right,
			ImageRef &img,
			std::function<Vec2D(Coor)> coor_func) {
	images.emplace_back(ImageToAdd{Range{upper_left, bottom_right}, img, coor_func});
	target_size.update_max(bottom_right);
}

Mat32f LinearBlender::run() {
        TotalTimer tm("blender_vectorized");
	Mat32f target(target_size.y, target_size.x, 3);

#define GET_COLOR_AND_W \
					Vec2D img_coor = img.map_coor(i, j); \
					if (img_coor.isNaN()) continue; \
					float r = img_coor.y, c = img_coor.x; \
					auto color = interpolate(*img.imgref.img, r, c); \
					if (color.x < 0) continue; \
					float	w = 0.5 - fabs(c / img.imgref.width() - 0.5); \
					if (not config::ORDERED_INPUT) /* blend both direction */\
						w *= (0.5 - fabs(r / img.imgref.height() - 0.5)); \
					color *= w

	if (LAZY_READ) {
		// Use weighted pixel, to iterate over images (and free them) instead of target.
		// Will be a little bit slower
		Mat<float> weight(target_size.y, target_size.x, 1);
		memset(weight.ptr(), 0, target_size.y * target_size.x * sizeof(float));
		fill(target, Color::BLACK);
#pragma omp parallel for schedule(dynamic)
		REP(k, (int)images.size()) {
			auto& img = images[k];
			img.imgref.load();
			auto& range = img.range;
			for (int i = range.min.y; i < range.max.y; ++i) {
				float *row = target.ptr(i);
				float *wrow = weight.ptr(i);
				for (int j = range.min.x; j < range.max.x; ++j) {
					GET_COLOR_AND_W;
					//#pragma omp critical
					{
						row[j*3] += color.x;
						row[j*3+1] += color.y;
						row[j*3+2] += color.z;
						wrow[j] += w;
					}
				}
			}
			img.imgref.release();
		}
//#pragma omp parallel for schedule(dynamic)
	/*	REP(i, target.height()) {
			auto row = target.ptr(i);
			auto wrow = weight.ptr(i);
			REP(j, target.width()) {
				if (wrow[j]) {
					*(row++) /= wrow[j]; *(row++) /= wrow[j]; *(row++) /= wrow[j];
				} else {
					*(row++) = -1; *(row++) = -1; *(row++) = -1;
				}
			}
		}*/

/*		//SIMD un-optimized
                int size = target.height() * target.width() * 3;
                int end = size / 4;
		int remaining = end + size%4;
                auto row = target.ptr(0);
                auto wrow = weight.ptr(0);
                float* wrow_3;
                wrow_3 = (float*)malloc(sizeof(float)*size);

                #pragma omp parallel for
                for(int i = 0;i < target.height();i++) {
                	for(int j = 0;j < target.width();j++) {
                                if (wrow[i*target.width()+j]) {
				float temp = 1/wrow[i*target.width()+j];
				wrow_3[i*target.width()+3*j] = temp;
				wrow_3[i*target.width()+3*j + 1] = temp;
				wrow_3[i*target.width()+3*j + 2] = temp;
                                }
				else {
					wrow_3[i*target.width()+3*j] = 0.0f;
					wrow_3[i*target.width()+3*j + 1] = 0.0f;
					wrow_3[i*target.width()+3*j + 2] = 0.0f;
				}
				
			}
		}
		
                
		//__m128 ones = _mm_set1_ps(0.001f);
		__m128 zeroes = _mm_set1_ps(0.f);
		__m128 row1 = _mm_set1_ps(-1.f);
                
                for(int j = 0;j < end;j++)
                {
                	const __m128 a = _mm_loadu_ps(row);
			const __m128 b = _mm_loadu_ps(wrow_3);

			
                        
                        //__m128 wrow_mask = _mm_add_ps(b, ones);
                        //__m128 row2 = _mm_div_ps(a, wrow_mask);
			//__m128 row2 = _mm_div_ps(a, b);
                        __m128 row2 = _mm_mul_ps(a, b);
                        __m128 not_zero_mask = _mm_cmpneq_ps(b, zeroes);
                        __m128 zero_mask = _mm_cmpeq_ps(b, zeroes);
                        row1 = _mm_and_ps(row1, zero_mask);
			row2 = _mm_and_ps(row2, not_zero_mask);
                        __m128 result = _mm_or_ps(row1, row2);

                        _mm_storeu_ps(row, result);
                        row += 4;
			wrow_3 += 4;
                                               
                }
		

		for(int j = end; j < remaining; j++) {

			if (wrow_3[j]) {
				*(row++) *= wrow_3[j]; *(row++) /= wrow_3[j]; *(row++) /= wrow_3[j];
			} 
                        else {
				*(row++) = -1; *(row++) = -1; *(row++) = -1;
			}
                }
*/

		//SIMD Optimized
		int size = target.height() * target.width();
                //int end = size / 3;
		//int remaining = end + size%4;
                auto row0 = target.ptr(0);
                auto wrow = weight.ptr(0);
		//__m128 mask = _mm_set_ps(1.0f, 1.0f, 1.0f, 0.0f);
		//__m128 zeroes = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
		//__m128 zero_mask = _mm_cmpneq_ps(mask, zeroes);
		//__m128 not_zero_mask = _mm_cmpeq_ps(mask, zeroes);
		
		#pragma omp parallel for
		for(int i =0;i<(size -1);i++) {
			float *row = row0 + 3*i;
			if(!wrow[i]) {
				row[0] = -1;row[1] = -1;row[2] = -1;
			}
			else {
				float temp = 1/wrow[i];
				__m128 row1 = _mm_set_ps(temp,temp,temp,1.0);
				__m128 row2 = _mm_loadu_ps(row);
				__m128 prod = _mm_mul_ps(row2, row1);
				//__m128 final_prod = _mm_and_ps(prod, zero_mask);
				//__m128 final_row = _mm_and_ps(row2, not_zero_mask);
				//_mm_storeu_ps(row,_mm_or_ps(final_prod, final_row)); 
				_mm_storeu_ps(row,prod); 
				
			}
			//row += 3;
		}
		if(wrow[size - 1]) {
			row0[size*3 - 3] /= wrow[size - 1];
			row0[size*3 - 2] /= wrow[size - 1];
			row0[size*3 - 1] /= wrow[size - 1];
		}
		else {
			
			row0[size*3 - 3] = -1;
			row0[size*3 - 2] = -1;
			row0[size*3 - 1] = -1;
		}

	} else {
		fill(target, Color::NO);
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < target.height(); i ++) {
			float *row = target.ptr(i);
			for (int j = 0; j < target.width(); j ++) {
				Color isum = Color::BLACK;
				float wsum = 0;
				for (auto& img : images) if (img.range.contain(i, j)) {
					GET_COLOR_AND_W;
					isum += color;
					wsum += w;
				}
				if (wsum > 0)	// keep original Color::NO
					(isum / wsum).write_to(row + j * 3);
			}
		}
	}
	return target;
}

}
