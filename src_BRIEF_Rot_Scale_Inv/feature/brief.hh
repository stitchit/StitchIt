//File: brief.hh
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#pragma once
#include <vector>
#include <utility>
#include "feature.hh"
#include "common/common.hh"
#include "dog.hh"
// BRIEF: Binary Robust Independent Elementary Features

#define NUM_ANGLES 360

namespace pano {

struct SampleCoords
{
	std::vector<std::pair<int, int>> patternX;
    std::vector<std::pair<int, int>> patternY;
};

struct BriefPattern {
	int s;	// size
	int n;
	SampleCoords patterns[NUM_ANGLES];
};


// Brief algorithm implementation
class BRIEF {
	public:
		BRIEF(const ScaleSpace& ss, const std::vector<SSPoint>&,
				const BriefPattern&);
		BRIEF(const BRIEF&) = delete;
		BRIEF& operator = (const BRIEF&) = delete;

		std::vector<Descriptor> get_descriptor() const;

		// s: size of patch. n: number of pair
		static BriefPattern gen_brief_pattern(int s, int n);

	protected:
		const ScaleSpace& ss;
		const std::vector<SSPoint>& points;
		const BriefPattern& pattern;

		Descriptor calc_descriptor(const SSPoint&) const;
};

}
