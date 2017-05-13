------------------------------BLENDER.CC---------------------------

//File: blender.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "blender.hh"

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
#pragma omp parallel for schedule(dynamic)
		REP(i, target.height()) {
			auto row = target.ptr(i);
			auto wrow = weight.ptr(i);
			REP(j, target.width()) {
				if (wrow[j]) {
					*(row++) /= wrow[j]; *(row++) /= wrow[j]; *(row++) /= wrow[j];
				} else {
					*(row++) = -1; *(row++) = -1; *(row++) = -1;
				}
			}
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

------------------------------CAMERA.CC---------------------------

//File: camera.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "camera.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <string>

#include "lib/timer.hh"
#include "match_info.hh"
#include "homography.hh"
using namespace std;
using namespace pano;
// Implement stuffs about camera K,R matrices
namespace {

// See: Creating Full View Panoramic Image Mosaics - Szeliski
double get_focal_from_matrix(const Homography& h) {
	double d1, d2; // Denominators
	double v1, v2; // Focal squares value candidates
	double f1, f0;

	d1 = h[6] * h[7];
	d2 = (h[7] - h[6]) * (h[7] + h[6]);
	v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;
	v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;
	if (v1 < v2)
		swap(v1, v2);
	if (v1 > 0 && v2 > 0)
		f1 = sqrt(abs(d1) > abs(d2) ? v1 : v2);
	else if (v1 > 0)
		f1 = sqrt(v1);
	else
		return 0;

	d1 = h[0] * h[3] + h[1] * h[4];
	d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
	v1 = -h[2] * h[5] / d1;
	v2 = (h[5] * h[5] - h[2] * h[2]) / d2;
	if (v1 < v2)
		swap(v1, v2);
	if (v1 > 0 && v2 > 0)
		f0 = sqrt(abs(d1) > abs(d2) ? v1 : v2);
	else if (v1 > 0)
		f0 = sqrt(v1);
	else
		return 0;
	if (std::isinf(f1) || std::isinf(f0))
		return 0;
	return sqrt(f1 * f0);
}

}

namespace pano {

Camera::Camera() : R(Homography::I()) { }

Homography Camera::K() const {
	Homography ret{Homography::I()};
	ret[0] = focal;
	ret[2] = ppx;
	ret[4] = focal * aspect;
	ret[5] = ppy;
	return ret;
}

double Camera::estimate_focal(
		const vector<vector<MatchInfo>>& matches) {
	int n = matches.size();
	vector<double> estimates;
	REP(i, n) REPL(j, i + 1, n) {
		auto& match = matches[i][j];
		if (match.confidence < EPS) continue;
		estimates.emplace_back(
				get_focal_from_matrix(match.homo));
	}
	int ne = estimates.size();
	if (ne < min(n - 1, 3))
		return -1;	// focal estimate fail
	sort(begin(estimates), end(estimates));
	if (ne % 2 == 1)
		return estimates[ne >> 1];
	else
		return (estimates[ne >> 1] + estimates[(ne >> 1) - 1]) * 0.5;
}


//https://en.wikipedia.org/wiki/Rotation_matrix?oldformat=true#Determining_the_axis
void Camera::rotation_to_angle(const Homography& r, double& rx, double& ry, double& rz) {
	using namespace Eigen;
	auto R_eigen = Map<const Eigen::Matrix<double, 3, 3, RowMajor>>(r.data);

	JacobiSVD<MatrixXd> svd(R_eigen, ComputeFullU | ComputeFullV);
	Matrix3d Rnew = svd.matrixU() * (svd.matrixV().transpose());
	if (Rnew.determinant() < 0)
		Rnew *= -1;

	// r is eigenvector of R with eigenvalue=1
	rx = Rnew(2,1) - Rnew(1,2);
	ry = Rnew(0,2) - Rnew(2,0);
	rz = Rnew(1,0) - Rnew(0,1);

	double s = sqrt(rx*rx + ry*ry + rz*rz);
	if (s < GEO_EPS) {
		rx = ry = rz = 0;
	} else {
		// 1 + 2 * cos(theta) = trace(R)
		double cos = (Rnew(0,0) + Rnew(1,1) + Rnew(2,2) - 1) * 0.5;
		cos = cos > 1. ? 1. : cos < -1. ? -1. : cos;		// clip
		double theta = acos(cos);

		double mul = 1.0 / s * theta;
		rx *= mul; ry *= mul; rz *= mul;
	}
}

//https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
void Camera::angle_to_rotation(double rx, double ry, double rz, Homography& r) {
	double theta = rx*rx + ry*ry + rz*rz;
	if (theta < GEO_EPS_SQR) {	// theta ^2
		// first order Taylor. see code of ceres-solver
		r = Homography{{1, -rz, ry, rz, 1, -rx, -ry, rx, 1}};
		return;
	}
	theta = sqrt(theta);
	double itheta = theta ? 1./theta : 0.;
	rx *= itheta; ry *= itheta; rz *= itheta;

	// outer product with itself
	double u_outp[] = {rx*rx, rx*ry, rx*rz, rx*ry, ry*ry, ry*rz, rx*rz, ry*rz, rz*rz };
	// cross product matrix
	double u_crossp[] = {0, -rz, ry, rz, 0, -rx, -ry, rx, 0 };

	r = Homography::I();

	double c = cos(theta),
				 s = sin(theta),
				 c1 = 1 - c;
	r.mult(c);
	REP(k, 9)
		r[k] += c1 * u_outp[k] + s * u_crossp[k];
}

void Camera::straighten(std::vector<Camera>& cameras) {
	using namespace Eigen;
	Matrix3d cov = Matrix3d::Zero();
	for (auto& c : cameras) {
		// R is from reference image to current image
		// the first row is X vector ([1,0,0] * R)
		Vector3d v;
		v << c.R[0], c.R[1], c.R[2];
		cov += v * v.transpose();
	}
	// want to solve Cov * u == 0
	auto V = cov.jacobiSvd(ComputeFullU | ComputeFullV).matrixV();
	Vector3d normY = V.col(2);		// corrected y-vector

	Vector3d vz = Vector3d::Zero();
	for (auto& c : cameras) {
		vz(0) += c.R[6];
		vz(1) += c.R[7];
		vz(2) += c.R[8];
	}
	Vector3d normX = normY.cross(vz);
	normX.normalize();
	Vector3d normZ = normX.cross(normY);

	double s = 0;
	for (auto& c : cameras) {
		Vector3d v; v << c.R[0], c.R[1], c.R[2];
		s += normX.dot(v);
	}
	if (s < 0) normX *= -1, normY *= -1;	// ?

	Homography r;
	REP(i, 3) r[i * 3] = normX(i);
	REP(i, 3) r[i * 3 + 1] = normY(i);
	REP(i, 3) r[i * 3 + 2] = normZ(i);
	for (auto& c : cameras)
		c.R = c.R * r;
}

}

-------------------------CAMERA_ESTIMATE.CC---------------------------

//File: camera_estimator.cc
//Date:
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include "camera_estimator.hh"
#include <queue>

#include "lib/debugutils.hh"
#include "lib/timer.hh"
#include "lib/utils.hh"
#include "lib/config.hh"
#include "camera.hh"
#include "match_info.hh"
#include "incremental_bundle_adjuster.hh"

using namespace std;
using namespace config;

namespace pano {

CameraEstimator::CameraEstimator(
    std::vector<std::vector<MatchInfo>>& matches,
    const std::vector<Shape2D>& image_shapes) :
  n(matches.size()),
  matches(matches),
  shapes(image_shapes),
  cameras(matches.size())
{ m_assert(matches.size() == shapes.size()); }

CameraEstimator::~CameraEstimator() = default;

void CameraEstimator::estimate_focal() {
  // assign an initial focal length
  double focal = Camera::estimate_focal(matches);
  if (focal > 0) {
    for (auto& c : cameras)
      c.focal = focal;
    print_debug("Estimated focal: %lf\n", focal);
  } else {
    print_debug("Cannot estimate focal. Will use a naive one.\n");
    REP(i, n) // hack focal
      cameras[i].focal = (shapes[i].w + shapes[i].h) * 0.5;
  }
}

vector<Camera> CameraEstimator::estimate() {
  GuardedTimer tm("Estimate Camera");
  estimate_focal();

  IncrementalBundleAdjuster iba(cameras);
  vector<bool> vst(n, false);
  traverse(
      [&](int node) {
        // set the starting point to identity
        cameras[node].R = Homography::I();
        cameras[node].ppx = cameras[node].ppy = 0;
      },
      [&](int now, int next) {
        print_debug("Best edge from %d to %d\n", now, next);
        auto Kfrom = cameras[now].K();

        // initialize camera[next]:
        auto Kto = cameras[next].K();
        auto Hinv = matches[now][next].homo;	// from next to now
        auto Mat = Kfrom.inverse() * Hinv * Kto;
        // this is camera extrincis R, i.e. going from identity to this image
        cameras[next].R = (cameras[now].Rinv() * Mat).transpose();
        cameras[next].ppx = cameras[next].ppy = 0;
        // initialize by the last camera. It shouldn't fail but it did,
        // may be deficiency in BA, or because of ignoring large error for now
        //cameras[next] = cameras[now];

        if (MULTIPASS_BA > 0) {
          // add next to BA
          vst[now] = vst[next] = true;
          REP(i, n) if (vst[i] && i != next) {
            const auto& m = matches[next][i];
            if (m.match.size() && m.confidence > 0) {
              iba.add_match(i, next, m);
              if (MULTIPASS_BA == 2) {
                print_debug("MULTIPASS_BA: %d -> %d\n", next, i);
                iba.optimize();
              }
            }
          }
          if (MULTIPASS_BA == 1)
            iba.optimize();
        }
      });

  if (MULTIPASS_BA == 0) {		// optimize everything together
    REPL(i, 1, n) REP(j, i) {
      auto& m = matches[j][i];
      if (m.match.size() && m.confidence > 0)
        iba.add_match(i, j, m);
    }
    iba.optimize();
  }

  if (STRAIGHTEN) Camera::straighten(cameras);
  return cameras;
}

void CameraEstimator::traverse(
    function<void(int)> callback_init_node,
    function<void(int, int)> callback_edge) {
  struct Edge {
    int v1, v2;
    float weight;
    Edge(int a, int b, float v):v1(a), v2(b), weight(v) {}
    bool operator < (const Edge& r) const { return weight < r.weight;	}
  };
  // choose a starting point
  Edge best_edge{-1, -1, 0};
  REP(i, n) REPL(j, i+1, n) {
    auto& m = matches[i][j];
    if (m.confidence > best_edge.weight)
      best_edge = Edge{i, j, m.confidence};
  }
  if (best_edge.v1 == -1)
    error_exit("No connected images are found!");
  callback_init_node(best_edge.v1);

  priority_queue<Edge> q;
  vector<bool> vst(n, false);

  auto enqueue_edges_from = [&](int from) {
    REP(i, n) if (i != from && !vst[i]) {
      auto& m = matches[from][i];
      if (m.confidence > 0)
        q.emplace(from, i, m.confidence);
    }
  };

  vst[best_edge.v1] = true;
  enqueue_edges_from(best_edge.v1);
  int cnt = 1;
  while (q.size()) {
    do {
      best_edge = q.top();
      q.pop();
    } while (q.size() && vst[best_edge.v2]);
    if (vst[best_edge.v2])	// the queue is exhausted
      break;
    vst[best_edge.v2] = true;
    cnt ++;
    callback_edge(best_edge.v1, best_edge.v2);
    enqueue_edges_from(best_edge.v2);
  }
  if (cnt != n) {
    string unconnected;
    REP(i, n) if (not vst[i])
      unconnected += to_string(i) + " ";
    error_exit(ssprintf(
          "Found a tree of size %d!=%d, image %s are not connected well!",
          cnt, n, unconnected.c_str()));
  }
}
}

--------------------------CYCLSTITCHER.CC---------------------------

//File: cylstitcher.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "cylstitcher.hh"

#include "lib/timer.hh"
#include "lib/config.hh"
#include "lib/imgproc.hh"
#include "feature/matcher.hh"
#include "transform_estimate.hh"
#include "blender.hh"
#include "match_info.hh"
#include "warp.hh"

using namespace config;
using namespace std;

namespace pano {

Mat32f CylinderStitcher::build() {
	calc_feature();
	bundle.identity_idx = imgs.size() >> 1;
	build_warp();
	free_feature();
	bundle.proj_method = ConnectedImages::ProjectionMethod::flat;
	bundle.update_proj_range();
	auto ret = bundle.blend();
	return perspective_correction(ret);
}

void CylinderStitcher::build_warp() {;
	GuardedTimer tm("build_warp()");
	int n = imgs.size(), mid = bundle.identity_idx;
	REP(i, n) bundle.component[i].homo = Homography::I();

	Timer timer;
	vector<MatchData> matches;		// matches[k]: k,k+1
	PairWiseMatcher pwmatcher(feats);
	matches.resize(n-1);
#pragma omp parallel for schedule(dynamic)
	REP(k, n - 1)
		matches[k] = pwmatcher.match(k, (k + 1) % n);
	print_debug("match time: %lf secs\n", timer.duration());

	vector<Homography> bestmat;

	float minslope = numeric_limits<float>::max();
	float bestfactor = 1;
	if (n - mid > 1) {
		float newfactor = 1;
		// XXX: ugly
		float slope = update_h_factor(newfactor, minslope, bestfactor, bestmat, matches);
		if (bestmat.empty())
			error_exit("Failed to find hfactor");
		float centerx1 = 0, centerx2 = bestmat[0].trans2d(0, 0).x;
		float order = (centerx2 > centerx1 ? 1 : -1);
		REP(k, 3) {
			if (fabs(slope) < SLOPE_PLAIN) break;
			newfactor += (slope < 0 ? order : -order) / (5 * pow(2, k));
			slope = update_h_factor(newfactor, minslope, bestfactor, bestmat, matches);
		}
	}
	print_debug("Best hfactor: %lf\n", bestfactor);
	CylinderWarper warper(bestfactor);
	REP(k, n) imgs[k].load();
#pragma omp parallel for schedule(dynamic)
	REP(k, n) warper.warp(*imgs[k].img, keypoints[k]);

	// accumulate
	REPL(k, mid + 1, n) bundle.component[k].homo = move(bestmat[k - mid - 1]);
#pragma omp parallel for schedule(dynamic)
	REPD(i, mid - 1, 0) {
		matches[i].reverse();
		MatchInfo info;
		bool succ = TransformEstimation(
				matches[i], keypoints[i + 1], keypoints[i],
				imgs[i+1].shape(), imgs[i].shape()).get_transform(&info);
		// Can match before, but not here. This would be a bug.
		if (! succ)
			error_exit(ssprintf("Failed to match between image %d and %d.", i, i+1));
		// homo: operate on half-shifted coor
		bundle.component[i].homo = info.homo;
	}
	REPD(i, mid - 2, 0)
		bundle.component[i].homo = bundle.component[i + 1].homo * bundle.component[i].homo;
	bundle.calc_inverse_homo();
}

float CylinderStitcher::update_h_factor(float nowfactor,
		float & minslope,
		float & bestfactor,
		vector<Homography>& mat,
		const vector<MatchData>& matches) {
	const int n = imgs.size(), mid = bundle.identity_idx;
	const int start = mid, end = n, len = end - start;

	vector<Shape2D> nowimgs;
	vector<vector<Vec2D>> nowkpts;
	REPL(k, start, end) {
		nowimgs.emplace_back(imgs[k].shape());
		nowkpts.push_back(keypoints[k]);
	}			// nowfeats[0] == feats[mid]

	CylinderWarper warper(nowfactor);
#pragma omp parallel for schedule(dynamic)
	REP(k, len)
		warper.warp(nowimgs[k], nowkpts[k]);

	vector<Homography> nowmat;		// size = len - 1
	nowmat.resize(len - 1);
	bool failed = false;
#pragma omp parallel for schedule(dynamic)
	REPL(k, 1, len) {
		MatchInfo info;
		bool succ = TransformEstimation(
				matches[k - 1 + mid], nowkpts[k - 1], nowkpts[k],
				nowimgs[k-1], nowimgs[k]).get_transform(&info);
		if (! succ)
			failed = true;
		//error_exit("The two image doesn't match. Failed");
		nowmat[k-1] = info.homo;
	}
	if (failed) return 0;

	REPL(k, 1, len - 1)
		nowmat[k] = nowmat[k - 1] * nowmat[k];	// transform to nowimgs[0] == imgs[mid]

	// check the slope of the result image
	Vec2D center2 = nowmat.back().trans2d(0, 0);
	const float slope = center2.y/ center2.x;
	print_debug("slope: %lf\n", slope);
	if (update_min(minslope, fabs(slope))) {
		bestfactor = nowfactor;
		mat = move(nowmat);
	}
	return slope;
}

Mat32f CylinderStitcher::perspective_correction(const Mat32f& img) {
	int w = img.width(), h = img.height();
	int refw = imgs[bundle.identity_idx].width(),
			refh = imgs[bundle.identity_idx].height();
	auto homo2proj = bundle.get_homo2proj();
	Vec2D proj_min = bundle.proj_range.min;

	vector<Vec2D> corners;
	auto cur = &(bundle.component.front());
	auto to_ref_coor = [&](Vec2D v) {
		v.x *= cur->imgptr->width(), v.y *= cur->imgptr->height();
		Vec homo = cur->homo.trans(v);
		homo.x /= refw, homo.y /= refh;
		Vec2D t_corner = homo2proj(homo);
		t_corner.x *= refw, t_corner.y *= refh;
		t_corner = t_corner - proj_min;
		corners.push_back(t_corner);
	};
	to_ref_coor(Vec2D(-0.5, -0.5));
	to_ref_coor(Vec2D(-0.5, 0.5));
	cur = &(bundle.component.back());
	to_ref_coor(Vec2D(0.5, -0.5));
	to_ref_coor(Vec2D(0.5, 0.5));

	// stretch the four corner to rectangle
	vector<Vec2D> corners_std = {
		Vec2D(0, 0), Vec2D(0, h),
		Vec2D(w, 0), Vec2D(w, h)};
	Matrix m = getPerspectiveTransform(corners, corners_std);
	Homography inv(m);

	LinearBlender blender;
	ImageRef tmp("this_should_not_be_used");
	tmp.img = new Mat32f(img);
	tmp._width = img.width(), tmp._height = img.height();
	blender.add_image(
			Coor(0,0), Coor(w,h), tmp,
			[=](Coor c) -> Vec2D {
		return inv.trans2d(Vec2D(c.x, c.y));
	});
	return blender.run();
}

}

-----------------------------DEBUG.CC---------------------------

//File: debug.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "blender.hh"
#include "multiband.hh"
#include "stitcher.hh"
#include "match_info.hh"

#include <iostream>

#include "lib/utils.hh"
#include "lib/imgproc.hh"
#include "lib/planedrawer.hh"

using namespace std;

namespace pano {

void LinearBlender::debug_run(int w, int h) {
#pragma omp parallel for schedule(dynamic)
	REP(k, (int)images.size()) {
		auto& img = images[k];
		img.imgref.load();
		Mat32f target(h, w, 3);
		fill(target, Color::NO);
		for (int i = 0; i < target.height(); i ++) {
			float *row = target.ptr(i);
			for (int j = 0; j < target.width(); j ++) {
				Color isum = Color::BLACK;
				if (img.range.contain(i, j)) {
					Vec2D img_coor = img.map_coor(i, j);
					if (!img_coor.isNaN()) {
						float r = img_coor.y, c = img_coor.x;
						isum = interpolate(*img.imgref.img, r, c);
					}
				}
				isum.write_to(row + j * 3);
			}
		}
		print_debug("Debug rendering %02d image\n", k);
		write_rgb(ssprintf("log/blend-%02d.jpg", k), target);
	}
}

void MultiBandBlender::debug_level(int level) const {
	int imgid = 0;
	// TODO omp
	for (auto& t: images) {
		auto& wimg = t.img;
		Mat32f img(wimg.rows(), wimg.cols(), 3);
		Mat32f weight(wimg.rows(), wimg.cols(), 3);
		REP(i, wimg.rows()) REP(j, wimg.cols()) {
			if (not t.meta.mask.get(i, j))
				wimg.at(i, j).c.write_to(img.ptr(i, j));
			else
				Color::NO.write_to(img.ptr(i, j));
			float* p = weight.ptr(i, j);
			p[0] = p[1] = p[2] = wimg.at(i, j).w;
		}
		print_debug("[MultiBand] debug output image %d\n", imgid);
		write_rgb(ssprintf("log/multiband%d-%d.jpg", imgid, level), img);
		write_rgb(ssprintf("log/multibandw%d-%d.jpg", imgid, level), weight);
		imgid ++;
	}
}


void Stitcher::draw_matchinfo() {
	int n = imgs.size();
	REP(i, n) imgs[i].load();
#pragma omp parallel for schedule(dynamic)
	REP(i, n) REPL(j, i+1, n) {
		Vec2D offset1(imgs[i].width()/2, imgs[i].height()/2);
		Vec2D offset2(imgs[j].width()/2 + imgs[i].width(), imgs[j].height()/2);
		Shape2D shape2{imgs[j].width(), imgs[j].height()},
						shape1{imgs[i].width(), imgs[i].height()};

		auto& m = pairwise_matches[i][j];
		if (m.confidence <= 0)
			continue;
		list<Mat32f> imagelist{*imgs[i].img, *imgs[j].img};
		Mat32f conc = hconcat(imagelist);
		PlaneDrawer pld(conc);
		for (auto& p : m.match) {
			pld.set_rand_color();
			pld.circle(p.first + offset1, 7);
			pld.circle(p.second + offset2, 7);
			pld.line(p.first + offset1, p.second + offset2);
		}

		pld.set_color(Color(0,0,0));

		Matrix homo(3,3);
		REP(i, 9) homo.ptr()[i] = m.homo[i];
		Homography inv = m.homo.inverse();
		auto p = overlap_region(shape1, shape2, homo, inv);
		for (auto& v: p) v += offset1;
		pld.polygon(p);

		Matrix invM(3, 3);
		REP(i, 9) invM.ptr()[i] = inv[i];
		p = overlap_region(shape2, shape1, invM, m.homo);
		for (auto& v: p) v += offset2;
		pld.polygon(p);

		print_debug("Dump matchinfo of %d->%d\n", i, j);
		write_rgb(ssprintf("log/match%d-%d.jpg", i, j), conc);
	}
}

void Stitcher::dump_matchinfo(const char* fname) const {
	print_debug("Dump matchinfo to %s\n", fname);
	ofstream fout(fname);
	m_assert(fout.good());
	int n = imgs.size();
	REP(i, n) REP(j, n) {
		auto& m = pairwise_matches[i][j];
		if (m.confidence <= 0) continue;
		fout << i << " " << j << endl;
		m.serialize(fout);
		fout << endl;
	}
	fout.close();
}

void Stitcher::load_matchinfo(const char* fname) {
	print_debug("Load matchinfo from %s\n", fname);
	ifstream fin(fname);
	int i, j;
	int n = imgs.size();
	pairwise_matches.resize(n);
	for (auto& k : pairwise_matches) k.resize(n);

	while (true) {
		fin >> i >> j;
		if (fin.eof()) break;
		pairwise_matches[i][j] = MatchInfo::deserialize(fin);
	}
	fin.close();
}

}

----------------------------HOMOGRAPHY.CC---------------------------

//File: homography.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "homography.hh"

#include <Eigen/Dense>
#include <vector>

#include "lib/matrix.hh"
#include "lib/polygon.hh"
#include "match_info.hh"

using namespace std;

namespace {
inline Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>
	to_eigenmap(const pano::Homography& m) {
		return Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
				(double*)m.data, 3, 3);
	}
}

namespace pano {

Homography Homography::inverse(bool* succ) const {
	using namespace Eigen;
	Homography ret;
	auto res = to_eigenmap(ret),
			 input = to_eigenmap(*this);
	FullPivLU<Eigen::Matrix<double,3,3,RowMajor>> lu(input);
	if (succ == nullptr) {
		m_assert(lu.isInvertible());
	} else {
		*succ = lu.isInvertible();
		if (! *succ) return ret;
	}
	res = lu.inverse().eval();
	return ret;
}

Homography Homography::operator * (const Homography& r) const {
	Homography ret;
	auto m1 = to_eigenmap(*this),
			 m2 = to_eigenmap(r),
			 res = to_eigenmap(ret);
	res = m1 * m2;
	return ret;
}

std::vector<Vec2D> overlap_region(
		const Shape2D& shape1, const Shape2D& shape2,
		const Matrix& homo, const Homography& inv) {
	// use sampled edge points, rather than 4 corner, to deal with distorted homography
	// for distorted homography, the range of projected z coordinate contains 0
	// or equivalently, some point is projected to infinity
	const int NR_POINT_ON_EDGE = 100;
	Matrix edge_points(3, 4 * NR_POINT_ON_EDGE);
	float stepw = shape2.w * 1.0 / NR_POINT_ON_EDGE,
				steph = shape2.h * 1.0 / NR_POINT_ON_EDGE;
	REP(i, NR_POINT_ON_EDGE) {
		Vec2D p{-shape2.halfw() + i * stepw, -shape2.halfh()};
		edge_points.at(0, i * 4) = p.x, edge_points.at(1, i * 4) = p.y;
		p = Vec2D{-shape2.halfw() + i * stepw, shape2.halfh()};
		edge_points.at(0, i * 4 + 1) = p.x, edge_points.at(1, i * 4 + 1) = p.y;
		p = Vec2D{-shape2.halfw(), -shape2.halfh() + i * steph};
		edge_points.at(0, i * 4 + 2) = p.x, edge_points.at(1, i * 4 + 2) = p.y;
		p = Vec2D{shape2.halfw(), -shape2.halfh() + i * steph};
		edge_points.at(0, i * 4 + 3) = p.x, edge_points.at(1, i * 4 + 3) = p.y;
	}
	REP(i, 4 * NR_POINT_ON_EDGE)
		edge_points.at(2, i) = 1;
	auto transformed_pts = homo * edge_points;	//3x4n
	vector<Vec2D> pts2in1;
	REP(i, 4 * NR_POINT_ON_EDGE) {
		float denom = 1.0 / transformed_pts.at(2, i);
		Vec2D pin1{transformed_pts.at(0, i) * denom, transformed_pts.at(1, i) * denom};
		if (shape1.shifted_in(pin1))
			pts2in1.emplace_back(pin1);
	}

	// also add 4 corner of 1 to build convex hull, in case some are valid
	auto corners = shape1.shifted_corner();
	for (auto& c : corners) {
		Vec2D cin2 = inv.trans2d(c);
		if (shape2.shifted_in(cin2))
			pts2in1.emplace_back(c);
	}
	auto ret = convex_hull(pts2in1);
	return ret;
}

}

-------------------------INCREMENTAL_BUNDLE.CC-------------------------

//File: incremental_bundle_adjuster.cc
//Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include "incremental_bundle_adjuster.hh"

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <array>

#include "camera.hh"
#include "match_info.hh"
#include "lib/config.hh"
#include "projection.hh"
#include "lib/timer.hh"
using namespace std;
using namespace pano;
using namespace config;

namespace {
const static int NR_PARAM_PER_CAMERA = 6;
const static int NR_TERM_PER_MATCH = 2;
const static bool SYMBOLIC_DIFF = true;
const static int LM_MAX_ITER = 100;
const static float ERROR_IGNORE = 800.f;

inline void camera_to_params(const Camera& c, double* ptr) {
  ptr[0] = c.focal;
  ptr[1] = c.ppx;
  ptr[2] = c.ppy;
  Camera::rotation_to_angle(c.R, ptr[3], ptr[4], ptr[5]);
}

inline void params_to_camera(const double* ptr, Camera& c) {
  c.focal = ptr[0];
  c.ppx = ptr[1];
  c.ppy = ptr[2];
  c.aspect = 1;	// keep it 1
  Camera::angle_to_rotation(ptr[3], ptr[4], ptr[5], c.R);
}

inline Homography cross_product_matrix(double x, double y, double z) {
  return {{ 0 , -z, y,
    z , 0 , -x,
    -y, x , 0}};
}

// See: http://arxiv.org/pdf/1312.0788.pdf
// A compact formula for the derivative of a 3-D rotation in exponential coordinates
// return 3 matrix, each is dR / dvi,
// where vi is the component of the euler-vector of this R
std::array<Homography, 3> dRdvi(const Homography& R) {
  double v[3];
  Camera::rotation_to_angle(R, v[0], v[1], v[2]);
  Vec vvec{v[0], v[1], v[2]};
  double vsqr = vvec.sqr();
  if (vsqr < GEO_EPS_SQR)
    return std::array<Homography, 3>{
        cross_product_matrix(1,0,0),
        cross_product_matrix(0,1,0),
        cross_product_matrix(0,0,1)};
  Homography r = cross_product_matrix(v[0], v[1], v[2]);
  std::array<Homography, 3> ret{r, r, r};
  REP(i, 3) ret[i].mult(v[i]);

  Vec I_R_e{1-R.data[0], -R.data[3], -R.data[6]};
  I_R_e = vvec.cross(I_R_e);
  ret[0] += cross_product_matrix(I_R_e.x, I_R_e.y, I_R_e.z);
  I_R_e = Vec{-R.data[1], 1-R.data[4], -R.data[7]};
  I_R_e = vvec.cross(I_R_e);
  ret[1] += cross_product_matrix(I_R_e.x, I_R_e.y, I_R_e.z);
  I_R_e = Vec{-R.data[2], -R.data[5], 1-R.data[8]};
  I_R_e = vvec.cross(I_R_e);
  ret[2] += cross_product_matrix(I_R_e.x, I_R_e.y, I_R_e.z);

  REP(i, 3) {
    ret[i].mult(1.0 / vsqr);
    ret[i] = ret[i] * R;
  }
  return ret;
}

// dK/dfocal = dKdfocal
static const Homography dKdfocal({
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 0.0});
static const Homography dKdppx({
    0.0, 0.0, 1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0});
static const Homography dKdppy({
    0.0, 0.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 0.0});

}	// namespace

namespace pano {

IncrementalBundleAdjuster::IncrementalBundleAdjuster(
    std::vector<Camera>& cameras):
  result_cameras(cameras),
  index_map(cameras.size())
{ }


void IncrementalBundleAdjuster::add_match(
    int i, int j, const MatchInfo& match) {
  match_pairs.emplace_back(i, j, match);
  match_cnt_prefix_sum.emplace_back(nr_pointwise_match);
  nr_pointwise_match += match.match.size();
  idx_added.insert(i);
  idx_added.insert(j);
}

void IncrementalBundleAdjuster::optimize() {
  if (idx_added.empty())
    error_exit("Calling optimize() without adding any matches!");
  using namespace Eigen;
  update_index_map();
  int nr_img = idx_added.size();
  J = MatrixXd{NR_TERM_PER_MATCH * nr_pointwise_match, NR_PARAM_PER_CAMERA * nr_img};
  JtJ = MatrixXd{NR_PARAM_PER_CAMERA * nr_img, NR_PARAM_PER_CAMERA * nr_img};

  ParamState state;
  for (auto& idx : idx_added)
    state.cameras.emplace_back(result_cameras[idx]);
  state.ensure_params();
  state.cameras.clear();		// TODO why do I need this
  auto err_stat = calcError(state);
  double best_err = err_stat.avg;
  print_debug("BA: init err: %lf\n", best_err);

  int itr = 0;
  int nr_non_decrease = 0;// number of non-decreasing iteration
  inlier_threshold = std::numeric_limits<int>::max();
  while (itr++ < LM_MAX_ITER) {
    auto update = get_param_update(state, err_stat.residuals, LM_LAMBDA);

    ParamState new_state;
    new_state.params = state.get_params();
    REP(i, new_state.params.size())
      new_state.params[i] -= update(i);
    err_stat = calcError(new_state);
    print_debug("BA: average err: %lf, max: %lf\n", err_stat.avg, err_stat.max);

    if (err_stat.avg >= best_err - 1e-3)
      nr_non_decrease ++;
    else {
      nr_non_decrease = 0;
      best_err = err_stat.avg;
      state = move(new_state);
    }
    if (nr_non_decrease > 5)
      break;
  }
  print_debug("BA: Error %lf after %d iterations\n", best_err, itr);

  auto results = state.get_cameras();
  int now = 0;
  for (auto& i : idx_added)
    result_cameras[i] = results[now++];
}

IncrementalBundleAdjuster::ErrorStats IncrementalBundleAdjuster::calcError(
    const ParamState& state) {
  ErrorStats ret(nr_pointwise_match * NR_TERM_PER_MATCH);
  auto cameras = state.get_cameras();

  int idx = 0;
  for (auto& pair: match_pairs) {
    int from = index_map[pair.from],
    to = index_map[pair.to];
    auto& c_from = cameras[from],
    & c_to = cameras[to];
    Homography Hto_to_from = (c_from.K() * c_from.R) *
      (c_to.Rinv() * c_to.K().inverse());

    for (const auto& p: pair.m.match) {
      Vec2D to = p.first, from = p.second;
      // we are estimating Hs that work on [-w/2,w/2] coordinate
      Vec2D transformed = Hto_to_from.trans2d(to);
      ret.residuals[idx] = from.x - transformed.x;
      ret.residuals[idx+1] = from.y - transformed.y;

      idx += 2;
    }
  }
  ret.update_stats(inlier_threshold);
  return ret;
}

void IncrementalBundleAdjuster::ErrorStats::update_stats(int) {
  // TODO which error func to use?
  auto error_func = [&](double diff) -> double {
    return sqr(diff);	// square error is good
    /*
     *diff = fabs(diff);
     *if (diff < inlier_threshold)
     *  return sqr(diff);
     *return 2.0 * inlier_threshold * diff - sqr(inlier_threshold);
     */
  };

  avg = max = 0;
  for (auto& e : residuals) {
    avg += error_func(e);
    update_max(max, fabs(e));
    //if (update_max(max, fabs(e))) PP(e);
  }
  avg /= residuals.size();
  avg = sqrt(avg);

}

Eigen::VectorXd IncrementalBundleAdjuster::get_param_update(
    const ParamState& state, const vector<double>& residual, float lambda) {
  TotalTimer tm("get_param_update");
  using namespace Eigen;
  int nr_img = idx_added.size();
  if (! SYMBOLIC_DIFF) {
    calcJacobianNumerical(state);
  } else {
    //calcJacobianNumerical(state);
    //auto Jcopy = J;
    calcJacobianSymbolic(state);
    // check correctness
    //PP((JtJ - J.transpose() * J).eval().maxCoeff());
    //PP((J - Jcopy).eval().maxCoeff());
  }
  Map<const VectorXd> err_vec(residual.data(), NR_TERM_PER_MATCH * nr_pointwise_match);
  auto b = J.transpose() * err_vec;

  REP(i, nr_img * NR_PARAM_PER_CAMERA) {
    // use different lambda for different param? from Lowe.
    // *= (1+lambda) ?
    if (i % NR_PARAM_PER_CAMERA >= 3) {
      JtJ(i, i) += lambda;
    } else {
      JtJ(i, i) += lambda / 10.f;
    }
  }
  //return JtJ.partialPivLu().solve(b).eval();
  return JtJ.colPivHouseholderQr().solve(b).eval();
}

void IncrementalBundleAdjuster::calcJacobianNumerical(const ParamState& old_state) {
  TotalTimer tm("calcJacobianNumerical");
  // Numerical Differentiation of Residual w.r.t all parameters
  const static double step = 1e-6;
  ParamState& state = const_cast<ParamState&>(old_state); // all mutated state will be recovered at last.
  REP(i, idx_added.size()) {
    REP(p, NR_PARAM_PER_CAMERA) {
      int param_idx = i * NR_PARAM_PER_CAMERA + p;
      double val = state.params[param_idx];
      state.mutate_param(param_idx, val + step);
      auto err1 = calcError(state);
      state.mutate_param(param_idx, val - step);
      auto err2 = calcError(state);
      state.mutate_param(param_idx, val);

      // calc deriv
      REP(k, err1.num_terms())
        J(k, param_idx) = (err1.residuals[k] - err2.residuals[k]) / (2 * step);
    }
  }
  JtJ = (J.transpose() * J).eval();
}

void IncrementalBundleAdjuster::calcJacobianSymbolic(const ParamState& state) {
  // Symbolic Differentiation of Residual w.r.t all parameters
  // See Section 4 of: Automatic Panoramic Image Stitching using Invariant Features - David Lowe,IJCV07.pdf
  TotalTimer tm("calcJacobianSymbolic");
  J.setZero();	// this took 1/3 time. J.rows() could reach 700000 sometimes.
  JtJ.setZero();
  const auto& cameras = state.get_cameras();
  // pre-calculate all derivatives of R
  vector<array<Homography, 3>> all_dRdvi(cameras.size());
  REP(i, cameras.size())
    all_dRdvi[i] = dRdvi(cameras[i].R);

  REP(pair_idx, match_pairs.size()) {
    const MatchPair& pair = match_pairs[pair_idx];
    int idx = match_cnt_prefix_sum[pair_idx] * 2;
    int from = index_map[pair.from],
    to = index_map[pair.to];
    int param_idx_from = from * NR_PARAM_PER_CAMERA,
        param_idx_to = to * NR_PARAM_PER_CAMERA;
    const auto &c_from = cameras[from],
    &c_to = cameras[to];
    const auto fromK = c_from.K();
    const auto toKinv = c_to.Kinv();
    const auto toRinv = c_to.Rinv();
    const auto& dRfromdvi = all_dRdvi[from];
    auto dRtodviT = all_dRdvi[to];	// copying. will modify!
    for (auto& m: dRtodviT) m = m.transpose();

    const Homography Hto_to_from = (fromK * c_from.R) * (toRinv * toKinv);

    for (const auto& p : pair.m.match) {
      Vec2D to = p.first;//, from = p.second;
      Vec homo = Hto_to_from.trans(to);
      double hz_sqr_inv = 1.0 / sqr(homo.z);
      double hz_inv = 1.0 / homo.z;

      Vec dhdv;	// d(homo)/d(variable)
      // calculate d(residual) / d(variable) = -d(point 2d) / d(variable)
      // d(point 2d coor) / d(variable) = d(p2d)/d(homo3d) * d(homo3d)/d(variable)
#define drdv(xx) \
      (dhdv = xx, Vec2D{ \
       -dhdv.x * hz_inv + dhdv.z * homo.x * hz_sqr_inv, \
       -dhdv.y * hz_inv + dhdv.z * homo.y * hz_sqr_inv})
//#define drdv(xx) -flat::gradproj(homo, xx)

      array<Vec2D, NR_PARAM_PER_CAMERA> dfrom, dto;

      // from:
      Homography m = c_from.R * toRinv * toKinv;
      Vec dot_u2 = m.trans(to);
      // focal
      dfrom[0] = drdv(dKdfocal.trans(dot_u2));
      // ppx
      dfrom[1] = drdv(dKdppx.trans(dot_u2));
      // ppy
      dfrom[2] = drdv(dKdppy.trans(dot_u2));
      // rot
      dot_u2 = (toRinv * toKinv).trans(to);
      dfrom[3] = drdv((fromK * dRfromdvi[0]).trans(dot_u2));
      dfrom[4] = drdv((fromK * dRfromdvi[1]).trans(dot_u2));
      dfrom[5] = drdv((fromK * dRfromdvi[2]).trans(dot_u2));

      // to: d(Kinv) / dv = -Kinv * d(K)/dv * Kinv
      m = fromK * c_from.R * toRinv * toKinv;
      dot_u2 = toKinv.trans(to) * (-1);
      // focal
      dto[0] = drdv((m * dKdfocal).trans(dot_u2));
      // ppx
      dto[1] = drdv((m * dKdppx).trans(dot_u2));
      // ppy
      dto[2] = drdv((m * dKdppy).trans(dot_u2));
      // rot
      m = fromK * c_from.R;
      dot_u2 = toKinv.trans(to);
      dto[3] = drdv((m * dRtodviT[0]).trans(dot_u2));
      dto[4] = drdv((m * dRtodviT[1]).trans(dot_u2));
      dto[5] = drdv((m * dRtodviT[2]).trans(dot_u2));
#undef drdv

      // fill J
      REP(i, 6) {
        J(idx, param_idx_from+i) = dfrom[i].x;
        J(idx, param_idx_to+i) = dto[i].x;
        J(idx+1, param_idx_from+i) = dfrom[i].y;
        J(idx+1, param_idx_to+i) = dto[i].y;
      }

      // fill JtJ
      REP(i, 6) REP(j, 6) {
        int i1 = param_idx_from + i,
            i2 = param_idx_to + j;
        auto val = dfrom[i].dot(dto[j]);
        JtJ(i1, i2) += val, JtJ(i2, i1) += val;
      }
      REP(i, 6) REPL(j, i, 6) {
        int i1 = param_idx_from + i,
            i2 = param_idx_from + j;
        auto val = dfrom[i].dot(dfrom[j]);
        JtJ(i1, i2) += val;
        if (i != j) JtJ(i2, i1) += val;

        i1 = param_idx_to + i, i2 = param_idx_to + j;
        val = dto[i].dot(dto[j]);
        JtJ(i1, i2) += val;
        if (i != j) JtJ(i2, i1) += val;
      }
      idx += 2;
    }
  }
}

vector<Camera>& IncrementalBundleAdjuster::ParamState::get_cameras() {
  if (cameras.size())
    return cameras;
  m_assert(params.size());
  cameras.resize(params.size() / NR_PARAM_PER_CAMERA);
  REP(i, cameras.size())
    params_to_camera(params.data() + i * NR_PARAM_PER_CAMERA, cameras[i]);
  return cameras;
}

void IncrementalBundleAdjuster::ParamState::ensure_params() const {
  if (params.size())
    return;
  m_assert(cameras.size());
  // this function serves the semantic of 'updating the cache'. thus it is const
  vector<double>& params = const_cast<vector<double>&>(this->params);
  params.resize(cameras.size() * NR_PARAM_PER_CAMERA);
  REP(i, cameras.size())
    camera_to_params(cameras[i], params.data() + i * NR_PARAM_PER_CAMERA);
}

void IncrementalBundleAdjuster::ParamState::mutate_param(int param_idx, double new_val) {
  ensure_params();
  auto& cameras = get_cameras();
  int camera_id = param_idx / NR_PARAM_PER_CAMERA;
  params[param_idx] = new_val;
  params_to_camera(
      params.data() + camera_id * NR_PARAM_PER_CAMERA,
      cameras[camera_id]);
}
}

---------------------------MULTIBAND.CC---------------------------

//File: multiband.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "multiband.hh"
#include "lib/imgproc.hh"
#include "feature/gaussian.hh"

using namespace std;
namespace pano {
void MultiBandBlender::add_image(
			const Coor& upper_left,
			const Coor& bottom_right,
			ImageRef &img,
			std::function<Vec2D(Coor)> coor_func) {
	images_to_add.emplace_back(ImageToAdd{Range{upper_left, bottom_right}, img, coor_func});
	target_size.update_max(bottom_right);
}

void MultiBandBlender::create_first_level() {
	GUARDED_FUNC_TIMER;

	int nr_image = images_to_add.size();
	meta_images.reserve(nr_image);	// we will need reference to this vector element
#pragma omp parallel for schedule(dynamic)
	REP(k, nr_image) {
		ImageToAdd& img = images_to_add[k];
		img.imgref.load();

		auto& range = img.range;
		Mat<WeightedPixel> wimg(range.height(), range.width(), 1);
		Mask2D mask(range.height(), range.width());
		REP(i, range.height()) REP(j, range.width()) {
			Coor target_coor{j + range.min.x, i + range.min.y};
			Vec2D orig_coor = img.coor_func(target_coor);
			Color c = interpolate(*img.imgref.img, orig_coor.y, orig_coor.x);
			if (c.get_min() < 0) {	// Color::NO
				wimg.at(i, j).w = 0;
				wimg.at(i, j).c = Color::BLACK;	// -1 will mess up with gaussian blur
				mask.set(i, j);
			} else {
				wimg.at(i, j).c = c;
				orig_coor.x = orig_coor.x / img.imgref.width() - 0.5;
				orig_coor.y = orig_coor.y / img.imgref.height() - 0.5;
				wimg.at(i, j).w = std::max(0.0,
						(0.5f - fabs(orig_coor.x)) * (0.5f - fabs(orig_coor.y))) + EPS;
				// ext? eps?
			}
		}
		img.imgref.release();
#pragma omp critical
		{
			meta_images.emplace_back(MetaImage{range, move(mask)});
			images.emplace_back(ImageToBlend{move(wimg), meta_images.back()});
		}
	}
	images_to_add.clear();
}

Mat32f MultiBandBlender::run() {
	create_first_level();
	update_weight_map();
	Mat32f target(target_size.y, target_size.x, 3);
	fill(target, Color::NO);

	Mask2D target_mask(target_size.y, target_size.x);

	for (auto& m : images)
		next_lvl_images.emplace_back(m);
	for (int level = 0; level < band_level; level ++) {
		bool is_last = (level == band_level - 1);
		GuardedTimer tmm("Blending level " + to_string(level));
		if (!is_last)
			create_next_level(level);
		//debug_level(level);
#pragma omp parallel for schedule(dynamic)
		REP(i, target_size.y) REP(j, target_size.x) {
			Color isum(0, 0, 0);
			float wsum = 0;
			REP(imgid, images.size())  {
				auto& img_cur = images[imgid];
				if (not img_cur.meta.range.contain(i, j)) continue;
				if (not img_cur.valid_on_target(j, i)) continue;

				float w = img_cur.weight_on_target(j, i);
				if (w <= 0) continue;

				auto& ccur = img_cur.color_on_target(j, i);

				if (not is_last) {
					auto & img_next = next_lvl_images[imgid];
					auto& cnext = img_next.color_on_target(j, i);
					isum += (ccur - cnext) * w;
				} else {
					isum += ccur * w;
				}
				wsum += w;
			}
			if (wsum < EPS)
				continue;
			isum /= wsum;
			float* p = target.ptr(i, j);
			if (not target_mask.get(i, j)) {		// first time to visit *p. Note that *p could be negative after visit.
				isum.write_to(p);
				target_mask.set(i, j);
			} else {
				p[0] += isum.x, p[1] += isum.y, p[2] += isum.z;
			}
		}
		swap(next_lvl_images, images);
	}
	images.clear(); next_lvl_images.clear();

	REP(i, target.rows()) REP(j, target.cols()) {
		if (target_mask.get(i, j)) {
			float* p = target.ptr(i, j);
			// weighted laplacian pyramid might introduce minor over/under flow
			p[0] = max(min(p[0], 1.0f), 0.f);
			p[1] = max(min(p[1], 1.0f), 0.f);
			p[2] = max(min(p[2], 1.0f), 0.f);
		}
	}
	return target;
}

void MultiBandBlender::update_weight_map() {
	GUARDED_FUNC_TIMER;
#pragma omp parallel for schedule(dynamic, 100)
	REP(i, target_size.y) REP(j, target_size.x) {
		float max = 0.f;
		float* maxp = nullptr;
		for (auto& img : images) {
			if (img.meta.range.contain(i, j)) {
				float& w = img.weight_on_target(j, i);
				if (w > max) {
					max = w;
					maxp = &w;
				}
				w = 0;
			}
		}
		if (maxp) *maxp = 1;
	}
}

void MultiBandBlender::create_next_level(int level) {
	TOTAL_FUNC_TIMER;
	GaussianBlur blurer(sqrt(level * 2 + 1.0) * 4);	// TODO size
#pragma omp parallel for schedule(dynamic)
	REP(i, (int)images.size())
		next_lvl_images[i].img = blurer.blur(images[i].img);
}


}	// namespace pano

---------------------------STITCHER.CC---------------------------

// File: stitcher.cc
// Date: Sun Sep 22 12:54:18 2013 +0800
// Author: Yuxin Wu <ppwwyyxxc@gmail.com>


#include "stitcher.hh"

#include <limits>
#include <string>
#include <cmath>
#include <queue>

#include "feature/matcher.hh"
#include "lib/imgproc.hh"
#include "lib/timer.hh"
#include "blender.hh"
#include "match_info.hh"
#include "transform_estimate.hh"
#include "camera_estimator.hh"
#include "camera.hh"
#include "warp.hh"
using namespace std;
using namespace pano;
using namespace config;

namespace pano {

// use in development
const static bool DEBUG_OUT = false;
const static char* MATCHINFO_DUMP = "log/matchinfo.txt";

Mat32f Stitcher::build() {
  calc_feature();
  // TODO choose a better starting point by MST use centrality

  pairwise_matches.resize(imgs.size());
  for (auto& k : pairwise_matches) k.resize(imgs.size());
  if (ORDERED_INPUT)
    linear_pairwise_match();
  else
    pairwise_match();
  free_feature();
  //load_matchinfo(MATCHINFO_DUMP);
  if (DEBUG_OUT) {
    draw_matchinfo();
    dump_matchinfo(MATCHINFO_DUMP);
  }
  assign_center();

  if (ESTIMATE_CAMERA)
    estimate_camera();
  else
    build_linear_simple();		// naive mode
  pairwise_matches.clear();
  // TODO automatically determine projection method even in naive mode
  if (ESTIMATE_CAMERA)
    bundle.proj_method = ConnectedImages::ProjectionMethod::spherical;
  else
    bundle.proj_method = ConnectedImages::ProjectionMethod::flat;
  print_debug("Using projection method: %d\n", bundle.proj_method);
  bundle.update_proj_range();
  return bundle.blend();
}

bool Stitcher::match_image(
    const PairWiseMatcher& pwmatcher, int i, int j) {
  auto match = pwmatcher.match(i, j);
  TransformEstimation transf(
      match, keypoints[i], keypoints[j],
      imgs[i].shape(), imgs[j].shape());	// from j to i. H(p_j) ~= p_i
  MatchInfo info;
  bool succ = transf.get_transform(&info);
  if (!succ) {
    if (-(int)info.confidence >= 8)	// reject for geometry reason
      print_debug("Reject bad match with %d inlier from %d to %d\n",
          -(int)info.confidence, i, j);
    return false;
  }
  auto inv = info.homo.inverse();	// TransformEstimation ensures invertible
  inv.mult(1.0 / inv[8]);	// TODO more stable?
  print_debug(
      "Connection between image %d and %d, ninliers=%lu/%d=%lf, conf=%f\n",
      i, j, info.match.size(), match.size(),
      info.match.size() * 1.0 / match.size(),
      info.confidence);

  // fill in pairwise matches
  pairwise_matches[i][j] = info;
  info.homo = inv;
  info.reverse();
  pairwise_matches[j][i] = move(info);
  return true;
}

void Stitcher::pairwise_match() {
  GuardedTimer tm("pairwise_match()");
  size_t n = imgs.size();
  vector<pair<int, int>> tasks;
  REP(i, n) REPL(j, i + 1, n) tasks.emplace_back(i, j);

  PairWiseMatcher pwmatcher(feats);

  int total_nr_match = 0;

#pragma omp parallel for schedule(dynamic)
  REP(k, (int)tasks.size()) {
    int i = tasks[k].first, j = tasks[k].second;
    bool succ = match_image(pwmatcher, i, j);
    if (succ)
      total_nr_match += pairwise_matches[i][j].match.size();
  }
  print_debug("Total number of matched keypoint pairs: %d\n", total_nr_match);
}

void Stitcher::linear_pairwise_match() {
  GuardedTimer tm("linear_pairwise_match()");
  int n = imgs.size();
  PairWiseMatcher pwmatcher(feats);
#pragma omp parallel for schedule(dynamic)
  REP(i, n) {
    int next = (i + 1) % n;
    if (!match_image(pwmatcher, i, next)) {
      if (i == n - 1)	// head and tail don't have to match
        continue;
      else
        error_exit(ssprintf("Image %d and %d don't match\n", i, next));
    }
    continue; // TODO FIXME a->b, b->a
    do {
      next = (next + 1) % n;
      if (next == i)
        break;
    } while (match_image(pwmatcher, i, next));
  }
}

void Stitcher::assign_center() {
  bundle.identity_idx = imgs.size() >> 1;
  //bundle.identity_idx = 0;
}

void Stitcher::estimate_camera() {
  vector<Shape2D> shapes;
  for (auto& m: imgs) shapes.emplace_back(m.shape());
  auto cameras = CameraEstimator{pairwise_matches, shapes}.estimate();

  // produced homo operates on [-w/2,w/2] coordinate
  REP(i, imgs.size()) {
    bundle.component[i].homo_inv = cameras[i].K() * cameras[i].R;
    bundle.component[i].homo = cameras[i].Rinv() * cameras[i].K().inverse();
  }
}

void Stitcher::build_linear_simple() {
  // TODO bfs over pairwise to build bundle
  // assume pano pairwise
  int n = imgs.size(), mid = bundle.identity_idx;
  bundle.component[mid].homo = Homography::I();

  auto& comp = bundle.component;

  // accumulate the transformations
  if (mid + 1 < n) {
    comp[mid+1].homo = pairwise_matches[mid][mid+1].homo;
    REPL(k, mid + 2, n)
      comp[k].homo = comp[k - 1].homo * pairwise_matches[k-1][k].homo;
  }
  if (mid - 1 >= 0) {
    comp[mid-1].homo = pairwise_matches[mid][mid-1].homo;
    REPD(k, mid - 2, 0)
      comp[k].homo = comp[k + 1].homo * pairwise_matches[k+1][k].homo;
  }
  // comp[k]: from k to identity. [-w/2,w/2]

  // when estimate_camera is not used, homo is KRRK(2d-2d), not KR(2d-3d)
  // need to somehow normalize(guess) focal length to make non-flat projection work
  double f = -1;
  if (not TRANS)    // the estimation method only works under fixed-center projection
    f = Camera::estimate_focal(pairwise_matches);
  if (f <= 0) {
    print_debug("Cannot estimate focal. Will use a naive one.\n");
    f = 0.5 * (imgs[mid].width() + imgs[mid].height());
  }
  REP(i, n) {
    auto M = Homography{{
        1.0/f, 0,     0,
        0,     1.0/f, 0,
        0,     0,     1
    }};
    comp[i].homo = M * comp[i].homo;
  }
  bundle.calc_inverse_homo();
}

}	// namepsace pano

------------------------STITCHER_BASE.CC---------------------------

//File: stitcherbase.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#include "stitcherbase.hh"
#include "lib/timer.hh"

namespace pano {

void StitcherBase::calc_feature() {
  GuardedTimer tm("calc_feature()");
  feats.resize(imgs.size());
  keypoints.resize(imgs.size());
  // detect feature
#pragma omp parallel for schedule(dynamic)
  REP(k, (int)imgs.size()) {
    imgs[k].load();
    feats[k] = feature_det->detect_feature(*imgs[k].img);
    if (config::LAZY_READ)
      imgs[k].release();
    if (feats[k].size() == 0)
      error_exit(ssprintf("Cannot find feature in image %d!\n", k));
    print_debug("Image %d has %lu features\n", k, feats[k].size());
    keypoints[k].resize(feats[k].size());
    REP(i, feats[k].size())
      keypoints[k][i] = feats[k][i].coor;
  }
}

void StitcherBase::free_feature() {
  feats.clear(); feats.shrink_to_fit();  // free memory for feature
  keypoints.clear(); keypoints.shrink_to_fit();  // free memory for feature
}

}


---------------------------STITCHER_IMAGE.CC---------------------------

//File: stitcher_image.cc
//Author: Yuxin Wu <ppwwyyxx@gmail.com>

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <cassert>
#include <memory>
#include "stitcher_image.hh"
#include "projection.hh"
#include "lib/config.hh"
#include "lib/timer.hh"
#include "multiband.hh"
#include "lib/imgproc.hh"
#include "blender.hh"

using namespace std;
using namespace config;

namespace pano {

void ConnectedImages::shift_all_homo() {
	int mid = identity_idx;
	Homography t2 = Homography::get_translation(
			component[mid].imgptr->width() * 0.5,
			component[mid].imgptr->height() * 0.5);
	REP(i, (int)component.size())
		if (i != mid) {
			Homography t1 = Homography::get_translation(
					component[i].imgptr->width() * 0.5,
					component[i].imgptr->height() * 0.5);
			component[i].homo = t2 * component[i].homo * t1.inverse();
		}
}

void ConnectedImages::calc_inverse_homo() {
	for (auto& m : component)
		m.homo_inv = m.homo.inverse();
}

void ConnectedImages::update_proj_range() {
	vector<Vec2D> corner;
	const static int CORNER_SAMPLE = 100;
	REP(i, CORNER_SAMPLE) REP(j, CORNER_SAMPLE)
		corner.emplace_back((double)i / CORNER_SAMPLE - 0.5, (double)j / CORNER_SAMPLE - 0.5);

	auto homo2proj = get_homo2proj();

	Vec2D proj_min = Vec2D::max();
	Vec2D proj_max = proj_min * (-1);
	for (auto& m : component) {
		Vec2D now_min(numeric_limits<double>::max(), std::numeric_limits<double>::max()),
					now_max = now_min * (-1);
		for (auto v : corner) {
			Vec homo = m.homo.trans(
					Vec2D(v.x * m.imgptr->width(), v.y * m.imgptr->height()));
			Vec2D t_corner = homo2proj(homo);
			now_min.update_min(t_corner);
			now_max.update_max(t_corner);
		}
		m.range = Range(now_min, now_max);
		proj_min.update_min(now_min);
		proj_max.update_max(now_max);
		print_debug("Range: (%lf,%lf)~(%lf,%lf)\n",
				m.range.min.x, m.range.min.y,
				m.range.max.x, m.range.max.y);
	}
	proj_range.min = proj_min, proj_range.max = proj_max;
}

Vec2D ConnectedImages::get_final_resolution() const {
	cout << "projmin: " << proj_range.min << ", projmax: " << proj_range.max << endl;

	int refw = component[identity_idx].imgptr->width(),
			refh = component[identity_idx].imgptr->height();
	auto homo2proj = get_homo2proj();
  const Homography& identity_H = component[identity_idx].homo;
  // transform corners to point in space to estimate range

  Vec id_img_corner2 = identity_H.trans(Vec2D{refw/2.0, refh/2.0}),
      id_img_corner1 = identity_H.trans(Vec2D{-refw/2.0, -refh/2.0});
  // the range of the identity image
  Vec2D id_img_range = homo2proj(id_img_corner2) - homo2proj(id_img_corner1);
  cout << "Identity projection range: " << id_img_range << endl;
  if (proj_method != ProjectionMethod::flat) {
    if (id_img_range.x < 0)
      id_img_range.x = 2 * M_PI + id_img_range.x;
    if (id_img_range.y < 0)
      id_img_range.y = M_PI + id_img_range.y;
  }

	Vec2D resolution = id_img_range / Vec2D(refw, refh),		// output-x-per-input-pixel, y-per-pixel
				target_size = proj_range.size() / resolution;
	double max_edge = max(target_size.x, target_size.y);
  print_debug("Target Image Size: (%lf, %lf)\n", target_size.x, target_size.y);
	if (max_edge > 80000 || target_size.x * target_size.y > 1e9)
		error_exit("Target size too large. Looks like a stitching failure!\n");
	// resize the result
	if (max_edge > MAX_OUTPUT_SIZE) {
		float ratio = max_edge / MAX_OUTPUT_SIZE;
		resolution *= ratio;
	}
  print_debug("Resolution: %lf,%lf\n", resolution.x, resolution.y);
	return resolution;
}

Mat32f ConnectedImages::blend() const {
	GuardedTimer tm("blend()");
	// it's hard to do coordinates.......
	auto proj2homo = get_proj2homo();
	Vec2D resolution = get_final_resolution();

	Vec2D size_d = proj_range.size() / resolution;
	Coor size(size_d.x, size_d.y);
	print_debug("Final Image Size: (%d, %d)\n", size.x, size.y);

	auto scale_coor_to_img_coor = [&](Vec2D v) {
		v = (v - proj_range.min) / resolution;
		return Coor(v.x, v.y);
	};

	// blending
	std::unique_ptr<BlenderBase> blender;
	if (MULTIBAND > 0)
		blender.reset(new MultiBandBlender{MULTIBAND});
	else
		blender.reset(new LinearBlender);
	for (auto& cur : component) {
		Coor top_left = scale_coor_to_img_coor(cur.range.min);
		Coor bottom_right = scale_coor_to_img_coor(cur.range.max);

		blender->add_image(top_left, bottom_right, *cur.imgptr,
				[=,&cur](Coor t) -> Vec2D {
					Vec2D c = Vec2D(t.x, t.y) * resolution + proj_range.min;
					Vec homo = proj2homo(Vec2D(c.x, c.y));
					Vec ret = cur.homo_inv.trans(homo);
					if (ret.z < 0)
						return Vec2D{-10, -10};	// was projected to the other side of the lens, discard
					double denom = 1.0 / ret.z;
					return Vec2D{ret.x*denom, ret.y*denom}
                + cur.imgptr->shape().center();
				});
	}
	//dynamic_cast<LinearBlender*>(blender.get())->debug_run(size.x, size.y);	// for debug
	return blender->run();
}

}


-------------------------TRANSFORM_ESTIMATE.CC---------------------------

// File: transform_estimate.cc
// Date: Fri May 03 23:04:58 2013 +0800
// Author: Yuxin Wu <ppwwyyxxc@gmail.com>

#include "transform_estimate.hh"

#include <set>
#include <random>

#include "feature/feature.hh"
#include "feature/matcher.hh"
#include "lib/polygon.hh"
#include "lib/config.hh"
#include "lib/imgproc.hh"
#include "lib/timer.hh"
#include "match_info.hh"
using namespace std;
using namespace config;

namespace {
const int ESTIMATE_MIN_NR_MATCH = 8;
}

namespace pano {

TransformEstimation::TransformEstimation(const MatchData& m_match,
		const std::vector<Vec2D>& kp1,
		const std::vector<Vec2D>& kp2,
		const Shape2D& shape1, const Shape2D& shape2):
	match(m_match), kp1(kp1), kp2(kp2),
	shape1(shape1), shape2(shape2),
	f2_homo_coor(match.size(), 3)
{
	if (CYLINDER || TRANS)
		transform_type = Affine;
	else
		transform_type = Homo;
	int n = match.size();
	if (n < ESTIMATE_MIN_NR_MATCH) return;
	REP(i, n) {
		Vec2D old = kp2[match.data[i].second];
		f2_homo_coor.at(i, 0) = old.x;
		f2_homo_coor.at(i, 1) = old.y;
		f2_homo_coor.at(i, 2) = 1;
	}
	ransac_inlier_thres = (shape1.w + shape1.h) * 0.5 / 800 * RANSAC_INLIER_THRES;
}

bool TransformEstimation::get_transform(MatchInfo* info) {
	TotalTimer tm("get_transform");
	// use Affine in cylinder mode, and Homography in normal mode
	// TODO more condidate set will require more ransac iterations
	int nr_match_used = (transform_type == Affine ? 6: 8) / 2 + 4;
	int nr_match = match.size();
	if (nr_match < nr_match_used)
		return false;

	vector<int> inliers;
	set<int> selected;

	int maxinlierscnt = -1;
	Homography best_transform;

	random_device rd;
	mt19937 rng(rd());

	for (int K = RANSAC_ITERATIONS; K --;) {
		inliers.clear();
		selected.clear();
		REP(_, nr_match_used) {
			int random;
			do {
				random = rng() % nr_match;
			} while (selected.find(random) != selected.end());
			selected.insert(random);
			inliers.push_back(random);
		}
		auto transform = calc_transform(inliers);
		if (! transform.health())
			continue;
		int n_inlier = get_inliers(transform).size();
		if (update_max(maxinlierscnt, n_inlier))
			best_transform = move(transform);
	}
	inliers = get_inliers(best_transform);
	return fill_inliers_to_matchinfo(inliers, info);
}

Homography TransformEstimation::calc_transform(const vector<int>& matches) const {
	vector<Vec2D> p1, p2;
	for (auto& i : matches) {
		p1.emplace_back(kp1[match.data[i].first]);
		p2.emplace_back(kp2[match.data[i].second]);
	}
	//return ((transform_type == Affine) ? getAffineTransform : getPerspectiveTransform)(p1, p2);

	// normalize the coordinates before DLT, so that points are zero-centered and have sqrt(2) mean distance to origin.
	// suggested by MVG Sec 4.4
	auto normalize = [](vector<Vec2D>& pts) {
		double sizeinv = 1.0 / pts.size();
		Vec2D mean{0, 0};
		// TODO it seems like when camera distortion is severe,
		// mean-subtract leads to more chances of failure
		/*
		 *for (const auto& p : pts) mean += p * sizeinv;
		 *for (auto& p : pts) p -= mean;
		 */

		double sqrsum = 0;
		for (const auto& p : pts) sqrsum += p.sqr() * sizeinv;
		double div_inv = sqrt(2.0 / sqrsum);
		for (auto& p : pts) p *= div_inv;
		return make_pair(mean, div_inv);
	};
	auto param1 = normalize(p1),
			 param2 = normalize(p2);

	// homo from p2 to p1
	Matrix homo = ((transform_type == Affine) ? getAffineTransform : getPerspectiveTransform)(p1, p2);

	Homography t1{{param1.second, 0, -param1.second * param1.first.x,
								 0, param1.second, -param1.second * param1.first.y,
								 0, 0, 1}};
	Homography t2{{param2.second, 0, -param2.second * param2.first.x,
								 0, param2.second, -param2.second * param2.first.y,
								 0, 0, 1}};
	// return transform on non-normalized coordinate
	Homography ret = t1.inverse() * Homography{homo} * t2;
	return ret;
}

vector<int> TransformEstimation::get_inliers(const Homography& trans) const {
	float INLIER_DIST = sqr(ransac_inlier_thres);
	TotalTimer tm("get_inlier");
	vector<int> ret;
	int n = match.size();

	Matrix transformed = f2_homo_coor.prod(trans.transpose().to_matrix());	// nx3
	REP(i, n) {
		const Vec2D& fcoor = kp1[match.data[i].first];
		double* ptr = transformed.ptr(i);
		double idenom = 1.f / ptr[2];
		double dist = (Vec2D{ptr[0] * idenom, ptr[1] * idenom} - fcoor).sqr();
		if (dist < INLIER_DIST)
			ret.push_back(i);
	}
	return ret;
}

bool TransformEstimation::fill_inliers_to_matchinfo(
		const std::vector<int>& inliers, MatchInfo* info) const {
	TotalTimer tm("fill inliers");
	info->confidence = -(float)inliers.size();		// only for debug
	if (inliers.size() < ESTIMATE_MIN_NR_MATCH)
		return false;

	// get the number of matched point in the polygon in the first/second image
	auto get_match_cnt = [&](vector<Vec2D>& poly, bool first) {
		if (poly.size() < 3) return 0;
		auto pip = PointInPolygon(poly);
		int cnt = 0;
		for (auto& p : match.data)
			if (pip.in_polygon(first ? kp1[p.first] : kp2[p.second]))
				cnt ++;
		return cnt;
	};
	// get the number of keypoint in the polygon
  // TODO shouldn't count undistinctive keypoints as keypoints. They should get filtered out earlier
	auto get_keypoint_cnt = [&](vector<Vec2D>& poly, bool first) {
		auto pip = PointInPolygon{poly};
		int cnt = 0;
		for (auto& p : first ? kp1 : kp2)
			if (pip.in_polygon(p))
				cnt ++;
		return cnt;
	};

	auto homo = calc_transform(inliers);			// from 2 to 1
	Matrix homoM = homo.to_matrix();
	bool succ = false;
	Homography inv = homo.inverse(&succ);
	if (not succ)	// cannot inverse. ill-formed.
		return false;
  // TODO guess if two images are identical
	auto overlap = overlap_region(shape1, shape2, homoM, inv);
	float r1m = inliers.size() * 1.0f / get_match_cnt(overlap, true);
	if (r1m < INLIER_IN_MATCH_RATIO) return false;
	float r1p = inliers.size() * 1.0f / get_keypoint_cnt(overlap, true);
	if (r1p < 0.01 || r1p > 1) return false;

	Matrix invM = inv.to_matrix();
	overlap = overlap_region(shape2, shape1, invM, homo);
	float r2m = inliers.size() * 1.0f / get_match_cnt(overlap, false);
	if (r2m < INLIER_IN_MATCH_RATIO) return false;
	float r2p = inliers.size() * 1.0f / get_keypoint_cnt(overlap, false);
	if (r2p < 0.01 || r2p > 1) return false;
	print_debug("r1mr1p: %lf,%lf, r2mr2p: %lf,%lf\n", r1m, r1p, r2m, r2p);

	info->confidence = (r1p + r2p) * 0.5;
	if (info->confidence < INLIER_IN_POINTS_RATIO)
		return false;

	// fill in result
	info->homo = homo;
	info->match.clear();
	for (auto& idx : inliers)
		info->match.emplace_back(
				kp1[match.data[idx].first],
				kp2[match.data[idx].second]);
	return true;
}

}


------------------------------WARP.CC---------------------------

// File: warp.cc
// Date: Thu Jul 04 11:43:00 2013 +0800
// Author: Yuxin Wu <ppwwyyxxc@gmail.com>


#include "warp.hh"
#include "lib/imgproc.hh"
using namespace std;
using namespace pano;

namespace pano {

Vec2D CylinderProject::proj(const Vec& p) const {
	real_t x = atan((p.x - center.x) / r);
	real_t y = (p.y - center.y) / (hypot(p.x - center.x, r));
	return Vec2D(x, y);
}

Vec2D CylinderProject::proj_r(const Vec2D& p) const {
	real_t x = r * tan(p.x) + center.x;
	real_t y = p.y * r / cos(p.x) + center.y;
	return Vec2D(x, y);
}

Mat32f CylinderProject::project(const Mat32f& img, vector<Vec2D>& pts) const {
	Shape2D shape{img.width(), img.height()};
	Vec2D offset = project(shape, pts);

	real_t sizefactor_inv = 1.0 / sizefactor;

	Mat32f mat(shape.h, shape.w, 3);
	fill(mat, Color::NO);
#pragma omp parallel for schedule(dynamic)
	REP(i, mat.height()) REP(j, mat.width()) {
		Vec2D oricoor = proj_r((Vec2D(j, i) - offset) * sizefactor_inv);
		if (between(oricoor.x, 0, img.width()) && between(oricoor.y, 0, img.height())) {
			Color c = interpolate(img, oricoor.y, oricoor.x);
			float* p = mat.ptr(i, j);
			p[0] = c.x, p[1] = c.y, p[2] = c.z;
		}
	}

	return mat;
}

Vec2D CylinderProject::project(Shape2D& shape, std::vector<Vec2D>& pts) const {
	Vec2D min(numeric_limits<real_t>::max(), numeric_limits<real_t>::max()),
				max(0, 0);
	REP(i, shape.h) REP(j, shape.w) {			// TODO finally: only use rect corners
		Vec2D newcoor = proj(Vec2D(j, i));
		min.update_min(newcoor), max.update_max(newcoor);
	}

	max = max * sizefactor, min = min * sizefactor;
	Vec2D realsize = max - min,
		    offset = min * (-1);
	Coor size = Coor(realsize.x, realsize.y);

	for (auto & f : pts) {
		Vec2D coor(f.x + shape.w / 2, f.y + shape.h / 2);
		f = proj(coor) * sizefactor + offset;
		f.x -= size.x / 2;
		f.y -= size.y / 2;
	}
	shape.w = size.x, shape.h = size.y;
	return offset;
}


CylinderProject CylinderWarper::get_projector(int w, int h) const {
	// 43.266 = hypot(36, 24)
	int r = hypot(w, h) * (config::FOCAL_LENGTH / 43.266);
	Vec cen(w / 2, h / 2 * h_factor, r);
	return CylinderProject(r, cen, r);
}

}
