// =============================
// include/vsnn/Perceptron.hpp (optional demo)
// =============================
#pragma once
#include <vector>
#include "Matrix.hpp"

using namespace std;

namespace vsnn {
	class PerceptronBinary {
	private:
		Matrix W_; // 1 x D
		float bias_ = 0.0f;
		static int Sign(float v) { return v >= 0.0f ? 1 : -1; }
	public:
		explicit PerceptronBinary(int dim) : W_(1, dim) { W_.Fill(0.0f); }
		void FitEpoch(const Matrix& X, const vector<int>& y) {
			const int N = X.Rows();
			const int D = X.Cols();
			for (int n = 0; n < N; ++n) {
				float s = bias_;
				for (int d = 0; d < D; ++d) s += W_(0, d) * X(n, d);
				const int target = (y[n] == 0) ? -1 : 1;
				if (Sign(s) != target) {
					for (int d = 0; d < D; ++d) W_(0, d) += static_cast<float>(target) * X(n, d);
					bias_ += static_cast<float>(target);
				}
			}
		}
		int PredictOne(const float* x, int D) const {
			float s = bias_;
			for (int d = 0; d < D; ++d) s += W_(0, d) * x[d];
			return (s >= 0.0f) ? 1 : 0;
		}
	};


	class PerceptronOVR {
	private:
		vector<PerceptronBinary> clfs_;
	public:
		PerceptronOVR(int dim, int num_classes) {
			clfs_.reserve(num_classes);
			for (int c = 0; c < num_classes; ++c) clfs_.emplace_back(dim);
		}
		void FitEpoch(const Matrix& X, const vector<int>& y) {
			const int C = static_cast<int>(clfs_.size());
			for (int c = 0; c < C; ++c) {
				vector<int> yb(y.size());
				for (size_t i = 0; i < y.size(); ++i) yb[i] = (y[i] == c) ? 1 : 0;
				clfs_[c].FitEpoch(X, yb);
			}
		}
		int PredictOne(const float* x, int D) const {
			// 간단 데모용
			int bestc = 0; float bests = -1e30f;
			for (int c = 0; c < static_cast<int>(clfs_.size()); ++c) {
				float s = 0.0f; // 간단 점수 (bias 포함 구현 생략)
				// 여기서는 Binary 클래스를 확장해 score를 노출하는 식으로 개선 가능
				s = (float)clfs_[c].PredictOne(x, D);
				if (s > bests) { bests = s; bestc = c; }
			}
			return bestc;
		}
	};
}