// =============================
// include/vsnn/Dense.hpp
// =============================
#pragma once
#include <vector>       
#include <algorithm>    
#include <random>       
#include "Layer.hpp"
#include "Ops.hpp"
#include "Initializer.hpp"

namespace vsnn {
	class Dense : public Layer {
	private:
		Matrix W_, b_;
		Matrix gW_, gb_;
		i32 in_ = 0, out_ = 0;
		std::vector<float> Wv_;         
		std::vector<float> bv_;         
		std::vector<float> gWv_;        
		std::vector<float> gbv_;
		inline float& Wv(i32 k, i32 j) { return Wv_[static_cast<size_t>(k) * out_ + j]; }
		inline float  Wv(i32 k, i32 j) const { return Wv_[static_cast<size_t>(k) * out_ + j]; }

		// Matrix ↔ Vector 동기화 헬퍼
		void SyncMatrixToVec_() {
			in_ = W_.Rows(); out_ = W_.Cols();
			Wv_.assign(static_cast<size_t>(in_) * out_, 0.f);
			for (i32 k = 0; k < in_; ++k) for (i32 j = 0; j < out_; ++j) Wv(k, j) = W_(k, j);
			bv_.assign(out_, 0.f);
			for (i32 j = 0; j < out_; ++j) bv_[j] = b_(0, j);
			gWv_.assign(static_cast<size_t>(in_) * out_, 0.f);
			gbv_.assign(out_, 0.f);
		}
		// Vector grad -> Matrix grad (Updater 호환용, 역전파 끝에 1회)
		void SyncGradVecToMatrix_() {
			if (gW_.Rows() != in_ || gW_.Cols() != out_) gW_.Reset(in_, out_);
			for (i32 k = 0; k < in_; ++k) for (i32 j = 0; j < out_; ++j)
				gW_(k, j) = gWv_[static_cast<size_t>(k) * out_ + j];
			if (gb_.Rows() != 1 || gb_.Cols() != out_) gb_.Reset(1, out_);
			for (i32 j = 0; j < out_; ++j) gb_(0, j) = gbv_[j];
		}
		// (빠른) Matrix -> Vector (가중치/바이어스) 매-Forward 직전 동기화
		void SyncWeightMatrixToVec_() {
			// 차원 변화 방어
			if (in_ != W_.Rows() || out_ != W_.Cols()) SyncMatrixToVec_();
			else {
				for (i32 k = 0; k < in_; ++k) for (i32 j = 0; j < out_; ++j) Wv_[static_cast<size_t>(k) * out_ + j] = W_(k, j);
				for (i32 j = 0; j < out_; ++j) bv_[j] = b_(0, j);
			}
		}
	public:
		Dense(i32 in_dim, i32 out_dim, float init_scale = 0.01f)
			: W_(in_dim, out_dim), b_(1, out_dim), gW_(in_dim, out_dim), gb_(1, out_dim) {
			Initializer::Uniform(W_, init_scale, 123);
			b_.Fill(0.0f); gW_.Fill(0.0f); gb_.Fill(0.0f);
			SyncMatrixToVec_();
		}
		void Forward(const Matrix& X, Matrix& Y) override {
			SyncWeightMatrixToVec_();

			const i32 B = X.Rows();
			Y.Reset(B, out_);
			for (i32 n = 0; n < B; ++n) {
				for (i32 j = 0; j < out_; ++j) {
					float acc = bv_[j];                           // bias
					const size_t base = static_cast<size_t>(j);   // (k,j)에서 j는 고정
					for (i32 k = 0; k < in_; ++k) {
						acc += X(n, k) * Wv_[static_cast<size_t>(k) * out_ + j];
					}
					Y(n, j) = acc;
				}
			}
		}

		void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override {
			const i32 B = X.Rows();

			// 0) 벡터 grad 0으로
			std::fill(gWv_.begin(), gWv_.end(), 0.f);
			std::fill(gbv_.begin(), gbv_.end(), 0.f);

			// 1) gbv[j] = sum_i dY(i,j)
			for (i32 i = 0; i < B; ++i)
				for (i32 j = 0; j < out_; ++j)
					gbv_[j] += dY(i, j);

			// 2) gWv(k,j) = sum_i X(i,k) * dY(i,j)
			for (i32 i = 0; i < B; ++i) {
				for (i32 j = 0; j < out_; ++j) {
					const float gy = dY(i, j);
					for (i32 k = 0; k < in_; ++k) {
						gWv_[static_cast<size_t>(k) * out_ + j] += X(i, k) * gy;
					}
				}
			}

			// 3) dX(i,k) = sum_j dY(i,j) * W(k,j)  (Wv_ 사용)
			dX.Reset(B, in_);
			for (i32 i = 0; i < B; ++i) {
				for (i32 k = 0; k < in_; ++k) {
					float acc = 0.f;
					const size_t wrow = static_cast<size_t>(k) * out_;
					for (i32 j = 0; j < out_; ++j) acc += dY(i, j) * Wv_[wrow + j];
					dX(i, k) = acc;
				}
			}

			// 4) 벡터 grad -> 기존 Matrix grad 로 1회 동기화 (Updater 호환)
			SyncGradVecToMatrix_();
		}

		void ZeroGrad() override {
			gW_.Fill(0.0f); gb_.Fill(0.0f); 
			std::fill(gWv_.begin(), gWv_.end(), 0.f);
			std::fill(gbv_.begin(), gbv_.end(), 0.f);
		}
		// Step는 Trainer에서 StudentUpdater로 처리하므로 no-op
		void Step(float) override {}


		// 접근자 (StudentUpdater용)
		Matrix& WRef() { return W_; }
		Matrix& bRef() { return b_; }
		Matrix& gWRef() { return gW_; }
		Matrix& gbRef() { return gb_; }
		const Matrix& W() const { return W_; }
		const Matrix& b() const { return b_; }
		const Matrix& gW() const { return gW_; }
		const Matrix& gb() const { return gb_; }
	};
}
