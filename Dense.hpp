// =============================
// include/vsnn/Dense.hpp
// =============================
#pragma once
#include "Layer.hpp"
#include "Ops.hpp"
#include "Initializer.hpp"


namespace vsnn {
	class Dense : public Layer {
	private:
		Matrix W_, b_;
		Matrix gW_, gb_;
	public:
		Dense(i32 in_dim, i32 out_dim, float init_scale = 0.01f)
			: W_(in_dim, out_dim), b_(1, out_dim), gW_(in_dim, out_dim), gb_(1, out_dim) {
			Initializer::Uniform(W_, init_scale, 123);
			b_.Fill(0.0f); gW_.Fill(0.0f); gb_.Fill(0.0f);
		}
		void Forward(const Matrix& X, Matrix& Y) override {
			Ops::MatMul(X, W_, Y);
			Ops::AddRowBias(Y, b_);
		}
		void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override {
			// gW = X^T * dY
			if (gW_.Rows() != W_.Rows() || gW_.Cols() != W_.Cols()) gW_.Reset(W_.Rows(), W_.Cols());
			for (i32 k = 0; k < W_.Rows(); ++k) {
				for (i32 j = 0; j < W_.Cols(); ++j) {
					float acc = 0.0f;
					for (i32 i = 0; i < X.Rows(); ++i) acc += X(i, k) * dY(i, j);
					gW_(k, j) = acc;
				}
			}
			// gb = sum_rows(dY)
			if (gb_.Rows() != 1 || gb_.Cols() != W_.Cols()) gb_.Reset(1, W_.Cols());
			for (i32 j = 0; j < W_.Cols(); ++j) {
				float acc = 0.0f; for (i32 i = 0; i < X.Rows(); ++i) acc += dY(i, j); gb_(0, j) = acc;
			}
			// dX = dY * W^T
			if (dX.Rows() != X.Rows() || dX.Cols() != W_.Rows()) dX.Reset(X.Rows(), W_.Rows());
			for (i32 i = 0; i < X.Rows(); ++i) {
				for (i32 k = 0; k < W_.Rows(); ++k) {
					float acc = 0.0f; for (i32 j = 0; j < W_.Cols(); ++j) acc += dY(i, j) * W_(k, j);
					dX(i, k) = acc;
				}
			}
		}
		void ZeroGrad() override { gW_.Fill(0.0f); gb_.Fill(0.0f); }
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
