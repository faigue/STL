// =============================
// include/vsnn/Ops.hpp
// =============================
#pragma once
#include <cmath>
#include <algorithm>
#include "Matrix.hpp"

using namespace std;

namespace vsnn {
	class Ops {
	public:
		// Y = X * W with shapes: (N,in) * (in,out) = (N,out)
		static void MatMul(const Matrix& X, const Matrix& W, Matrix& Y) {
			assert(X.Cols() == W.Rows());
			if (Y.Rows() != X.Rows() || Y.Cols() != W.Cols()) Y.Reset(X.Rows(), W.Cols());
			for (i32 n = 0; n < X.Rows(); ++n) {
				for (i32 j = 0; j < W.Cols(); ++j) {
					float acc = 0.0f;
					for (i32 k = 0; k < X.Cols(); ++k) acc += X(n, k) * W(k, j);
					Y(n, j) = acc;
				}
			}
		}
		static void AddRowBias(Matrix& Y, const Matrix& b) {
			assert(b.Rows() == 1 && b.Cols() == Y.Cols());
			for (i32 n = 0; n < Y.Rows(); ++n)
				for (i32 j = 0; j < Y.Cols(); ++j) Y(n, j) += b(0, j);
		}
		static void ReLUForward(const Matrix& X, Matrix& Y) {
			if (Y.Rows() != X.Rows() || Y.Cols() != X.Cols()) Y.Reset(X.Rows(), X.Cols());
			for (i32 r = 0; r < X.Rows(); ++r)
				for (i32 c = 0; c < X.Cols(); ++c)
					Y(r, c) = (X(r, c) > 0.0f) ? X(r, c) : 0.0f;
		}
		static void ReLUBackward(const Matrix& X, const Matrix& dY, Matrix& dX) {
			if (dX.Rows() != X.Rows() || dX.Cols() != X.Cols()) dX.Reset(X.Rows(), X.Cols());
			for (i32 r = 0; r < X.Rows(); ++r)
				for (i32 c = 0; c < X.Cols(); ++c)
					dX(r, c) = (X(r, c) > 0.0f) ? dY(r, c) : 0.0f;
		}
		static void SoftmaxRow(const float* logits, float* probs, int C) {
			float m = logits[0];
			for (int i = 1; i < C; ++i) m = max(m, logits[i]);
			float s = 0.0f; for (int i = 0; i < C; ++i) { probs[i] = exp(logits[i] - m); s += probs[i]; }
			if (s == 0.0f) s = 1e-12f;
			for (int i = 0; i < C; ++i) probs[i] /= s;
		}
	};
}
