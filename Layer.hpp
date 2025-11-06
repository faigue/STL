// =============================
// include/vsnn/Layer.hpp
// =============================
#pragma once
#include "Matrix.hpp"


namespace vsnn {
	class Layer {
	public:
		virtual ~Layer() = default;
		virtual void Forward(const Matrix& X, Matrix& Y) = 0;
		virtual void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) = 0;
		virtual void ZeroGrad() {}
		virtual void Step(float /*lr*/) {}
	};
}