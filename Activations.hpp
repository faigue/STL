// =============================
// include/vsnn/Activations.hpp
// =============================
#pragma once
#include "Layer.hpp"
#include "Ops.hpp"


namespace vsnn {
	class ReLU : public Layer {
	public:
		void Forward(const Matrix& X, Matrix& Y) override { Ops::ReLUForward(X, Y); }
		void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override { Ops::ReLUBackward(X, dY, dX); }
	};
}