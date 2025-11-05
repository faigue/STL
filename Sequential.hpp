// =============================
// include/vsnn/Sequential.hpp
// =============================
#pragma once
#include <memory>
#include <vector>
#include "Layer.hpp"

using namespace std;

namespace vsnn {
	class Sequential {
	private:
		vector<unique_ptr<Layer>> layers_;
		vector<Matrix> acts_; // X0..X_L
	public:
		template<typename T, typename... Args>
		T* Add(Args&&... args) {
			layers_.push_back(make_unique<T>(forward<Args>(args)...));
			return static_cast<T*>(layers_.back().get());
		}
		void Forward(const Matrix& X, Matrix& out) {
			acts_.resize(layers_.size() + 1);
			acts_[0] = X;
			Matrix cur = X, nxt;
			for (size_t i = 0; i < layers_.size(); ++i) {
				layers_[i]->Forward(cur, nxt);
				acts_[i + 1] = nxt;
				cur = acts_[i + 1];
			}
			out = acts_.back();
		}
		void Backward(const Matrix& dOut) {
			Matrix cur_d = dOut, prev_d;
			for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
				layers_[i]->Backward(acts_[i], cur_d, prev_d);
				cur_d = prev_d;
			}
		}
		void ZeroGrad() { for (auto& L : layers_) L->ZeroGrad(); }
		void Step(float lr) { for (auto& L : layers_) L->Step(lr); }
		// ---- Introspection for Updater ----
		size_t NumLayers() const { return layers_.size(); }
		Layer* LayerAt(size_t i) { return layers_[i].get(); }
		const Layer* LayerAt(size_t i) const { return layers_[i].get(); }
	};
}