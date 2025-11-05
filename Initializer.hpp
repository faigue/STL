// =============================
// include/vsnn/Initializer.hpp
// =============================
#pragma once
#include <random>
#include "Matrix.hpp"

using namespace std;

namespace vsnn {
	class Initializer {
	public:
		static void Uniform(Matrix& W, float scale = 0.01f, uint64_t seed = 42) {
			mt19937_64 gen(seed);
			uniform_real_distribution<float> dist(-scale, scale);
			for (auto& x : W.Raw()) x = dist(gen);
		}
		static void Normal(Matrix& W, float stddev = 0.01f, uint64_t seed = 42) {
			mt19937_64 gen(seed);
			normal_distribution<float> dist(0.0f, stddev);
			for (auto& x : W.Raw()) x = dist(gen);
		}
	};
}