// =============================
// include/vsnn/Matrix.hpp
// =============================
#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace std;

namespace vsnn {
	using f32 = float;
	using i32 = int32_t;


	class Matrix {
	private:
		i32 rows_ = 0, cols_ = 0;
		vector<f32> data_;
	public:
		Matrix() = default;
		Matrix(i32 r, i32 c) { Reset(r, c); }
		void Reset(i32 r, i32 c) {
			rows_ = r; cols_ = c; data_.assign(static_cast<size_t>(r) * c, 0.0f);
		}
		inline i32 Rows() const { return rows_; }
		inline i32 Cols() const { return cols_; }
		inline f32* Data() { return data_.data(); }
		inline const f32* Data() const { return data_.data(); }
		inline f32& operator()(i32 r, i32 c) { return data_[static_cast<size_t>(r) * cols_ + c]; }
		inline f32 operator()(i32 r, i32 c) const { return data_[static_cast<size_t>(r) * cols_ + c]; }
		inline void Fill(f32 v) { fill(data_.begin(), data_.end(), v); }
		inline const vector<f32>& Raw() const { return data_; }
		inline vector<f32>& Raw() { return data_; }
	};
}