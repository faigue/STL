// =============================
// include/vsnn/Matrix.hpp
// =============================
#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include < valarray >

using namespace std;

namespace vsnn {
    using f32 = float;
    using i32 = int32_t;


    class Matrix {
    private:

        int rows_;
        int cols_;
        std::valarray<f32> data_;

    public:
        Matrix() : rows_(0), cols_(0), data_() {}

        Matrix(int r, int c)
            : rows_(r), cols_(c), data_(f32(0), static_cast<size_t>(r)* c) {
        }

        void Reset(i32 r, i32 c) {
            rows_ = r; cols_ = c;
            data_.resize(static_cast<size_t>(r) * c, f32(0));
        }

        inline f32& operator()(int r, int c) {
            assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
            return data_[r * cols_ + c];
        }

        inline f32 operator() (int r, int c) const {
            assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
            return data_[r * cols_ + c];
        }

        inline int Rows() const { return rows_; }
        inline int Cols() const { return cols_; }

        inline std::valarray<f32>& Raw() { return data_; }
        inline const std::valarray<f32>& Raw() const { return data_; }

        inline std::valarray<f32>& Data() { return data_; }
        inline const std::valarray<f32>& Data() const { return data_; }

        inline void Fill(f32 value) { data_ = value; }
    };
}