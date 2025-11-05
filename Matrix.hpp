// =============================
// include/vsnn/Matrix.hpp
// =============================
#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

using namespace std;

namespace vsnn {
    using f32 = float;
    using i32 = int32_t;


    class Matrix {
    private:
        i32 rows_ = 0, cols_ = 0;
        vector<f32> data_;
        // Sparse storage (row-major linear index -> value)
        unordered_map<size_t, f32> sparse_;
        bool use_sparse_ = false;
        inline size_t LinIdx(i32 r, i32 c) const { return static_cast<size_t>(r) * static_cast<size_t>(cols_) + static_cast<size_t>(c); }
    public:
        Matrix() = default;
        Matrix(i32 r, i32 c) { Reset(r, c); }
        void Reset(i32 r, i32 c) {
            rows_ = r; cols_ = c; data_.assign(static_cast<size_t>(r) * c, 0.0f);
            sparse_.clear(); use_sparse_ = false;
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
        // ---- Sparse helpers ----
        inline void EnableSparse(bool on = true) { use_sparse_ = on; }
        inline bool IsSparse() const { return use_sparse_; }
        inline void SparseClear() { sparse_.clear(); }
        inline unordered_map<size_t, f32>& Sparse() { return sparse_; }
        inline const unordered_map<size_t, f32>& Sparse() const { return sparse_; }
        inline void SparseSet(i32 r, i32 c, f32 v) {
            const size_t idx = LinIdx(r, c);
            if (v == 0.0f) { auto it = sparse_.find(idx); if (it != sparse_.end()) sparse_.erase(it); }
            else { sparse_[idx] = v; }
        }
        inline void SparseSetByIndex(size_t idx, f32 v) {
            if (v == 0.0f) { auto it = sparse_.find(idx); if (it != sparse_.end()) sparse_.erase(it); }
            else { sparse_[idx] = v; }
        }
        inline void SparseAddByIndex(size_t idx, f32 dv) {
            if (dv == 0.0f) return;
            auto it = sparse_.find(idx);
            if (it == sparse_.end()) {
                const f32 nv = dv;
                if (nv != 0.0f) sparse_.emplace(idx, nv);
            }
            else {
                it->second += dv;
                if (it->second == 0.0f) sparse_.erase(it);
            }
        }
        inline size_t LinearIndex(i32 r, i32 c) const { return LinIdx(r, c); }
    };
}