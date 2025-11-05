// =============================
// include/vsnn/Timer.hpp
// =============================
#pragma once
#include <chrono>

using namespace std;

namespace vsnn {
    class Timer {
    public:
        using clock = chrono::high_resolution_clock;
    private:
        clock::time_point t0_;
    public:
        void Tic() { t0_ = clock::now(); }
        double TocMs() const {
            auto t1 = clock::now();
            return chrono::duration<double, milli>(t1 - t0_).count();
        }
    };
}


#pragma once
#include <type_traits>
#include "Sequential.hpp"
#include "Dense.hpp"
#include "Timer.hpp"


namespace vsnn {
    class TrainUpdater {
    public:
        // W <- W - lr * gW
        // b <- b - lr * gb
        static void Update(Sequential& model, float lr) {
            for (size_t li = 0; li < model.NumLayers(); ++li) {
                auto* L = model.LayerAt(li);
                auto* D = dynamic_cast<Dense*>(L);
                if (!D) continue;
                Matrix& W = D->WRef(); Matrix& gW = D->gWRef();
                Matrix& b = D->bRef(); Matrix& gb = D->gbRef();
                // W update (iterate only non-zero gradients)
                for (const auto& kv : gW.Sparse()) {
                    const size_t idx = kv.first; const float grad = kv.second;
                    W.SparseAddByIndex(idx, -lr * grad);
                }
                // b update
                for (int j = 0; j < b.Cols(); ++j) {
                    b(0, j) -= lr * gb(0, j);
                }
            }
        }
    };
}