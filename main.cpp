// =============================
// src/main.cpp
// =============================
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <array>
#include <numeric>

#include "Matrix.hpp"
#include "Dense.hpp"
#include "Activations.hpp"
#include "Loss.hpp"
#include "Sequential.hpp"
#include "Perceptron.hpp"
#include "Timer.hpp"
#include "Trainer.hpp"
#include "Ops.hpp"


using namespace vsnn;
using namespace std;

static bool LoadCovertypeCSV(const std::string& path, Matrix& X, vector<int>& y,
    int max_rows = -1, int stride = 1) {
    std::ifstream fin(path);
    if (!fin.is_open()) return false;
    std::string line; int row = 0; int kept = 0;


    std::vector<std::array<float, 54>> feats;//특정 데이터를 한 행에 54개의 float로 저장(54개의 열), 행은 가변적
    feats.reserve(10000);//벡터의 용량을  10,000으로 설정
    std::vector<int> labels;//정답 레이블 저장하는 벡터, 정답 클래스는 0~6사이의 정수=나무의 종 개수
    labels.reserve(10000);

    while (std::getline(fin, line)) {//하나의 유효한 데이터 샘플 추출, fin=파일 입력 스트림으로 읽어 line 변수에 저장
        if (line.empty()) continue;//빈 공간이면 아래 조건문 실행 X
        if ((row++ % stride) != 0) continue; // 서브샘플링
        std::stringstream ss(line);
        std::string tok; std::array<float, 54> f{};
        int col = 0; bool ok = true; float v = 0.f;
        for (; col < 54; ++col) {
            if (!std::getline(ss, tok, ',')) { ok = false; break; }
            try { v = std::stof(tok); }
            catch (...) { ok = false; break; }
            f[col] = v;
        }
        if (!ok) continue;
        if (!std::getline(ss, tok, ',')) continue; // class(55번째 값: 정답 레이블)
        int lab = 0; try { lab = std::stoi(tok); }
        catch (...) { continue; }
        if (lab < 1 || lab > 7) continue; // 1..7
        labels.push_back(lab - 1); // 0..6
        feats.push_back(f);
        ++kept;
        if (max_rows > 0 && kept >= max_rows) break;
    }
    //임시로 읽어 모은 데이터를 최종적인 행렬 X와 벡터 y 구조로 확정하고 복사하는 역할
    const int N = static_cast<int>(feats.size());
    if (N == 0) return false;
    X.Reset(N, 54); y = labels;
    for (int n = 0; n < N; ++n) for (int d = 0; d < 54; ++d) X(n, d) = feats[n][d];

    return true;
}
// 입력 행렬 X의 특정 열에 대해 표준화를 적용하여 모든 특징의 스케일을 통일한다.
// 해당 함수는 covtype 데이터셋의 54개 특징 중 연속형 특징 10개만을 골라 평균을 0으로, 표준편차는 1로 조정한다.
static void StandardizeCovertype(Matrix& X) {
    const int N = X.Rows();
    const int cont = 10; // 표준화할 특징 수 (0..9)

    // N이 0이면 계산 중단
    if (N == 0) return;

    vector<float> mean(cont, 0.f), stdv(cont, 0.f);

    // =========================================================
    // 1단계: 평균 (mean) 계산 (1차 순회: 데이터 접근 1회)
    // =========================================================
    for (int d = 0; d < cont; ++d) {
        // 합계 계산
        for (int n = 0; n < N; ++n) {
            mean[d] += X(n, d);
        }
        // 평균 확정
        mean[d] /= static_cast<float>(N);
    }

    // =========================================================
    // 2단계: 표준편차 (stdv) 계산 및 X 업데이트 (2차 순회: 데이터 접근 1회)
    // 표준편차 계산과 표준화 적용을 단일 루프에서 동시 처리합니다.
    // =========================================================
    for (int d = 0; d < cont; ++d) {
        float sq_diff_sum = 0.0f; // (값 - 평균)^2의 합계 (분자)

        // 2-1. 분산 계산을 위한 제곱 차이 합계 누적 (데이터 순회)
        for (int n = 0; n < N; ++n) {
            float z = X(n, d) - mean[d];
            sq_diff_sum += z * z;
        }

        // 2-2. 표준편차 확정 (루프 밖에서 처리)
        stdv[d] = std::sqrt(sq_diff_sum / static_cast<float>(N));

        // 0 나누기 방지
        if (stdv[d] == 0.f) stdv[d] = 1.f;

        // 2-3. 표준화 적용 (N 루프를 다시 실행)
        // [주석]: 표준편차가 확정된 후 X 행렬을 업데이트하는 루프입니다.
        //         2-1 루프와 2-3 루프를 합칠 수 있는 완벽한 최적화는 
        //         여기서 불가능하므로, 2차 순회 내에서 두 개의 N 루프를 사용합니다.
        for (int n = 0; n < N; ++n) {
            X(n, d) = (X(n, d) - mean[d]) / stdv[d];
        }
    }
}

//데이터셋에서 특정 행(샘플)들을 골라내어 새로운 행렬과 벡터를 만드는 역할, 테스트셋과 훈련 셋으로 분할
static void GatherRows(const Matrix& X, const vector<int>& y, const vector<int>& idx, Matrix& Xo, vector<int>& yo) {
    const int N = (int)idx.size(), D = X.Cols();
    Xo.Reset(N, D); yo.resize(N);
    for (int i = 0; i < N; ++i) { int n = idx[i]; for (int d = 0; d < D; ++d) Xo(i, d) = X(n, d); yo[i] = y[n]; }
}

//최종 출력인 logits
static int ArgMaxRow0(const Matrix& logits) {
    int C = logits.Cols(); int bi = 0; float bv = logits(0, 0);
    for (int j = 1; j < C; ++j) { if (logits(0, j) > bv) { bv = logits(0, j); bi = j; } }
    return bi;
}

int main() {
    // ---------------------------------------------------------
    // 0) 데이터 준비
    // ---------------------------------------------------------
    Matrix X; vector<int> y;
    const string path = "covtype.data"; // UCI 원본 파일명
    const int max_rows = 120000; // 전체(581k) 중 상한. 전체 쓰려면 -1로.
    const int stride = 2;//전체 데이터 58만개 중 12만개를 읽고, 읽을 때 1칸씩 띄어서 읽음

    if (!LoadCovertypeCSV(path, X, y, max_rows, stride)) {//데이터 로드 및 데이터 제한
        cerr << "[ERROR] " << path << " 로드 실패. 경로/포맷을 확인하세요." << endl;
        return 1;
    }
    StandardizeCovertype(X);//표준화

    const int N = X.Rows(); vector<int> idx(N); iota(idx.begin(), idx.end(), 0);
    mt19937 rng(0); shuffle(idx.begin(), idx.end(), rng);//무작위 순서 채움
    const int Ntrain = (int)(N * 0.9);//90%만 사용
    vector<int> idx_tr(idx.begin(), idx.begin() + Ntrain), idx_te(idx.begin() + Ntrain, idx.end());
    Matrix Xtr, Xte; vector<int> ytr, yte; GatherRows(X, y, idx_tr, Xtr, ytr); GatherRows(X, y, idx_te, Xte, yte);

    cout << "[Dataset] rows=" << N << " (train=" << Xtr.Rows() << ", test=" << Xte.Rows() << ") D=54 C=7" << endl;


    // ---------------------------------------------------------
    // 1) 모델 구성
    // ---------------------------------------------------------
    Sequential model; model.Add<Dense>(54, 256); model.Add<ReLU>(); model.Add<Dense>(256, 7); // 이 부분은 절대 건들지 마세요!!


    // ---------------------------------------------------------
    // 2) (요청사항) 트레이닝 전에 피드포워드만 돌려서 출력 확인
    // ---------------------------------------------------------
    cout << "[Inference-only before training]" << endl;
    cout << "five examples" << endl;
    for (int n = 0; n < min(5, (int)yte.size()); ++n) {
        Matrix X1(1, Xte.Cols()); for (int d = 0; d < Xte.Cols(); ++d) X1(0, d) = Xte(n, d);
        Matrix logits; model.Forward(X1, logits);
        int pred = ArgMaxRow0(logits);//한 행의 최종 클래스의 인덱스
        SoftmaxCrossEntropy CE1; vector<int> y1(1, yte[n]);
        float loss1 = CE1.Forward(logits, y1);
        cout << "\n" << endl;
        cout << fixed << setprecision(4)
            << " sample index:" << n
            << " pred=" << pred
            << " answer=" << yte[n]
            << " loss=" << loss1 << "";
    }

    // ---------------------------------------------------------
    // 3) 학습 실행 (업데이트 시간 별도 측정: Trainer -> StudentUpdater::Update)
    // ---------------------------------------------------------
    Timer TotalTimer;
    TotalTimer.Tic();

    TrainConfig cfg; cfg.epochs = 1; cfg.batch = 1024; cfg.lr = 5e-2f; cfg.warmup = 1; cfg.repeats = 3; // 이 부분은 절대 건들지 마세요!!
    auto report = Trainer::Train<TrainUpdater>(model, X, y, cfg);
    double total_ms = TotalTimer.TocMs();

    cout << "\n" << endl;
    cout << "[Training report]" << endl;
    cout << " total training time (ms): " << total_ms << "\n";
    cout << " final loss : " << report.last_loss << "";

    // ---------------------------------------------------------
    // 4) 트레이닝 후 피드포워드 결과 재확인
    // ---------------------------------------------------------
    cout << "\n" << endl;
    cout << "[Inference - only after training]" << endl;
    cout << "five examples" << endl;
    for (int n = 0; n < min(5, (int)yte.size()); ++n) {
        Matrix X1(1, Xte.Cols()); for (int d = 0; d < Xte.Cols(); ++d) X1(0, d) = Xte(n, d);
        Matrix logits; model.Forward(X1, logits);
        int pred = ArgMaxRow0(logits);
        SoftmaxCrossEntropy CE1; vector<int> y1(1, yte[n]);
        float loss1 = CE1.Forward(logits, y1);
        cout << "\n" << endl;
        cout << fixed << setprecision(4)
            << " sample index:" << n
            << " pred=" << pred
            << " answer=" << yte[n]
            << " loss=" << loss1 << "";
    }

    // ---------------------------------------------------------
    // 5) 테스팅 후 정확도 확인
    // ---------------------------------------------------------
    int correct = 0; Matrix logits;
    for (int n = 0; n < Xte.Rows(); ++n) {
        Matrix X1(1, Xte.Cols()); for (int d = 0; d < Xte.Cols(); ++d) X1(0, d) = Xte(n, d);
        model.Forward(X1, logits);
        if (ArgMaxRow0(logits) == yte[n]) ++correct;
    }
    double acc = (double)correct / max(1, Xte.Rows());
    cout << "\n" << endl;
    cout << "[Test accuracy] " << fixed << setprecision(4) << acc << endl;

    return 0;

}