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


	std::vector<std::array<float, 54>> feats; feats.reserve(10000);
	std::vector<int> labels; labels.reserve(10000);

	while (std::getline(fin, line)) {
		if (line.empty()) continue;
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
		if (!std::getline(ss, tok, ',')) continue; // class
		int lab = 0; try { lab = std::stoi(tok); }
		catch (...) { continue; }
		if (lab < 1 || lab > 7) continue; // 1..7
		labels.push_back(lab - 1); // 0..6
		feats.push_back(f);
		++kept;
		if (max_rows > 0 && kept >= max_rows) break;
	}

	const int N = static_cast<int>(feats.size());
	if (N == 0) return false;
	X.Reset(N, 54); y = labels;
	for (int n = 0; n < N; ++n) for (int d = 0; d < 54; ++d) X(n, d) = feats[n][d];

	return true;
}

static void StandardizeCovertype(Matrix& X) {
	// 연속형 10개만 표준화, one-hot 44개는 그대로
	const int N = X.Rows();
	const int D = X.Cols();
	const int cont = 10; // 0..9
	vector<float> mean(cont, 0.f), stdv(cont, 0.f);
	for (int d = 0; d < cont; ++d) {
		for (int n = 0; n < N; ++n) mean[d] += X(n, d);
		mean[d] /= std::max(1, N);
		for (int n = 0; n < N; ++n) { float z = X(n, d) - mean[d]; stdv[d] += z * z; }
		stdv[d] = std::sqrt(stdv[d] / std::max(1, N));
		if (stdv[d] == 0.f) stdv[d] = 1.f;
		for (int n = 0; n < N; ++n) X(n, d) = (X(n, d) - mean[d]) / stdv[d];
	}
	(void)D; // 나머지 44개는 0/1 그대로 유지
}


static void GatherRows(const Matrix& X, const vector<int>& y, const vector<int>& idx, Matrix& Xo, vector<int>& yo) {
	const int N = (int)idx.size(), D = X.Cols();
	Xo.Reset(N, D); yo.resize(N);
	for (int i = 0; i < N; ++i) { int n = idx[i]; for (int d = 0; d < D; ++d) Xo(i, d) = X(n, d); yo[i] = y[n]; }
}


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
	const int stride = 2; // 2로 하면 절반 샘플 사용. 더 줄이려면 4,8...


	if (!LoadCovertypeCSV(path, X, y, max_rows, stride)) {
		cerr << "[ERROR] " << path << " 로드 실패. 경로/포맷을 확인하세요." << endl;
		return 1;
	}
	StandardizeCovertype(X);

	const int N = X.Rows(); vector<int> idx(N); iota(idx.begin(), idx.end(), 0);
	mt19937 rng(0); shuffle(idx.begin(), idx.end(), rng);
	const int Ntrain = (int)(N * 0.9);
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