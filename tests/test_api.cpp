#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "kibitz/kibitz.h"
#include "test_data.hpp"

namespace {

constexpr float kCp = 400.0f;
// PyTorch and C++ add the same floats in a different order; 1e-4 in z is 0.04 centipawns.
constexpr float kTolerance = 1e-4f;

struct Evaluator {
  kibitz_evaluator* ev = nullptr;
  Evaluator(const std::string& path, kibitz_backend backend) {
    const kibitz_status s = kibitz_open(path.c_str(), backend, &ev);
    if (s != KIBITZ_OK) ADD_FAILURE() << "open failed: " << kibitz_last_error();
  }
  ~Evaluator() { kibitz_close(ev); }
  Evaluator(const Evaluator&) = delete;
  Evaluator& operator=(const Evaluator&) = delete;
};

struct BackendCase {
  kibitz_backend backend;
  const char* model;
  const char* name;
};

std::vector<BackendCase> available_backends() {
  std::vector<BackendCase> v{{KIBITZ_BACKEND_NATIVE, "kibitz.weights", "Native"}};
  if (kibitz_has_backend(KIBITZ_BACKEND_ONNXRUNTIME))
    v.push_back({KIBITZ_BACKEND_ONNXRUNTIME, "kibitz.onnx", "OnnxRuntime"});
  return v;
}

std::vector<const char*> fen_pointers(const std::vector<ParityCase>& cases) {
  std::vector<const char*> p;
  for (const auto& c : cases) p.push_back(c.fen.c_str());
  return p;
}

}  // namespace

class EachBackend : public ::testing::TestWithParam<BackendCase> {
 protected:
  Evaluator evaluator{model_path(GetParam().model), GetParam().backend};
};

TEST_P(EachBackend, MatchesPyTorchOn100Positions) {
  const auto cases = parity_cases();
  ASSERT_EQ(cases.size(), 100u);
  float worst = 0.0f;
  for (const auto& c : cases) {
    float cp = 0.0f;
    ASSERT_EQ(kibitz_eval_fen(evaluator.ev, c.fen.c_str(), &cp), KIBITZ_OK) << kibitz_last_error();
    worst = std::max(worst, std::fabs(cp / kCp - c.z));
  }
  EXPECT_LT(worst, kTolerance);
  std::printf("[ info ] worst |z - PyTorch| = %.2e\n", static_cast<double>(worst));
}

TEST_P(EachBackend, BatchGivesTheSameAnswersAsOneAtATime) {
  const auto cases = parity_cases();
  const auto fens = fen_pointers(cases);
  std::vector<float> batch(cases.size());
  ASSERT_EQ(kibitz_eval_batch(evaluator.ev, fens.data(), fens.size(), batch.data()), KIBITZ_OK);
  for (size_t i = 0; i < cases.size(); ++i) {
    float one = 0.0f;
    ASSERT_EQ(kibitz_eval_fen(evaluator.ev, fens[i], &one), KIBITZ_OK);
    EXPECT_NEAR(batch[i], one, 1e-3f) << cases[i].fen;
  }
}

TEST_P(EachBackend, EmptyBatchIsFine) {
  EXPECT_EQ(kibitz_eval_batch(evaluator.ev, nullptr, 0, nullptr), KIBITZ_OK);
}

TEST_P(EachBackend, OneBadFenRejectsTheWholeBatchAndNamesIt) {
  const char* fens[] = {"4k3/8/8/8/8/8/8/4K3 w - - 0 1", "not a fen", "4k3/8/8/8/8/8/8/4K3 b - - 0 1"};
  float out[3] = {};
  EXPECT_EQ(kibitz_eval_batch(evaluator.ev, fens, 3, out), KIBITZ_ERR_FEN);
  EXPECT_NE(std::string(kibitz_last_error()).find("fens[1]"), std::string::npos) << kibitz_last_error();
}

TEST_P(EachBackend, SharedAcrossEightThreadsGivesIdenticalResults) {
  const auto cases = parity_cases();
  const auto fens = fen_pointers(cases);
  std::vector<float> expected(cases.size());
  ASSERT_EQ(kibitz_eval_batch(evaluator.ev, fens.data(), fens.size(), expected.data()), KIBITZ_OK);

  std::vector<std::vector<float>> results(8, std::vector<float>(cases.size()));
  std::vector<int> failures(8, 0);
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      for (int round = 0; round < 25; ++round) {
        for (size_t i = 0; i < fens.size(); ++i) {
          if (kibitz_eval_fen(evaluator.ev, fens[i], &results[t][i]) != KIBITZ_OK) ++failures[t];
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  for (int t = 0; t < 8; ++t) {
    EXPECT_EQ(failures[t], 0);
    for (size_t i = 0; i < cases.size(); ++i) EXPECT_NEAR(results[t][i], expected[i], 1e-3f);
  }
}

INSTANTIATE_TEST_SUITE_P(Api, EachBackend, ::testing::ValuesIn(available_backends()),
                         [](const auto& test) { return std::string(test.param.name); });

TEST(Api, BothBackendsAgree) {
  if (!kibitz_has_backend(KIBITZ_BACKEND_ONNXRUNTIME)) GTEST_SKIP() << "built without ONNX Runtime";
  Evaluator native(model_path("kibitz.weights"), KIBITZ_BACKEND_NATIVE);
  Evaluator ort(model_path("kibitz.onnx"), KIBITZ_BACKEND_ONNXRUNTIME);
  for (const auto& c : parity_cases()) {
    float a = 0.0f, b = 0.0f;
    ASSERT_EQ(kibitz_eval_fen(native.ev, c.fen.c_str(), &a), KIBITZ_OK);
    ASSERT_EQ(kibitz_eval_fen(ort.ev, c.fen.c_str(), &b), KIBITZ_OK);
    EXPECT_NEAR(a, b, kTolerance * kCp) << c.fen;
  }
}

TEST(Api, RejectsNullArguments) {
  kibitz_evaluator* ev = nullptr;
  EXPECT_EQ(kibitz_open(nullptr, KIBITZ_BACKEND_NATIVE, &ev), KIBITZ_ERR_ARGUMENT);
  EXPECT_EQ(kibitz_open(model_path("kibitz.weights").c_str(), KIBITZ_BACKEND_NATIVE, nullptr),
            KIBITZ_ERR_ARGUMENT);
  float cp = 0.0f;
  EXPECT_EQ(kibitz_eval_fen(nullptr, "4k3/8/8/8/8/8/8/4K3 w", &cp), KIBITZ_ERR_ARGUMENT);
  kibitz_close(nullptr);  // allowed
}

TEST(Api, ReportsAMissingFile) {
  kibitz_evaluator* ev = nullptr;
  EXPECT_EQ(kibitz_open("/no/such/file.weights", KIBITZ_BACKEND_NATIVE, &ev), KIBITZ_ERR_IO);
  EXPECT_EQ(ev, nullptr);
  EXPECT_STRNE(kibitz_last_error(), "");
}

TEST(Api, RejectsCorruptWeightsFiles) {
  const auto dir = std::filesystem::temp_directory_path();
  std::ifstream src(model_path("kibitz.weights"), std::ios::binary);
  std::vector<char> good((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());

  auto write = [&](const char* name, std::vector<char> bytes) {
    const auto p = (dir / name).string();
    std::ofstream(p, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return p;
  };
  auto truncated = good;
  truncated.resize(good.size() - 4);
  auto wrong_magic = good;
  wrong_magic[0] = 'X';
  auto huge_hidden = good;
  huge_hidden[8] = huge_hidden[9] = huge_hidden[10] = huge_hidden[11] = '\x7f';

  for (const auto& p : {write("kb_truncated.weights", truncated), write("kb_magic.weights", wrong_magic),
                        write("kb_hidden.weights", huge_hidden), write("kb_empty.weights", {})}) {
    kibitz_evaluator* ev = nullptr;
    EXPECT_EQ(kibitz_open(p.c_str(), KIBITZ_BACKEND_NATIVE, &ev), KIBITZ_ERR_FORMAT) << p;
    EXPECT_EQ(ev, nullptr);
    std::filesystem::remove(p);
  }
}

TEST(Api, OnnxBackendRejectsANonModelFile) {
  if (!kibitz_has_backend(KIBITZ_BACKEND_ONNXRUNTIME)) GTEST_SKIP() << "built without ONNX Runtime";
  kibitz_evaluator* ev = nullptr;
  // A weights file is not an ONNX model.
  EXPECT_EQ(kibitz_open(model_path("kibitz.weights").c_str(), KIBITZ_BACKEND_ONNXRUNTIME, &ev),
            KIBITZ_ERR_FORMAT);
  EXPECT_EQ(ev, nullptr);
}

TEST(Api, ErrorMessagesArePerThread) {
  kibitz_evaluator* ev = nullptr;
  ASSERT_NE(kibitz_open("/missing", KIBITZ_BACKEND_NATIVE, &ev), KIBITZ_OK);
  std::string other;
  std::thread([&] { other = kibitz_last_error(); }).join();
  EXPECT_EQ(other, "");                 // the other thread saw no error
  EXPECT_STRNE(kibitz_last_error(), "");  // this one still has its own
}

TEST(Api, EvaluationIsFromTheSideToMovesPointOfView) {
  Evaluator ev(model_path("kibitz.weights"), KIBITZ_BACKEND_NATIVE);
  float white_up_queen = 0.0f, black_to_move = 0.0f;
  ASSERT_EQ(kibitz_eval_fen(ev.ev, "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", &white_up_queen), KIBITZ_OK);
  ASSERT_EQ(kibitz_eval_fen(ev.ev, "4k3/8/8/8/8/8/8/3QK3 b - - 0 1", &black_to_move), KIBITZ_OK);
  EXPECT_GT(white_up_queen, 300.0f);
  EXPECT_LT(black_to_move, -300.0f);
}

// --- execution providers and the INT8 model ---------------------------------------------------

namespace {

// The INT8 model's measured error is 18 cp at worst on these positions (tools/quantize.py).
constexpr float kInt8ToleranceCp = 25.0f;

float worst_cp_error(kibitz_evaluator* ev) {
  float worst = 0.0f;
  for (const auto& c : parity_cases()) {
    float cp = 0.0f;
    EXPECT_EQ(kibitz_eval_fen(ev, c.fen.c_str(), &cp), KIBITZ_OK) << kibitz_last_error();
    worst = std::max(worst, std::fabs(cp - c.z * kCp));
  }
  return worst;
}

struct OpenedEx {
  kibitz_evaluator* ev = nullptr;
  kibitz_status status;
  OpenedEx(const std::string& path, kibitz_provider provider, int fallback) {
    kibitz_options o;
    kibitz_options_init(&o);
    o.backend = KIBITZ_BACKEND_ONNXRUNTIME;
    o.provider = provider;
    o.allow_cpu_fallback = fallback;
    status = kibitz_open_ex(path.c_str(), &o, &ev);
  }
  ~OpenedEx() { kibitz_close(ev); }
  OpenedEx(const OpenedEx&) = delete;
  OpenedEx& operator=(const OpenedEx&) = delete;
};

const kibitz_provider kAccelerators[] = {KIBITZ_PROVIDER_QNN, KIBITZ_PROVIDER_XNNPACK};

}  // namespace

TEST(Options, InitSetsTheDefaults) {
  kibitz_options o;
  std::memset(&o, 0xAB, sizeof o);
  kibitz_options_init(&o);
  EXPECT_EQ(o.struct_size, sizeof(kibitz_options));
  EXPECT_EQ(o.backend, KIBITZ_BACKEND_NATIVE);
  EXPECT_EQ(o.provider, KIBITZ_PROVIDER_CPU);
  EXPECT_EQ(o.allow_cpu_fallback, 1);
  kibitz_options_init(nullptr);  // allowed
}

TEST(Options, OpenExRejectsUninitialisedOptions) {
  kibitz_options o{};  // struct_size 0
  kibitz_evaluator* ev = reinterpret_cast<kibitz_evaluator*>(&o);  // must be reset to null
  EXPECT_EQ(kibitz_open_ex(model_path("kibitz.weights").c_str(), &o, &ev), KIBITZ_ERR_ARGUMENT);
  EXPECT_EQ(ev, nullptr);
  EXPECT_EQ(kibitz_open_ex(model_path("kibitz.weights").c_str(), nullptr, &ev), KIBITZ_ERR_ARGUMENT);
}

TEST(Options, NativeBackendThroughOpenEx) {
  kibitz_options o;
  kibitz_options_init(&o);
  kibitz_evaluator* ev = nullptr;
  ASSERT_EQ(kibitz_open_ex(model_path("kibitz.weights").c_str(), &o, &ev), KIBITZ_OK);
  EXPECT_LT(worst_cp_error(ev), kTolerance * kCp);
  kibitz_close(ev);
}

TEST(Providers, CpuIsAlwaysThereWhenOnnxRuntimeIs) {
  EXPECT_EQ(kibitz_has_provider(KIBITZ_PROVIDER_CPU), kibitz_has_backend(KIBITZ_BACKEND_ONNXRUNTIME));
  EXPECT_EQ(kibitz_has_provider(static_cast<kibitz_provider>(3)), 0);
}

TEST(Providers, ReportsWhatThisBuildHas) {
  for (const auto p : {KIBITZ_PROVIDER_CPU, KIBITZ_PROVIDER_QNN, KIBITZ_PROVIDER_XNNPACK})
    std::printf("[ info ] provider %d available: %d\n", static_cast<int>(p), kibitz_has_provider(p));
}

TEST(Providers, MissingProviderWithoutFallbackIsUnavailable) {
  if (!kibitz_has_backend(KIBITZ_BACKEND_ONNXRUNTIME)) GTEST_SKIP() << "built without ONNX Runtime";
  int checked = 0;
  for (const auto p : kAccelerators) {
    if (kibitz_has_provider(p)) continue;
    OpenedEx opened(model_path("kibitz.onnx"), p, /*fallback=*/0);
    EXPECT_EQ(opened.status, KIBITZ_ERR_UNAVAILABLE) << static_cast<int>(p);
    EXPECT_EQ(opened.ev, nullptr);
    EXPECT_STRNE(kibitz_last_error(), "");
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "every provider is available in this build";
}

TEST(Providers, MissingProviderWithFallbackRunsOnCpu) {
  if (!kibitz_has_backend(KIBITZ_BACKEND_ONNXRUNTIME)) GTEST_SKIP() << "built without ONNX Runtime";
  for (const auto p : kAccelerators) {
    OpenedEx opened(model_path("kibitz.onnx"), p, /*fallback=*/1);
    ASSERT_EQ(opened.status, KIBITZ_OK) << kibitz_last_error();
    EXPECT_LT(worst_cp_error(opened.ev), kTolerance * kCp) << static_cast<int>(p);
  }
}

// Runs only where the provider exists (for QNN: a Snapdragon device with the QNN build of ONNX
// Runtime). Fallback is off, so the whole INT8 graph must run on the provider.
TEST(Providers, Int8ModelRunsEntirelyOnEachAvailableAccelerator) {
  int ran = 0;
  for (const auto p : kAccelerators) {
    if (!kibitz_has_provider(p)) continue;
    OpenedEx opened(model_path("kibitz.int8.onnx"), p, /*fallback=*/0);
    if (opened.status != KIBITZ_OK) {
      ADD_FAILURE() << "provider " << static_cast<int>(p) << ": " << kibitz_last_error();
      continue;
    }
    EXPECT_LT(worst_cp_error(opened.ev), kInt8ToleranceCp);
    ++ran;
  }
  if (ran == 0) GTEST_SKIP() << "no accelerator provider in this build";
}

TEST(Int8, MatchesPyTorchWithin25Centipawns) {
  if (!kibitz_has_backend(KIBITZ_BACKEND_ONNXRUNTIME)) GTEST_SKIP() << "built without ONNX Runtime";
  Evaluator ev(model_path("kibitz.int8.onnx"), KIBITZ_BACKEND_ONNXRUNTIME);
  const float worst = worst_cp_error(ev.ev);
  EXPECT_LT(worst, kInt8ToleranceCp);
  std::printf("[ info ] INT8 worst |cp - PyTorch| = %.1f\n", static_cast<double>(worst));
}

TEST(Int8, IsAboutAQuarterOfTheFloatModel) {
  const auto fp32 = std::filesystem::file_size(model_path("kibitz.onnx"));
  const auto int8 = std::filesystem::file_size(model_path("kibitz.int8.onnx"));
  EXPECT_LT(int8 * 3, fp32);
}
