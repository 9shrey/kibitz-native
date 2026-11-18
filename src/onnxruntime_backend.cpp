// The same network run by ONNX Runtime from the exported .onnx file.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.hpp"
#include "kibitz/kibitz.h"

namespace kibitz {
namespace {

// One ONNX Runtime environment per process, created on first use.
Ort::Env& env() {
  static Ort::Env instance(ORT_LOGGING_LEVEL_WARNING, "kibitz");
  return instance;
}

// The name ONNX Runtime lists a provider under in GetAvailableProviders(), and the name its
// generic AppendExecutionProvider() takes. nullptr for the CPU provider, which is always there.
struct ProviderNames {
  const char* listed;
  const char* append;
  const char* label;
};

ProviderNames names(int provider) {
  switch (provider) {
    case KIBITZ_PROVIDER_CPU:
      return {"CPUExecutionProvider", nullptr, "CPU"};
    case KIBITZ_PROVIDER_QNN:
      return {"QNNExecutionProvider", "QNN", "QNN"};
    case KIBITZ_PROVIDER_XNNPACK:
      return {"XnnpackExecutionProvider", "XNNPACK", "XNNPACK"};
    default:
      throw Error(KIBITZ_ERR_ARGUMENT, "unknown execution provider");
  }
}

bool available(const char* listed) {
  const auto providers = Ort::GetAvailableProviders();
  return std::find(providers.begin(), providers.end(), listed) != providers.end();
}

std::unordered_map<std::string, std::string> provider_options(int provider) {
  if (provider == KIBITZ_PROVIDER_QNN) {
    // The HTP (Hexagon NPU) backend; QNN also has CPU and GPU backends selected the same way.
#ifdef _WIN32
    return {{"backend_path", "QnnHtp.dll"}};
#else
    return {{"backend_path", "libQnnHtp.so"}};
#endif
  }
  if (provider == KIBITZ_PROVIDER_XNNPACK) return {{"intra_op_num_threads", "1"}};
  return {};
}

class OnnxRuntimeBackend final : public Backend {
 public:
  OnnxRuntimeBackend(const std::filesystem::path& path, int provider, bool allow_cpu_fallback) {
    if (!std::filesystem::exists(path)) throw Error(KIBITZ_ERR_IO, "cannot open " + path.string());
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);  // callers parallelise across positions themselves

    const ProviderNames p = names(provider);
    if (p.append != nullptr) {
      if (available(p.listed)) {
        try {
          options.AppendExecutionProvider(p.append, provider_options(provider));
        } catch (const Ort::Exception& e) {
          if (!allow_cpu_fallback)
            throw Error(KIBITZ_ERR_UNAVAILABLE, std::string(p.label) + " provider failed: " + e.what());
        }
        // Without this, operators the provider can't run are quietly given to the CPU.
        if (!allow_cpu_fallback) options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      } else if (!allow_cpu_fallback) {
        throw Error(KIBITZ_ERR_UNAVAILABLE, std::string("this ONNX Runtime build has no ") + p.label +
                                                " execution provider");
      }
    }

    try {
      // path.c_str() is wchar_t* on Windows and char* elsewhere, matching ORTCHAR_T.
      session_ = std::make_unique<Ort::Session>(env(), path.c_str(), options);
    } catch (const Ort::Exception& e) {
      // With fallback off, a model the provider can't fully run fails here, not at Run().
      const bool provider_refused = p.append != nullptr && !allow_cpu_fallback;
      throw Error(provider_refused ? KIBITZ_ERR_RUNTIME : KIBITZ_ERR_FORMAT,
                  std::string("ONNX Runtime could not load the model: ") + e.what());
    }
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
      throw Error(KIBITZ_ERR_FORMAT, "expected a model with one input and one output");
    auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2 || shape[1] != kInputs)
      throw Error(KIBITZ_ERR_FORMAT, "model input must be [batch, 768]");
  }

  void evaluate(std::span<const Features> positions, std::span<float> z) const override {
    // Larger batches amortise ONNX Runtime's per-call overhead; chunking caps the memory used.
    constexpr std::size_t kChunk = 512;
    std::vector<float> input;
    for (std::size_t start = 0; start < positions.size(); start += kChunk) {
      const std::size_t n = std::min(kChunk, positions.size() - start);
      input.assign(n * kInputs, 0.0f);
      for (std::size_t p = 0; p < n; ++p) {
        const Features& f = positions[start + p];
        for (int i = 0; i < f.count; ++i) input[p * kInputs + f.index[static_cast<std::size_t>(i)]] = 1.0f;
      }

      const std::array<int64_t, 2> in_shape{static_cast<int64_t>(n), kInputs};
      const std::array<int64_t, 1> out_shape{static_cast<int64_t>(n)};
      auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value in = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                                      in_shape.data(), in_shape.size());
      // ONNX Runtime writes straight into the caller's buffer: no copy of the results.
      Ort::Value out = Ort::Value::CreateTensor<float>(memory, z.data() + start, n,
                                                       out_shape.data(), out_shape.size());
      const char* in_names[] = {"features"};
      const char* out_names[] = {"z"};
      try {
        session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, &out, 1);
      } catch (const Ort::Exception& e) {
        throw Error(KIBITZ_ERR_RUNTIME, std::string("ONNX Runtime failed: ") + e.what());
      }
    }
  }

 private:
  std::unique_ptr<Ort::Session> session_;  // Session::Run is safe to call from many threads
};

}  // namespace

std::unique_ptr<Backend> make_onnxruntime_backend(const std::filesystem::path& model, int provider,
                                                  bool allow_cpu_fallback) {
  return std::make_unique<OnnxRuntimeBackend>(model, provider, allow_cpu_fallback);
}

bool onnxruntime_has_provider(int provider) { return available(names(provider).listed); }

}  // namespace kibitz
