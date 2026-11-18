// The C API. Its only job is to translate: C strings and status codes on the outside, C++
// objects and exceptions on the inside. No exception may escape a function marked extern "C".

#include "kibitz/kibitz.h"

#include <cstddef>
#include <filesystem>
#include <new>
#include <string>
#include <vector>

#include "backend.hpp"
#include "features.hpp"

#ifndef KIBITZ_VERSION
#define KIBITZ_VERSION "0.0.0"
#endif

struct kibitz_evaluator {
  std::unique_ptr<kibitz::Backend> backend;
};

namespace {

constexpr float kCpScale = 400.0f;
thread_local std::string g_last_error;

kibitz_status fail(kibitz_status status, std::string message) {
  g_last_error = std::move(message);
  return status;
}

template <typename F>
kibitz_status guarded(F&& body) {
  g_last_error.clear();
  try {
    return body();
  } catch (const kibitz::Error& e) {
    return fail(static_cast<kibitz_status>(e.status), e.what());
  } catch (const std::bad_alloc&) {
    return fail(KIBITZ_ERR_RUNTIME, "out of memory");
  } catch (const std::exception& e) {
    return fail(KIBITZ_ERR_RUNTIME, e.what());
  } catch (...) {
    return fail(KIBITZ_ERR_RUNTIME, "unknown error");
  }
}

}  // namespace

extern "C" {

const char* kibitz_version(void) { return KIBITZ_VERSION; }

int kibitz_has_backend(kibitz_backend backend) {
  switch (backend) {
    case KIBITZ_BACKEND_NATIVE:
      return 1;
    case KIBITZ_BACKEND_ONNXRUNTIME:
#ifdef KIBITZ_WITH_ONNXRUNTIME
      return 1;
#else
      return 0;
#endif
  }
  return 0;
}

int kibitz_has_provider(kibitz_provider provider) {
#ifdef KIBITZ_WITH_ONNXRUNTIME
  try {
    return kibitz::onnxruntime_has_provider(provider) ? 1 : 0;
  } catch (...) {
    return 0;  // unknown provider value
  }
#else
  (void)provider;
  return 0;
#endif
}

void kibitz_options_init(kibitz_options* options) {
  if (options == nullptr) return;
  *options = kibitz_options{};
  options->struct_size = sizeof(kibitz_options);
  options->backend = KIBITZ_BACKEND_NATIVE;
  options->provider = KIBITZ_PROVIDER_CPU;
  options->allow_cpu_fallback = 1;
}

kibitz_status kibitz_open(const char* model_path, kibitz_backend backend, kibitz_evaluator** out) {
  kibitz_options options;
  kibitz_options_init(&options);
  options.backend = backend;
  return kibitz_open_ex(model_path, &options, out);
}

kibitz_status kibitz_open_ex(const char* model_path, const kibitz_options* options,
                             kibitz_evaluator** out) {
  return guarded([&] {
    if (out != nullptr) *out = nullptr;
    if (out == nullptr || model_path == nullptr || options == nullptr)
      return fail(KIBITZ_ERR_ARGUMENT, "null argument");
    // Any size that covers the fields this version reads is fine, so callers built against
    // an older or newer header keep working. Too small means kibitz_options_init wasn't called.
    if (options->struct_size < offsetof(kibitz_options, allow_cpu_fallback) + sizeof(int))
      return fail(KIBITZ_ERR_ARGUMENT, "kibitz_options not initialised (call kibitz_options_init)");
    const kibitz_backend backend = options->backend;
    // Paths are UTF-8 on every platform, including Windows.
    const auto path = std::filesystem::path(
        std::u8string(reinterpret_cast<const char8_t*>(model_path)));
    auto ev = std::make_unique<kibitz_evaluator>();
    switch (backend) {
      case KIBITZ_BACKEND_NATIVE:
        ev->backend = kibitz::make_native_backend(path);
        break;
      case KIBITZ_BACKEND_ONNXRUNTIME:
#ifdef KIBITZ_WITH_ONNXRUNTIME
        ev->backend = kibitz::make_onnxruntime_backend(path, options->provider,
                                                       options->allow_cpu_fallback != 0);
        break;
#else
        return fail(KIBITZ_ERR_UNAVAILABLE, "built without ONNX Runtime (KIBITZ_WITH_ONNXRUNTIME=OFF)");
#endif
      default:
        return fail(KIBITZ_ERR_ARGUMENT, "unknown backend");
    }
    *out = ev.release();
    return KIBITZ_OK;
  });
}

kibitz_status kibitz_eval_batch(const kibitz_evaluator* ev, const char* const* fens, size_t count,
                                float* centipawns) {
  return guarded([&] {
    if (ev == nullptr || (count > 0 && (fens == nullptr || centipawns == nullptr)))
      return fail(KIBITZ_ERR_ARGUMENT, "null argument");
    std::vector<kibitz::Features> positions(count);
    std::string error;
    for (size_t i = 0; i < count; ++i) {
      if (fens[i] == nullptr) return fail(KIBITZ_ERR_ARGUMENT, "fens[" + std::to_string(i) + "] is null");
      if (!kibitz::encode_fen(fens[i], positions[i], error))
        return fail(KIBITZ_ERR_FEN, "fens[" + std::to_string(i) + "]: " + error);
    }
    ev->backend->evaluate(positions, std::span<float>(centipawns, count));
    for (size_t i = 0; i < count; ++i) centipawns[i] *= kCpScale;
    return KIBITZ_OK;
  });
}

kibitz_status kibitz_eval_fen(const kibitz_evaluator* ev, const char* fen, float* centipawns) {
  const char* one[] = {fen};
  const kibitz_status status = kibitz_eval_batch(ev, one, 1, centipawns);
  if (status == KIBITZ_ERR_FEN) g_last_error.erase(0, g_last_error.find(':') + 2);  // drop "fens[0]: "
  return status;
}

void kibitz_close(kibitz_evaluator* ev) { delete ev; }

const char* kibitz_last_error(void) { return g_last_error.c_str(); }

}  // extern "C"
