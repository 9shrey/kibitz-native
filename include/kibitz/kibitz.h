/*
 * kibitz.h - C API for the Kibitz chess evaluation network.
 *
 * A plain C interface so the library can be loaded from any language (C, C++, Python ctypes,
 * C#, Rust) and swapped without recompiling callers. No C++ types or exceptions cross this
 * boundary: every function reports failure through a kibitz_status code, and
 * kibitz_last_error() gives the message for the calling thread.
 *
 * Thread safety: one evaluator may be used from many threads at once for evaluation.
 * Opening and closing must not race with evaluation on the same evaluator.
 */
#ifndef KIBITZ_KIBITZ_H
#define KIBITZ_KIBITZ_H

#include <stddef.h>

/* Symbol export: only the functions below are visible outside the shared library. */
#if defined(_WIN32)
#  if defined(KIBITZ_BUILDING)
#    define KIBITZ_API __declspec(dllexport)
#  else
#    define KIBITZ_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define KIBITZ_API __attribute__((visibility("default")))
#else
#  define KIBITZ_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kibitz_evaluator kibitz_evaluator;

typedef enum kibitz_backend {
  KIBITZ_BACKEND_NATIVE = 0,      /* hand-written C++ forward pass; loads a .weights file */
  KIBITZ_BACKEND_ONNXRUNTIME = 1  /* ONNX Runtime; loads a .onnx file */
} kibitz_backend;

typedef enum kibitz_status {
  KIBITZ_OK = 0,
  KIBITZ_ERR_ARGUMENT = 1,    /* a null pointer or an out-of-range value */
  KIBITZ_ERR_IO = 2,          /* the model file could not be read */
  KIBITZ_ERR_FORMAT = 3,      /* the model file is not a valid model */
  KIBITZ_ERR_FEN = 4,         /* a position string is not valid FEN */
  KIBITZ_ERR_RUNTIME = 5,     /* the inference engine failed */
  KIBITZ_ERR_UNAVAILABLE = 6  /* the backend was not compiled into this build */
} kibitz_status;

/* Library version, e.g. "0.1.0". */
KIBITZ_API const char* kibitz_version(void);

/* Whether a backend is compiled into this build (1) or not (0). */
KIBITZ_API int kibitz_has_backend(kibitz_backend backend);

/* Load a model. On success *out owns the evaluator; free it with kibitz_close. */
KIBITZ_API kibitz_status kibitz_open(const char* model_path, kibitz_backend backend,
                                     kibitz_evaluator** out);

/* Evaluate one position. *centipawns is from the side to move's point of view. */
KIBITZ_API kibitz_status kibitz_eval_fen(const kibitz_evaluator* ev, const char* fen,
                                         float* centipawns);

/* Evaluate `count` positions in one call (faster than one at a time with ONNX Runtime).
 * If any FEN is invalid nothing is evaluated and KIBITZ_ERR_FEN names the first bad index. */
KIBITZ_API kibitz_status kibitz_eval_batch(const kibitz_evaluator* ev, const char* const* fens,
                                           size_t count, float* centipawns);

/* Free an evaluator. Passing NULL is allowed and does nothing. */
KIBITZ_API void kibitz_close(kibitz_evaluator* ev);

/* The message for the most recent failure on the calling thread ("" if none). The pointer is
 * valid until the next kibitz_* call on the same thread. */
KIBITZ_API const char* kibitz_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* KIBITZ_KIBITZ_H */
