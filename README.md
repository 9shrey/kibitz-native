# kibitz-native

The neural network from [Kibitz](https://github.com/9shrey/chess-agent), my browser chess engine,
as a C/C++ shared library (`libkibitz.so` / `kibitz.dll` / `libkibitz.dylib`) with two
interchangeable backends:

- **ONNX Runtime**, loading the network exported from PyTorch to ONNX
- **native C++**, a hand-written forward pass with no dependencies

Both are behind one plain C API, so the library can be called from C, C++, Python, C#, Rust or
anything else with a C FFI. Built with CMake and tested in CI with GCC and Clang on Linux, MSVC on
Windows, and Clang on macOS.

```c
#include <kibitz/kibitz.h>

kibitz_evaluator* ev;
kibitz_open("models/kibitz.onnx", KIBITZ_BACKEND_ONNXRUNTIME, &ev);
float cp;
kibitz_eval_fen(ev, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", &cp);
kibitz_close(ev);
```

```text
$ kibitz-eval models/kibitz.weights "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
 +1875.8	4k3/8/8/8/8/8/8/3QK3 w - - 0 1
```

## From PyTorch to a .so

```text
Kibitz checkpoint (PyTorch)
   │  tools/export.py
   ├──► models/kibitz.onnx      ── ONNX Runtime backend ──┐
   ├──► models/kibitz.weights   ── native C++ backend  ───┼──► libkibitz (C API) ──► C, C++, Python (ctypes)
   └──► tests/data/parity.tsv   (PyTorch's answers for 100 positions, the reference for every test)
```

Kibitz trains its first layer as an `EmbeddingBag`: a sum over the ~30 squares that have a piece
on them. ONNX Runtime and most inference engines expect an ordinary dense layer, so the exporter
rewrites it as one (`Linear(768, 256)` over a 0/1 input), checks the rewrite gives identical
outputs, exports to ONNX (opset 18, dynamic batch size), and runs the ONNX file through ONNX
Runtime to confirm it matches PyTorch before writing anything.

**Accuracy:** on 100 held-out positions, both backends match PyTorch to within **7 × 10⁻⁷**
(about 0.0003 centipawns).

## Performance

Single thread, Intel i7-9750H, 200,000 positions, best of 5 (`kibitz-bench`):

| Backend | Batch 1 | Batch 64 | Batch 1024 |
|---|---:|---:|---:|
| native C++ | 301,672 /s | 307,610 /s | 318,148 /s |
| ONNX Runtime 1.30.1 | 31,246 /s | 164,020 /s | 170,547 /s |

Two things in this table took work to understand:

**ONNX Runtime needs batches.** At batch size 1 its per-call overhead (building tensors,
dispatching the graph) costs about 30 µs, ten times the network's actual arithmetic. Batching
spreads that overhead out, which is why the library's batch API exists.

**The first native version was 2× slower than ONNX Runtime, not faster.** It already skipped
~96% of layer 1 (adding up the ~30 active rows instead of multiplying a 768-wide vector), yet
managed only 80,000 positions/s. The bottleneck was layer 2, written as 32 dot products. A dot
product is one running sum, and compilers won't vectorise it without `-ffast-math`, because
reordering float additions changes the result. Storing layer 2 transposed turns it into 32
*independent* sums updated together, which the compiler vectorises with exact IEEE semantics; it
also lets the loop skip the ~50% of hidden units that ReLU zeroed. Result: **4× faster** (80k to
318k positions/s), with identical output.

## The C API

[`include/kibitz/kibitz.h`](include/kibitz/kibitz.h)

- **No C++ crosses the boundary.** Every function returns a `kibitz_status`; exceptions are
  caught inside and turned into codes, with the message from `kibitz_last_error()`, which is
  thread-local so threads never see each other's errors.
- **Only the API is exported.** The library is built with hidden visibility and
  `__declspec(dllexport)` / `visibility("default")` on the seven public functions; `nm -D` on the
  `.so` lists exactly those seven. It has a proper SONAME (`libkibitz.so.0`).
- **Thread safe.** One evaluator can be shared by many threads (tested with 8 threads × 25
  rounds against the single-threaded answers).
- **Validated input.** Malformed FENs (wrong rank widths, unknown pieces, missing kings, more than
  32 pieces) and corrupt weights files (bad magic, truncated, impossible sizes) return errors
  instead of reading out of bounds.
- **Relocatable.** On Linux and macOS the library finds `libonnxruntime` next to itself via an
  `$ORIGIN` / `@loader_path` RPATH; on Windows CMake copies `onnxruntime.dll` beside `kibitz.dll`.

## Build

Needs CMake 3.24+ and a C++20 compiler. ONNX Runtime 1.30.1 and GoogleTest are downloaded by
CMake (ONNX Runtime is checked against its SHA-256); set `ONNXRUNTIME_ROOT` to use your own, or
`-DKIBITZ_WITH_ONNXRUNTIME=OFF` to build only the native backend.

```bash
cmake --preset release            # Ninja, Release, warnings as errors
cmake --build --preset release
ctest --preset release
./build/release/bin/kibitz-bench
```

| Preset | |
|---|---|
| `release` | Ninja, GCC or Clang, `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror` |
| `sanitize` | Debug build with AddressSanitizer and UndefinedBehaviorSanitizer |
| `msvc` | Visual Studio 2022, x64, `/W4 /WX` |

From Python, no compiling needed:

```python
from kibitz import Evaluator          # python/kibitz.py, uses ctypes
with Evaluator("models/kibitz.onnx", backend="onnxruntime") as ev:
    print(ev.eval_batch(["4k3/8/8/8/8/8/8/3QK3 w - - 0 1"]))
```

## Tests

35 tests, run by CTest on every platform in CI:

- **Parity:** both backends against PyTorch on 100 positions, against each other, and batch
  against one-at-a-time.
- **Feature encoding:** the C++ FEN encoder against Kibitz's Python encoder on 20 positions, the
  black-to-move board mirroring, and 13 malformed FENs.
- **API contract:** null arguments, missing and corrupt model files, per-thread errors, and 8
  threads sharing one evaluator.
- **Python:** the same library loaded through ctypes.

The sanitizer build runs the whole suite under ASan and UBSan. ONNX Runtime's prebuilt library
makes one 14-byte allocation it never frees, so [`tests/lsan.supp`](tests/lsan.supp) suppresses
leaks *inside `libonnxruntime.so` only*; a leak planted in libkibitz's own code was checked to
still fail the build.

Breaking the board mirroring on purpose fails 5 tests, including both PyTorch parity tests.

## Regenerating the model files

```bash
cd tools && uv run python export.py --kibitz /path/to/chess-agent
```

## Not done

- No GPU or NPU execution providers (CUDA, QNN, DirectML); CPU only.
- No SIMD intrinsics: the native speed-up comes from data layout the compiler can vectorise on
  its own, which keeps the code portable across x64 and ARM.
- Weights are float32; int8 quantisation would roughly halve the native backend's memory traffic.
