# kibitz-native

The neural network from [Kibitz](https://github.com/9shrey/chess-agent), my browser chess engine,
as a C/C++ shared library (`libkibitz.so` / `kibitz.dll` / `libkibitz.dylib`) with two
interchangeable backends:

- **ONNX Runtime**, loading the network exported from PyTorch to ONNX
- **native C++**, a hand-written forward pass with no dependencies

The ONNX Runtime backend can run on a chosen **execution provider** (CPU, Qualcomm's QNN for
Snapdragon NPUs, or XNNPACK), and there is an **INT8-quantized** copy of the model in the QDQ
format QNN expects, within 18 centipawns of PyTorch at worst on the test positions.

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
   │      │  tools/quantize.py                            │
   │      └──► models/kibitz.int8.onnx ── ONNX Runtime ───┤   (CPU, QNN or XNNPACK provider)
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
| ONNX Runtime, INT8 model | 50,166 /s | 198,977 /s | 214,562 /s |

The native backend is also ahead on every platform CI builds for (GitHub-hosted runners, batch
1024, 100,000 positions; shared machines, so treat these as rough):

| CI runner | native C++ | ONNX Runtime | native vs ORT |
|---|---:|---:|---:|
| macOS arm64, Clang | 544,914 /s | 148,636 /s | 3.7× |
| Windows x64, MSVC | 442,346 /s | 281,998 /s | 1.6× |
| Linux x64, GCC | 516,777 /s | 457,385 /s | 1.1× |

Two things in the first table took work to understand:

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

## Execution providers and INT8

```c
kibitz_options o;
kibitz_options_init(&o);              /* sets struct_size and the defaults */
o.backend = KIBITZ_BACKEND_ONNXRUNTIME;
o.provider = KIBITZ_PROVIDER_QNN;     /* Hexagon NPU on Snapdragon */
o.allow_cpu_fallback = 0;             /* fail instead of silently running on the CPU */
kibitz_open_ex("models/kibitz.int8.onnx", &o, &ev);
```

- **Providers.** `KIBITZ_PROVIDER_CPU`, `KIBITZ_PROVIDER_QNN` (HTP backend, `QnnHtp.dll` /
  `libQnnHtp.so`) and `KIBITZ_PROVIDER_XNNPACK`. `kibitz_has_provider()` asks the loaded ONNX
  Runtime which ones it was built with. With `allow_cpu_fallback = 1` (the default) a missing
  provider, or operators it can't run, go to the CPU. With `0` a missing provider is
  `KIBITZ_ERR_UNAVAILABLE`, and the session is created with `session.disable_cpu_ep_fallback`,
  so a model the provider can't run entirely fails at open instead of quietly running on the CPU.
- **Why a struct with `struct_size`.** New options can be added later without a new function or a
  broken ABI: later fields go on the end, and the library uses `struct_size` to tell which fields
  the caller knows about. A zeroed, uninitialised struct is rejected.
  `kibitz_open()` is unchanged and still works.
- **INT8 model.** [`tools/quantize.py`](tools/quantize.py) runs ONNX Runtime's static
  quantization: INT8 weights (symmetric, per output channel) and UINT8 activations, calibrated on
  2,000 positions from Kibitz's training split, in QDQ format. It is a quarter of the size
  (213 KB vs 821 KB).
- **Accuracy.** 18 cp worst and 3.3 cp mean against PyTorch on the 100 test positions; mean 3.6 cp
  and p99 19.6 cp against the float32 model on 20,000 validation positions the calibration never
  saw. The first attempt quantized every layer and was 1,179 cp wrong on a +5,137 cp position:
  the output layer's 8-bit range came from calibration and capped large scores. Keeping that
  layer (32 weights) in float32 fixed it.
- **Speed on CPU.** INT8 makes ONNX Runtime 1.6× faster at batch 1 and 1.26× at batch 1024, but
  the native float32 backend is still faster. The INT8 model is really for NPUs, which run
  integer maths far more efficiently than float.
- **Tested so far:** CI's ONNX Runtime packages are the standard CPU builds, so the tests cover
  provider selection, fallback and the `ERR_UNAVAILABLE` path, plus INT8 accuracy on CPU. The
  test that runs the INT8 model on QNN with fallback off only runs where the QNN build of ONNX
  Runtime is present; it hasn't been run on Snapdragon hardware yet.

## The C API

[`include/kibitz/kibitz.h`](include/kibitz/kibitz.h)

- **No C++ crosses the boundary.** Every function returns a `kibitz_status`; exceptions are
  caught inside and turned into codes, with the message from `kibitz_last_error()`, which is
  thread-local so threads never see each other's errors.
- **Only the API is exported.** The library is built with hidden visibility and
  `__declspec(dllexport)` / `visibility("default")` on the ten public functions; `nm -D` on the
  `.so` lists exactly those ten. It has a proper SONAME (`libkibitz.so.0`).
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

45 tests, run by CTest on every platform in CI:

- **Parity:** both backends against PyTorch on 100 positions, against each other, and batch
  against one-at-a-time.
- **Feature encoding:** the C++ FEN encoder against Kibitz's Python encoder on 20 positions, the
  black-to-move board mirroring, and 13 malformed FENs.
- **API contract:** null arguments, missing and corrupt model files, per-thread errors, and 8
  threads sharing one evaluator.
- **Providers and INT8:** options defaults, uninitialised options rejected, a missing provider
  failing without fallback and running on the CPU with it, and the INT8 model within 25 cp of
  PyTorch.
- **Python:** the same library loaded through ctypes, including the provider options.

The sanitizer build runs the whole suite under ASan and UBSan. ONNX Runtime's prebuilt library
makes one 14-byte allocation it never frees, so [`tests/lsan.supp`](tests/lsan.supp) suppresses
leaks *inside `libonnxruntime.so` only*; a leak planted in libkibitz's own code was checked to
still fail the build.

Breaking the board mirroring on purpose fails 5 tests, including both PyTorch parity tests.

## Regenerating the model files

```bash
cd tools && uv run python export.py --kibitz /path/to/chess-agent
uv run python quantize.py --kibitz /path/to/chess-agent   # INT8 model; fails if worse than 25 cp
```

## Not done

- Not yet run on an NPU: QNN is wired up and tested for selection and fallback, not on Snapdragon
  hardware. No DirectML or CUDA.
- No SIMD intrinsics: the native speed-up comes from data layout the compiler can vectorise on
  its own, which keeps the code portable across x64 and ARM.
- The native backend is still float32; INT8 weights would cut its memory traffic to a quarter.
