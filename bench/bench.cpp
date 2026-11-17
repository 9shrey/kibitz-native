// Positions per second for each backend at several batch sizes, single-threaded.
//
//   kibitz-bench [positions]      (default 200000)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "kibitz/kibitz.h"

namespace {

std::vector<std::string> load_fens() {
  std::ifstream in(std::string(KIBITZ_TEST_DATA_DIR) + "/parity.tsv");
  std::vector<std::string> fens;
  for (std::string line; std::getline(in, line);)
    if (!line.empty() && line[0] != '#') fens.push_back(line.substr(line.find('\t') + 1));
  return fens;
}

double best_seconds(const kibitz_evaluator* ev, const std::vector<const char*>& fens, size_t batch,
                    std::vector<float>& out) {
  double best = 1e9;
  for (int rep = 0; rep < 5; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < fens.size(); i += batch) {
      const size_t n = std::min(batch, fens.size() - i);
      if (kibitz_eval_batch(ev, fens.data() + i, n, out.data() + i) != KIBITZ_OK) {
        std::fprintf(stderr, "eval failed: %s\n", kibitz_last_error());
        std::exit(1);
      }
    }
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const size_t total = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 200000;
  const auto unique = load_fens();
  std::vector<const char*> fens(total);
  for (size_t i = 0; i < total; ++i) fens[i] = unique[i % unique.size()].c_str();
  std::vector<float> out(total);

  struct Backend {
    kibitz_backend id;
    const char* model;
    const char* name;
  };
  const Backend backends[] = {{KIBITZ_BACKEND_NATIVE, "kibitz.weights", "native C++"},
                              {KIBITZ_BACKEND_ONNXRUNTIME, "kibitz.onnx", "ONNX Runtime"}};

  std::printf("%zu positions, single thread, best of 5\n\n", total);
  std::printf("%-14s %8s %16s %14s\n", "backend", "batch", "positions/sec", "us/position");
  for (const auto& b : backends) {
    if (!kibitz_has_backend(b.id)) continue;
    kibitz_evaluator* ev = nullptr;
    if (kibitz_open((std::string(KIBITZ_MODELS_DIR) + "/" + b.model).c_str(), b.id, &ev) != KIBITZ_OK) {
      std::fprintf(stderr, "%s\n", kibitz_last_error());
      return 1;
    }
    for (const size_t batch : {size_t{1}, size_t{64}, size_t{1024}}) {
      const double s = best_seconds(ev, fens, batch, out);
      std::printf("%-14s %8zu %16.0f %14.2f\n", b.name, batch, static_cast<double>(total) / s,
                  s * 1e6 / static_cast<double>(total));
    }
    kibitz_close(ev);
  }
  return 0;
}
