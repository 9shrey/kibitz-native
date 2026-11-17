// The network written out by hand: no inference engine, no dependencies.
//
// Layer 1 is the expensive one (768 inputs x 256 hidden), but a chess position only has ~30
// pieces, so its input vector is ~96% zeros. Instead of a 768 x 256 matrix-vector product this
// adds up the ~30 weight rows of the pieces actually on the board: about 25x less arithmetic
// than the dense layer ONNX Runtime runs.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#include "backend.hpp"
#include "kibitz/kibitz.h"

namespace kibitz {
namespace {

constexpr char kMagic[4] = {'K', 'B', 'T', 'Z'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::size_t kL2 = 32;
constexpr int kMaxHidden = 1024;

std::uint32_t read_u32_le(const unsigned char* p) {
  return static_cast<std::uint32_t>(p[0]) | static_cast<std::uint32_t>(p[1]) << 8 |
         static_cast<std::uint32_t>(p[2]) << 16 | static_cast<std::uint32_t>(p[3]) << 24;
}

class NativeBackend final : public Backend {
 public:
  explicit NativeBackend(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error(KIBITZ_ERR_IO, "cannot open " + path.string());
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
    if (bytes.size() < 12 || std::memcmp(bytes.data(), kMagic, 4) != 0)
      throw Error(KIBITZ_ERR_FORMAT, path.string() + " is not a Kibitz weights file");
    if (read_u32_le(bytes.data() + 4) != kFormatVersion)
      throw Error(KIBITZ_ERR_FORMAT, "unsupported weights format version");
    hidden_ = static_cast<int>(read_u32_le(bytes.data() + 8));
    if (hidden_ <= 0 || hidden_ > kMaxHidden)
      throw Error(KIBITZ_ERR_FORMAT, "hidden layer size out of range");

    const std::size_t h = static_cast<std::size_t>(hidden_);
    const std::size_t floats = kInputs * h + h + kL2 * h + kL2 + kL2 + 1;
    if (bytes.size() != 12 + floats * sizeof(float))
      throw Error(KIBITZ_ERR_FORMAT, "weights file has the wrong size for its header");

    params_.resize(floats);
    // Stored little-endian; every platform this builds for is little-endian too.
    std::memcpy(params_.data(), bytes.data() + 12, floats * sizeof(float));
    w1_ = params_.data();
    b1_ = w1_ + kInputs * h;
    w2_ = b1_ + h;
    b2_ = w2_ + kL2 * h;
    w3_ = b2_ + kL2;
    b3_ = w3_ + kL2;

    // Layer 2 stored transposed ([hidden][32]) so evaluate() can walk it one input at a time.
    w2t_.resize(h * kL2);
    for (std::size_t k = 0; k < kL2; ++k)
      for (std::size_t j = 0; j < h; ++j) w2t_[j * kL2 + k] = w2_[k * h + j];
  }

  void evaluate(std::span<const Features> positions, std::span<float> z) const override {
    const std::size_t h = static_cast<std::size_t>(hidden_);
    std::array<float, kMaxHidden> acc{};  // on the stack: no allocation per position
    for (std::size_t p = 0; p < positions.size(); ++p) {
      const Features& f = positions[p];

      std::copy(b1_, b1_ + h, acc.begin());
      for (int i = 0; i < f.count; ++i) {
        const float* row = w1_ + static_cast<std::size_t>(f.index[static_cast<std::size_t>(i)]) * h;
        for (std::size_t j = 0; j < h; ++j) acc[j] += row[j];
      }
      for (std::size_t j = 0; j < h; ++j) acc[j] = std::max(acc[j], 0.0f);

      // Layer 2, input by input: each step adds one hidden value times its 32 weights into 32
      // independent sums. With no single running total, the compiler can vectorise this loop
      // without -ffast-math (a dot product can't be: reordering float adds changes the result).
      // Hidden values that ReLU zeroed (about half) are skipped outright.
      std::array<float, kL2> l2{};
      std::copy(b2_, b2_ + kL2, l2.begin());
      for (std::size_t j = 0; j < h; ++j) {
        const float a = acc[j];
        if (a == 0.0f) continue;
        const float* col = w2t_.data() + j * kL2;
        for (std::size_t k = 0; k < kL2; ++k) l2[k] += col[k] * a;
      }
      float out = *b3_;
      for (std::size_t k = 0; k < kL2; ++k) out += w3_[k] * std::max(l2[k], 0.0f);
      z[p] = out;
    }
  }

 private:
  int hidden_ = 0;
  std::vector<float> params_;
  std::vector<float> w2t_;     // layer 2 transposed: [hidden][32]
  const float* w1_ = nullptr;  // [768][hidden], one row per input feature
  const float* b1_ = nullptr;  // [hidden]
  const float* w2_ = nullptr;  // [32][hidden]
  const float* b2_ = nullptr;  // [32]
  const float* w3_ = nullptr;  // [32]
  const float* b3_ = nullptr;  // [1]
};

}  // namespace

std::unique_ptr<Backend> make_native_backend(const std::filesystem::path& weights) {
  return std::make_unique<NativeBackend>(weights);
}

}  // namespace kibitz
