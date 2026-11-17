#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include "features.hpp"

namespace kibitz {

/// Thrown inside the library and turned into a kibitz_status at the C boundary.
struct Error : std::runtime_error {
  int status;
  Error(int status_code, const std::string& message)
      : std::runtime_error(message), status(status_code) {}
};

/// One way of running the network. Implementations must allow concurrent evaluate() calls.
class Backend {
 public:
  virtual ~Backend() = default;
  /// Writes the raw network output z for each position (centipawns = 400 * z).
  virtual void evaluate(std::span<const Features> positions, std::span<float> z) const = 0;
};

std::unique_ptr<Backend> make_native_backend(const std::filesystem::path& weights);

#ifdef KIBITZ_WITH_ONNXRUNTIME
std::unique_ptr<Backend> make_onnxruntime_backend(const std::filesystem::path& model);
#endif

}  // namespace kibitz
