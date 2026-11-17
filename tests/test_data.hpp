#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Reference data written by tools/export.py from Kibitz's Python code.

struct ParityCase {
  float z;  // PyTorch's output; centipawns = 400 * z
  std::string fen;
};

inline std::vector<std::vector<std::string>> read_tsv(const std::string& name) {
  std::ifstream in(std::string(KIBITZ_TEST_DATA_DIR) + "/" + name);
  if (!in) throw std::runtime_error("missing test data " + name);
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto tab = line.find('\t');
    rows.push_back({line.substr(0, tab), line.substr(tab + 1)});
  }
  return rows;
}

inline std::vector<ParityCase> parity_cases() {
  std::vector<ParityCase> cases;
  for (const auto& row : read_tsv("parity.tsv")) cases.push_back({std::stof(row[0]), row[1]});
  return cases;
}

inline std::string model_path(const char* file) { return std::string(KIBITZ_MODELS_DIR) + "/" + file; }
