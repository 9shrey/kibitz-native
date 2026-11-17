#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>

#include "features.hpp"
#include "test_data.hpp"

using kibitz::encode_fen;
using kibitz::Features;

namespace {

std::vector<int> sorted_indices(const Features& f) {
  std::vector<int> v(f.index.begin(), f.index.begin() + f.count);
  std::sort(v.begin(), v.end());
  return v;
}

}  // namespace

TEST(Features, MatchKibitzPythonEncoderOn20Positions) {
  const auto rows = read_tsv("features.tsv");
  ASSERT_EQ(rows.size(), 20u);
  for (const auto& row : rows) {
    std::vector<int> expected;
    std::istringstream ss(row[1]);
    for (int i; ss >> i;) expected.push_back(i);
    Features f;
    std::string error;
    ASSERT_TRUE(encode_fen(row[0], f, error)) << row[0] << ": " << error;
    EXPECT_EQ(sorted_indices(f), expected) << row[0];
  }
}

TEST(Features, StartPositionHas32PiecesAndKingsWhereExpected) {
  Features f;
  std::string error;
  ASSERT_TRUE(encode_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", f, error));
  EXPECT_EQ(f.count, 32);
  const auto idx = sorted_indices(f);
  EXPECT_TRUE(std::binary_search(idx.begin(), idx.end(), 5 * 64 + 4));    // our king, e1
  EXPECT_TRUE(std::binary_search(idx.begin(), idx.end(), 11 * 64 + 60));  // their king, e8
}

TEST(Features, BlackToMoveIsTheMirrorOfWhiteToMove) {
  // The same position with colours swapped and the board flipped must encode identically.
  Features w, b;
  std::string error;
  ASSERT_TRUE(encode_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", w, error));
  ASSERT_TRUE(encode_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1", b, error));
  EXPECT_EQ(sorted_indices(w), sorted_indices(b));
}

TEST(Features, OnlyTheFirstTwoFieldsMatter) {
  Features a, b;
  std::string error;
  ASSERT_TRUE(encode_fen("4k3/8/8/8/8/8/8/4K3 w", a, error));
  ASSERT_TRUE(encode_fen("4k3/8/8/8/8/8/8/4K3 w - - 37 90", b, error));
  EXPECT_EQ(sorted_indices(a), sorted_indices(b));
}

class InvalidFen : public ::testing::TestWithParam<const char*> {};

TEST_P(InvalidFen, IsRejectedWithAMessage) {
  Features f;
  std::string error;
  EXPECT_FALSE(encode_fen(GetParam(), f, error));
  EXPECT_FALSE(error.empty());
}

INSTANTIATE_TEST_SUITE_P(Features, InvalidFen, ::testing::Values(
    "",                                                     // empty
    "4k3/8/8/8/8/8/8/4K3",                                  // no side to move
    "4k3/8/8/8/8/8/8/4K3 x",                                // bad side to move
    "4k3/8/8/8/8/8/8/4K3 white",                            // side to move is one letter
    "4k3/8/8/8/8/8/4K3 w",                                  // 7 ranks
    "4k3/8/8/8/8/8/8/8/4K3 w",                              // 9 ranks
    "4k4/8/8/8/8/8/8/4K3 w",                                // rank of 9 squares
    "4k2/8/8/8/8/8/8/4K3 w",                                // rank of 7 squares
    "4k3/8/8/8/8/8/8/4KX2 w",                               // unknown piece
    "4k3/8/8/8/8/8/8/8 w",                                  // no white king
    "4k3/8/8/8/8/8/8/3KK3 w",                               // two white kings
    "4k3/8/8/8/8/8/8/4K39 w",                               // digit 9
    "PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/8/8/4K2k w"  // more than 32 pieces
));
