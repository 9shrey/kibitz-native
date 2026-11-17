#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace kibitz {

inline constexpr int kInputs = 768;      // 12 piece planes x 64 squares
inline constexpr int kMaxPieces = 32;

/// The active inputs for one position: indices into the 768-wide input, one per piece.
struct Features {
  std::array<std::uint16_t, kMaxPieces> index{};
  int count = 0;
};

/// Encode a FEN the same way Kibitz trains: index = plane * 64 + square (a1 = 0 .. h8 = 63),
/// planes 0-5 are the side to move's P N B R Q K and 6-11 the opponent's. With black to move
/// the board is mirrored (square ^ 56) so the network always sees the mover at the bottom.
///
/// Returns false and sets `error` for anything that is not a legal-looking FEN: wrong rank
/// count or width, unknown pieces, a missing side to move, or not exactly one king per side.
bool encode_fen(std::string_view fen, Features& out, std::string& error);

}  // namespace kibitz
