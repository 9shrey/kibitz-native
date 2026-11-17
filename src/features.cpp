#include "features.hpp"

namespace kibitz {
namespace {

int plane_of(char c) {
  switch (c) {
    case 'P': return 0;
    case 'N': return 1;
    case 'B': return 2;
    case 'R': return 3;
    case 'Q': return 4;
    case 'K': return 5;
    case 'p': return 6;
    case 'n': return 7;
    case 'b': return 8;
    case 'r': return 9;
    case 'q': return 10;
    case 'k': return 11;
    default: return -1;
  }
}

}  // namespace

bool encode_fen(std::string_view fen, Features& out, std::string& error) {
  out.count = 0;
  const auto space = fen.find(' ');
  if (space == std::string_view::npos) {
    error = "FEN has no side to move";
    return false;
  }
  const std::string_view placement = fen.substr(0, space);
  const std::string_view rest = fen.substr(space + 1);
  if (rest.empty() || (rest[0] != 'w' && rest[0] != 'b') ||
      (rest.size() > 1 && rest[1] != ' ')) {
    error = "side to move must be 'w' or 'b'";
    return false;
  }
  const bool black = rest[0] == 'b';

  int rank = 7, file = 0, white_kings = 0, black_kings = 0;
  for (const char c : placement) {
    if (c == '/') {
      if (file != 8) {
        error = "rank " + std::to_string(rank + 1) + " does not have 8 squares";
        return false;
      }
      if (--rank < 0) {
        error = "more than 8 ranks";
        return false;
      }
      file = 0;
      continue;
    }
    if (c >= '1' && c <= '8') {
      file += c - '0';
    } else {
      int plane = plane_of(c);
      if (plane < 0) {
        error = std::string("unknown piece '") + c + "'";
        return false;
      }
      if (file > 7) {
        error = "rank " + std::to_string(rank + 1) + " has more than 8 squares";
        return false;
      }
      if (out.count == kMaxPieces) {
        error = "more than 32 pieces";
        return false;
      }
      white_kings += c == 'K';
      black_kings += c == 'k';
      int square = rank * 8 + file;
      if (black) {
        plane = (plane + 6) % 12;
        square ^= 56;
      }
      out.index[static_cast<std::size_t>(out.count++)] =
          static_cast<std::uint16_t>(plane * 64 + square);
      ++file;
    }
    if (file > 8) {
      error = "rank " + std::to_string(rank + 1) + " has more than 8 squares";
      return false;
    }
  }
  if (rank != 0 || file != 8) {
    error = "board must have exactly 8 ranks of 8 squares";
    return false;
  }
  if (white_kings != 1 || black_kings != 1) {
    error = "each side needs exactly one king";
    return false;
  }
  return true;
}

}  // namespace kibitz
