/* kibitz-eval: evaluate positions from the command line, in plain C against the public API.
 *
 *   kibitz-eval [--onnx] MODEL "FEN" ["FEN" ...]
 *   kibitz-eval [--onnx] MODEL < positions.txt        (one FEN per line)
 */
#include <stdio.h>
#include <string.h>

#include "kibitz/kibitz.h"

static int eval_one(const kibitz_evaluator* ev, const char* fen) {
  float cp = 0.0f;
  if (kibitz_eval_fen(ev, fen, &cp) != KIBITZ_OK) {
    fprintf(stderr, "kibitz-eval: %s: %s\n", fen, kibitz_last_error());
    return 1;
  }
  printf("%+8.1f\t%s\n", (double)cp, fen);
  return 0;
}

int main(int argc, char** argv) {
  int arg = 1;
  kibitz_backend backend = KIBITZ_BACKEND_NATIVE;
  if (arg < argc && strcmp(argv[arg], "--onnx") == 0) {
    backend = KIBITZ_BACKEND_ONNXRUNTIME;
    ++arg;
  }
  if (arg >= argc) {
    fprintf(stderr, "usage: kibitz-eval [--onnx] MODEL [FEN ...]\n");
    return 2;
  }

  kibitz_evaluator* ev = NULL;
  if (kibitz_open(argv[arg], backend, &ev) != KIBITZ_OK) {
    fprintf(stderr, "kibitz-eval: %s\n", kibitz_last_error());
    return 1;
  }
  ++arg;

  int failures = 0;
  if (arg < argc) {
    for (; arg < argc; ++arg) failures += eval_one(ev, argv[arg]);
  } else {
    char line[512];
    while (fgets(line, (int)sizeof line, stdin)) {
      line[strcspn(line, "\r\n")] = '\0';
      if (line[0] != '\0') failures += eval_one(ev, line);
    }
  }
  kibitz_close(ev);
  return failures ? 1 : 0;
}
