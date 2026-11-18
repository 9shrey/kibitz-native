"""The shared library loaded from Python with ctypes. Run by CTest (see tests/CMakeLists.txt)."""

import os
import unittest
from pathlib import Path

from kibitz import Evaluator, KibitzError, has_provider

MODELS = Path(os.environ["KIBITZ_MODELS_DIR"])
DATA = Path(os.environ["KIBITZ_TEST_DATA_DIR"])


def parity():
    rows = [l.split("\t") for l in (DATA / "parity.tsv").read_text().splitlines() if not l.startswith("#")]
    return [(float(z), fen) for z, fen in rows]


class CtypesTest(unittest.TestCase):
    def check_backend(self, model, backend):
        cases = parity()
        with Evaluator(str(MODELS / model), backend) as ev:
            got = ev.eval_batch([fen for _, fen in cases])
            worst = max(abs(cp / 400 - z) for cp, (z, _) in zip(got, cases))
            self.assertLess(worst, 1e-4)
            self.assertAlmostEqual(ev.eval(cases[0][1]), got[0], places=2)

    def test_native_matches_pytorch(self):
        self.check_backend("kibitz.weights", "native")

    def test_onnxruntime_matches_pytorch(self):
        self.check_backend("kibitz.onnx", "onnxruntime")

    def test_int8_model_within_25_centipawns(self):
        cases = parity()
        with Evaluator(str(MODELS / "kibitz.int8.onnx"), "onnxruntime") as ev:
            got = ev.eval_batch([fen for _, fen in cases])
        self.assertLess(max(abs(cp - z * 400) for cp, (z, _) in zip(got, cases)), 25)

    def test_provider_choice(self):
        model = str(MODELS / "kibitz.onnx")
        with Evaluator(model, "onnxruntime", provider="qnn") as ev:  # falls back to CPU if absent
            self.assertIsInstance(ev.eval(parity()[0][1]), float)
        if not has_provider("qnn"):
            with self.assertRaises(KibitzError) as ctx:
                Evaluator(model, "onnxruntime", provider="qnn", allow_cpu_fallback=False)
            self.assertIn("ERR_UNAVAILABLE", str(ctx.exception))

    def test_errors_become_python_exceptions(self):
        with Evaluator(str(MODELS / "kibitz.weights")) as ev:
            with self.assertRaises(KibitzError) as ctx:
                ev.eval("definitely not a position")
            self.assertIn("ERR_FEN", str(ctx.exception))
        with self.assertRaises(KibitzError):
            Evaluator(str(MODELS / "missing.weights"))


if __name__ == "__main__":
    unittest.main()
