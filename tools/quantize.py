"""Quantize models/kibitz.onnx to INT8 and measure what it costs in accuracy.

Writes models/kibitz.int8.onnx: the same network in QDQ format (QuantizeLinear/DequantizeLinear
pairs around each MatMul), which is the format ONNX Runtime's QNN execution provider expects for
Qualcomm NPUs. On CPU, ONNX Runtime fuses the QDQ pairs into integer kernels.

    uv run python quantize.py --kibitz ~/Documents/Codes/chess-agent

- Weights: INT8, symmetric, one scale per output channel.
- The last layer (32 -> 1) stays float32, so large evaluations are not clipped.
- Activations: UINT8, one scale per tensor, calibrated on 2,000 positions from Kibitz's
  *training* split. The 100 parity positions and 20,000 validation positions used to measure the
  error below are never seen by the calibration.

The script fails if the INT8 model's error against the float32 model is larger than
--max-cp centipawns on any parity position.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parent.parent
CP = 400.0  # centipawns = 400 * z
PAD = 768  # Kibitz's padding index in its cached feature lists


def dense(padded: np.ndarray) -> np.ndarray:
    """Kibitz's cache stores each position as 32 feature indices padded with 768."""
    x = np.zeros((len(padded), PAD + 1), dtype=np.float32)
    np.put_along_axis(x, padded.astype(np.int64), 1.0, axis=1)
    return x[:, :PAD]


class Calibration(CalibrationDataReader):
    def __init__(self, x: np.ndarray, batch: int = 100):
        self.batches = iter([{"features": x[i : i + batch]} for i in range(0, len(x), batch)])

    def get_next(self):
        return next(self.batches, None)


def run(path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(["z"], {"features": x})[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kibitz", type=Path, required=True, help="path to the Kibitz repo")
    ap.add_argument("--max-cp", type=float, default=25.0, help="largest allowed parity error, cp")
    args = ap.parse_args()

    cache = np.load(args.kibitz / "train" / "data" / "cache.npz")
    rng = np.random.default_rng(0)
    calib = dense(cache["X_train"][rng.choice(len(cache["X_train"]), 2000, replace=False)])
    val = dense(cache["X_val"][:20000])

    fp32 = ROOT / "models" / "kibitz.onnx"
    int8 = ROOT / "models" / "kibitz.int8.onnx"
    with tempfile.TemporaryDirectory() as tmp:
        prepped = Path(tmp) / "prepped.onnx"
        quant_pre_process(str(fp32), str(prepped))
        quantize_static(
            str(prepped),
            str(int8),
            Calibration(calib),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            # Layer 3 (32 weights) stays float32. Quantizing its output caps evaluations at the
            # range seen in calibration: a +5,137 cp position came out 1,179 cp wrong.
            nodes_to_exclude=["/l3/Gemm"],
        )

    # Parity positions: the same 100 the C++ tests use, with PyTorch's answers.
    rows = [
        line.split("\t")
        for line in (ROOT / "tests" / "data" / "parity.tsv").read_text().splitlines()
        if not line.startswith("#")
    ]
    z_torch = np.array([float(z) for z, _ in rows], dtype=np.float32)
    sys.path.insert(0, str(args.kibitz / "train"))
    from features import board_features  # noqa: E402  (Kibitz's own encoder)

    parity = np.zeros((len(rows), PAD), dtype=np.float32)
    for i, (_, fen) in enumerate(rows):
        parity[i, board_features(chess.Board(fen))] = 1.0

    err_parity = np.abs(run(int8, parity) - z_torch) * CP
    z_fp32_val, z_int8_val = run(fp32, val), run(int8, val)
    err_val = np.abs(z_int8_val - z_fp32_val) * CP

    print(f"{int8.name}: {int8.stat().st_size} bytes (float32: {fp32.stat().st_size})")
    print(f"parity, 100 positions vs PyTorch: max {err_parity.max():.1f} cp, mean {err_parity.mean():.1f} cp")
    print(
        f"validation, 20,000 positions vs float32: mean {err_val.mean():.1f} cp, "
        f"p99 {np.percentile(err_val, 99):.1f} cp, max {err_val.max():.1f} cp"
    )
    assert err_parity.max() < args.max_cp, f"INT8 error {err_parity.max():.1f} cp > {args.max_cp}"


if __name__ == "__main__":
    main()
