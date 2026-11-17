"""Export Kibitz's trained evaluation network for the C++ library.

Writes, from a Kibitz checkpoint:
  models/kibitz.onnx     the network as ONNX: float32 features[batch, 768] -> z[batch]
  models/kibitz.weights  the same weights as raw float32, for the hand-written C++ backend
  tests/data/parity.tsv  100 positions and PyTorch's output for each (the reference)
  tests/data/features.tsv  the feature indices Python computes for 20 positions

    uv run python export.py --kibitz ~/Documents/Codes/chess-agent

Every artifact is checked before the script exits: the ONNX file is run through ONNX Runtime and
must match PyTorch to 1e-5 on all 100 positions.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MAGIC, FORMAT_VERSION = b"KBTZ", 1


class DenseEvalNet(nn.Module):
    """Kibitz's network with layer 1 as an ordinary Linear over a 768-wide 0/1 vector.

    Kibitz trains layer 1 as an EmbeddingBag (a sum over ~30 active feature rows). ONNX
    Runtime and other inference engines handle a plain dense layer everywhere, and the
    results are identical: summing the active rows is a matrix-vector product with a 0/1 input.
    """

    def __init__(self, src):
        super().__init__()
        hidden = src.hidden
        self.l1 = nn.Linear(768, hidden)
        self.l2, self.l3 = src.l2, src.l3
        with torch.no_grad():
            self.l1.weight.copy_(src.first_layer().T)  # (hidden, 768)
            self.l1.bias.copy_(src.b1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.l1(features))
        h = torch.relu(self.l2(h))
        return self.l3(h).squeeze(-1)


def dense(feature_lists: list[list[int]]) -> torch.Tensor:
    x = torch.zeros(len(feature_lists), 768)
    for row, feats in enumerate(feature_lists):
        x[row, feats] = 1.0
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kibitz", type=Path, required=True, help="path to the Kibitz repo")
    args = ap.parse_args()
    train = args.kibitz / "train"
    sys.path.insert(0, str(train))
    from features import board_features  # noqa: E402  (Kibitz's own encoder)
    from model import EvalNet  # noqa: E402

    state = torch.load(train / "checkpoints" / "best.pt", map_location="cpu")
    src = EvalNet(state["l1.weight"].shape[1])
    src.load_state_dict(state)
    src.eval()
    model = DenseEvalNet(src).eval()
    hidden = src.hidden

    fens = json.loads((args.kibitz / "web" / "test" / "parity.json").read_text())["fens"]
    feats = [board_features(chess.Board(f)) for f in fens]
    x = dense(feats)
    with torch.no_grad():
        z_dense = model(x).numpy()
        from features import pad  # noqa: E402

        z_bag = src(torch.tensor([pad(f) for f in feats])).numpy()
    assert np.abs(z_dense - z_bag).max() < 1e-5, "dense rewrite changed the outputs"

    models = ROOT / "models"
    data = ROOT / "tests" / "data"
    models.mkdir(exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    onnx_path = models / "kibitz.onnx"
    torch.onnx.export(
        model,
        (x[:1],),
        onnx_path,
        input_names=["features"],
        output_names=["z"],
        dynamic_axes={"features": {0: "batch"}, "z": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    z_ort = session.run(["z"], {"features": x.numpy()})[0]
    ort_err = float(np.abs(z_ort - z_dense).max())
    assert ort_err < 1e-5, f"ONNX Runtime differs from PyTorch by {ort_err}"

    # Raw weights: magic, version, hidden, then float32 arrays, little-endian.
    w1 = src.first_layer().numpy().astype("<f4")  # (768, hidden), one row per feature
    tensors = [
        w1,
        src.b1.detach().numpy(),
        src.l2.weight.detach().numpy(),  # (32, hidden)
        src.l2.bias.detach().numpy(),
        src.l3.weight.detach().numpy(),  # (1, 32)
        src.l3.bias.detach().numpy(),
    ]
    with open(models / "kibitz.weights", "wb") as f:
        f.write(MAGIC + struct.pack("<II", FORMAT_VERSION, hidden))
        for t in tensors:
            f.write(np.ascontiguousarray(t, dtype="<f4").tobytes())

    with open(data / "parity.tsv", "w") as f:
        f.write("# z from PyTorch (checkpoint best.pt)\tFEN\n")
        for z, fen in zip(z_dense, fens, strict=True):
            f.write(f"{z:.9g}\t{fen}\n")
    with open(data / "features.tsv", "w") as f:
        f.write("# FEN\tsorted feature indices from Kibitz's Python encoder\n")
        for fen, ft in list(zip(fens, feats, strict=True))[:20]:
            f.write(f"{fen}\t{' '.join(map(str, sorted(ft)))}\n")

    print(
        f"hidden={hidden} params={sum(p.numel() for p in model.parameters())} "
        f"onnx={onnx_path.stat().st_size} bytes, max |ORT - PyTorch| = {ort_err:.2e}"
    )


if __name__ == "__main__":
    main()
