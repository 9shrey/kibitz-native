"""Python bindings for libkibitz through ctypes: no compiler or build step on the Python side.

    from kibitz import Evaluator
    with Evaluator("models/kibitz.onnx", backend="onnxruntime") as ev:
        ev.eval("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        ev.eval_batch([...])

Set KIBITZ_LIBRARY to the path of libkibitz.so / kibitz.dll / libkibitz.dylib.
"""

from __future__ import annotations

import ctypes
import os

_BACKENDS = {"native": 0, "onnxruntime": 1}
_STATUS = {
    1: "ERR_ARGUMENT",
    2: "ERR_IO",
    3: "ERR_FORMAT",
    4: "ERR_FEN",
    5: "ERR_RUNTIME",
    6: "ERR_UNAVAILABLE",
}


class KibitzError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{_STATUS.get(status, status)}: {message}")
        self.status = status


def _load(path: str | None = None) -> ctypes.CDLL:
    path = path or os.environ["KIBITZ_LIBRARY"]
    if os.name == "nt":
        # Windows does not search a DLL's own folder for its dependencies (onnxruntime.dll).
        os.add_dll_directory(os.path.dirname(os.path.abspath(path)))
    lib = ctypes.CDLL(path)
    lib.kibitz_version.restype = ctypes.c_char_p
    lib.kibitz_last_error.restype = ctypes.c_char_p
    lib.kibitz_has_backend.argtypes = [ctypes.c_int]
    lib.kibitz_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    lib.kibitz_eval_fen.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
    lib.kibitz_eval_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.kibitz_close.argtypes = [ctypes.c_void_p]
    lib.kibitz_close.restype = None
    return lib


class Evaluator:
    def __init__(self, model: str, backend: str = "native", library: str | None = None):
        self._lib = _load(library)
        self._handle = ctypes.c_void_p()
        self._check(self._lib.kibitz_open(model.encode(), _BACKENDS[backend], ctypes.byref(self._handle)))

    def _check(self, status: int) -> None:
        if status != 0:
            raise KibitzError(status, self._lib.kibitz_last_error().decode())

    def eval(self, fen: str) -> float:
        """Centipawns from the side to move's point of view."""
        out = ctypes.c_float()
        self._check(self._lib.kibitz_eval_fen(self._handle, fen.encode(), ctypes.byref(out)))
        return out.value

    def eval_batch(self, fens: list[str]) -> list[float]:
        arr = (ctypes.c_char_p * len(fens))(*(f.encode() for f in fens))
        out = (ctypes.c_float * len(fens))()
        self._check(self._lib.kibitz_eval_batch(self._handle, arr, len(fens), out))
        return list(out)

    def close(self) -> None:
        if self._handle:
            self._lib.kibitz_close(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
