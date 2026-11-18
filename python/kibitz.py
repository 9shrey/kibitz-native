"""Python bindings for libkibitz through ctypes: no compiler or build step on the Python side.

    from kibitz import Evaluator
    with Evaluator("models/kibitz.int8.onnx", backend="onnxruntime", provider="qnn") as ev:
        ev.eval("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        ev.eval_batch([...])

Set KIBITZ_LIBRARY to the path of libkibitz.so / kibitz.dll / libkibitz.dylib.
"""

from __future__ import annotations

import ctypes
import os

_BACKENDS = {"native": 0, "onnxruntime": 1}
_PROVIDERS = {"cpu": 0, "qnn": 1, "xnnpack": 2}
_STATUS = {
    1: "ERR_ARGUMENT",
    2: "ERR_IO",
    3: "ERR_FORMAT",
    4: "ERR_FEN",
    5: "ERR_RUNTIME",
    6: "ERR_UNAVAILABLE",
}


class _Options(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("backend", ctypes.c_int),
        ("provider", ctypes.c_int),
        ("allow_cpu_fallback", ctypes.c_int),
    ]


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
    lib.kibitz_options_init.argtypes = [ctypes.POINTER(_Options)]
    lib.kibitz_options_init.restype = None
    lib.kibitz_open_ex.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_Options),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.kibitz_has_provider.argtypes = [ctypes.c_int]
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


def has_provider(provider: str, library: str | None = None) -> bool:
    """Whether this build includes an ONNX Runtime execution provider ("cpu", "qnn", "xnnpack")."""
    return bool(_load(library).kibitz_has_provider(_PROVIDERS[provider]))


class Evaluator:
    def __init__(
        self,
        model: str,
        backend: str = "native",
        provider: str = "cpu",
        allow_cpu_fallback: bool = True,
        library: str | None = None,
    ):
        """provider ("cpu", "qnn" or "xnnpack") only applies to the onnxruntime backend."""
        self._lib = _load(library)
        self._handle = ctypes.c_void_p()
        opts = _Options()
        self._lib.kibitz_options_init(ctypes.byref(opts))
        opts.backend = _BACKENDS[backend]
        opts.provider = _PROVIDERS[provider]
        opts.allow_cpu_fallback = int(allow_cpu_fallback)
        status = self._lib.kibitz_open_ex(model.encode(), ctypes.byref(opts), ctypes.byref(self._handle))
        self._check(status)

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
