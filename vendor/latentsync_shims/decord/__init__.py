"""Minimal decord stand-in (decord has no wheels for this interpreter): cv2 video, librosa audio."""
import cv2, numpy as np


class _Arr:
    def __init__(self, a): self._a = a
    def asnumpy(self): return self._a


class _Ctx:
    pass


def cpu(i=0): return _Ctx()
def gpu(i=0): return _Ctx()


class VideoReader:
    def __init__(self, uri, ctx=None, width=-1, height=-1, num_threads=0, fault_tol=-1):
        cap = cv2.VideoCapture(str(uri)); self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release(); self._f = np.stack(fr) if fr else np.zeros((0, 1, 1, 3), np.uint8)
    def __len__(self): return len(self._f)
    def __getitem__(self, i): return _Arr(self._f[i])
    def get_batch(self, idx): return _Arr(self._f[list(idx)])
    def get_avg_fps(self): return self._fps


class AudioReader:
    def __init__(self, uri, ctx=None, sample_rate=16000, mono=True):
        import librosa
        y, _ = librosa.load(str(uri), sr=sample_rate, mono=mono)
        self._a = y[None, :] if y.ndim == 1 else y
    def __getitem__(self, i): return _Arr(self._a[:, i] if not isinstance(i, slice) else self._a[:, i])
    def shape(self): return self._a.shape
