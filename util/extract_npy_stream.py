# -*- coding: utf-8 -*-
"""
Streaming extraction of numpy object-array caches (e.g. the 27 GB train.npy).

Problem: np.load() unpickles the WHOLE object array -> peak memory ~= file size.
Solution: a custom Unpickler that intercepts every sample dict as soon as it is
built (load_setitems) and writes it out immediately (compact npz shards),
replacing it on the pickle stack with None. Peak memory = one sample (~10 MB).

Usage:
    python util/extract_npy_stream.py --src Data2/cache/train.npy \
        --out cache_sda_real --max_samples 1000 --shard 50 --resize 256
"""
import os, sys, io, argparse, pickle
import numpy as np
import torch
import torch.nn.functional as F


class _Sink:
    """Collects samples into compact shards on disk."""
    def __init__(self, out_dir, shard=50, resize=0, max_samples=int(1e9)):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.shard, self.resize = out_dir, shard, resize
        self.max_samples = max_samples
        self.buf = []            # list of (X, y, yg, lead, id)
        self.shard_idx = 0
        self.total = 0
        self.done = False

    def _dump(self):
        if not self.buf:
            return
        X = np.stack([b[0] for b in self.buf]).astype(np.float32)
        y = np.stack([b[1] for b in self.buf]).astype(np.float32)
        yg = np.stack([b[2] for b in self.buf]).astype(np.float32)
        lead = np.asarray([b[3] for b in self.buf], np.float32)
        ids = np.asarray([b[4] for b in self.buf], np.int64)
        path = os.path.join(self.out_dir, f"part_{self.shard_idx:04d}.npz")
        np.savez(path, X=X, y=y, yg=yg, lead=lead, ids=ids)
        print(f"  shard {self.shard_idx}: {len(self.buf)} samples -> {path} "
              f"({os.path.getsize(path)/1e6:.0f} MB)", flush=True)
        self.shard_idx += 1
        self.buf = []

    def add(self, sample):
        """sample: dict with 'input'/'target' (from the cached dataset)."""
        try:
            inp, tgt = sample["input"], sample["target"]
            x = np.concatenate([inp["sparse"], inp["valid_mask"],
                                inp["era5"], inp["gtsm"]], axis=1)     # [T,6,H,W]
            y, yg = tgt["sparse"], tgt["gtsm"]
            if self.resize and x.shape[-1] != self.resize:
                # x: [T,C,H,W] -> [1, T*C, H, W] bilinear -> back to [T,C,h,w]
                T_, C_, H, W = x.shape
                Tc = T_ * C_
                xr = F.interpolate(torch.from_numpy(x).reshape(1, Tc, H, W),
                                   size=(self.resize, self.resize), mode="bilinear")
                x = xr[0].reshape(C_, T_, self.resize, self.resize).permute(1, 0, 2, 3).numpy()
                # y / yg: [H,W] -> [1,1,H,W] nearest (keep sparse pixels)
                y = F.interpolate(torch.from_numpy(np.asarray(y, np.float32))[None],
                                  size=(self.resize, self.resize), mode="nearest")[0].numpy()
                yg = F.interpolate(torch.from_numpy(np.asarray(yg, np.float32))[None],
                                   size=(self.resize, self.resize), mode="nearest")[0].numpy()
            lead = float(np.asarray(inp["td_lead"]).reshape(-1)[0])
            gid = int(np.asarray(tgt["id"]).reshape(-1)[0])
            self.buf.append((x, y, yg, lead, gid))
            self.total += 1
            if len(self.buf) >= self.shard:
                self._dump()
            if self.total >= self.max_samples:
                self.done = True
        except Exception as e:
            print("  skip sample:", type(e).__name__, str(e)[:80], flush=True)


class StreamingUnpickler(pickle._Unpickler):
    """Unpickles dict {'dataset': [sample, ...]} while writing each sample to disk."""
    dispatch = dict(pickle._Unpickler.dispatch)  # subclassing must re-register opcodes

    def __init__(self, file, sink):
        super().__init__(file)
        self.sink = sink

    def load_setitems(self):
        # build the dict exactly like the base class (pickle._Unpickler.load_setitems)
        items = self.pop_mark()
        d = self.stack[-1]
        for i in range(0, len(items), 2):
            d[items[i]] = items[i + 1]
        # intercept: if this is a dataset sample, stream it out and drop it
        if isinstance(d, dict) and "input" in d and "target" in d:
            if self.sink.done:
                raise StopIteration("reached max_samples")
            self.sink.add(d)
            self.stack[-1] = None  # placeholder, keeps pickle stack semantics
    dispatch[ord('u')] = load_setitems


class NumpyStreamingUnpickler(StreamingUnpickler):
    """Handles the OUTER 0-d object ndarray: its BUILD state carries the pickled
    dataset dict as raw bytes; we stream-unpickle those bytes instead of letting
    numpy materialise the whole 27 GB object. Inherits load_setitems (samples)."""
    dispatch = dict(StreamingUnpickler.dispatch)

    def __init__(self, file, sink):
        super().__init__(file, sink)

    def load_build(self):
        state = self.stack[-1]
        inst = self.stack[-2]
        raw = None
        if isinstance(inst, np.ndarray) and inst.dtype == object and inst.ndim == 0                 and isinstance(state, tuple) and len(state) == 4                 and isinstance(state[1], np.dtype) and state[1] == object:
            raw = state[3]
        if isinstance(raw, (bytes, bytearray)) and len(raw) > 100:
            # stream-unpickle the nested dataset dict, then drop the array
            inner = StreamingUnpickler(io.BytesIO(bytes(raw)), self.sink)
            try:
                inner.load()
            except StopIteration:
                pass
            self.stack[-2] = None
            self.stack.pop()
            return
        super().load_build()
    dispatch[ord('b')] = load_build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="cache_sda_real")
    ap.add_argument("--max_samples", type=int, default=int(1e9))
    ap.add_argument("--shard", type=int, default=50)
    ap.add_argument("--resize", type=int, default=0)
    args = ap.parse_args()

    sink = _Sink(args.out, shard=args.shard, resize=args.resize, max_samples=args.max_samples)
    with open(args.src, "rb") as f:
        # skip the numpy header (v1.0 .npy: magic(6) + version(2) + hdrlen(2) + header)
        import struct
        f.read(6); f.read(2)
        hdr_len = struct.unpack("<H", f.read(2))[0]
        f.read(hdr_len)
        up = NumpyStreamingUnpickler(f, sink)
        try:
            up.load()
        except StopIteration:
            print("stopped early at max_samples", flush=True)
        except Exception as e:
            print("unpickle finished/stopped:", type(e).__name__, str(e)[:100], flush=True)
    sink._dump()  # flush remainder
    print(f"DONE. {sink.total} samples streamed -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
