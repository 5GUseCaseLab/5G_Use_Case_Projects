"""
bench.py -- measure Tier 0 inference latency. Run THIS ON THE UNO Q, not on your laptop:
a laptop number says nothing about the board.

  pip install onnxruntime numpy
  python bench.py tier0.onnx --runs 30 [--threads 4]

Reports model-only time (preprocessing in unoq_service.py adds a few ms on top).
Random weights and random input are fine here: latency doesn't depend on the values.
"""
import argparse
import time

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    args = ap.parse_args()

    opts = ort.SessionOptions()
    if args.threads:
        opts.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, sess_options=opts, providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape
    x = np.random.randn(*shape).astype(np.float32)

    for _ in range(3):  # warm-up: first runs are slower
        sess.run(None, {"image": x})
    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        sess.run(None, {"image": x})
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)
    print(f"input {shape} | {args.runs} runs | mean {times.mean():.0f} ms | median {np.median(times):.0f} ms | "
          f"p95 {np.percentile(times, 95):.0f} ms | threads {args.threads or 'default'}")


if __name__ == "__main__":
    main()
