"""EXPERIMENT 2 (client side): measure round-trip latency to one or more inference servers.
Run from a machine on the 5G network (e.g. a laptop on the camera's 5G link):

  python latency_client.py --image frame.jpg --url MEC=http://10.0.0.5:8000 --url CLOUD=http://34.x.x.x:8000 --n 100
Reports mean / median / p95 (95% of requests were faster than this) and the server compute time.
"""
import argparse
import statistics as st
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--url", action="append", required=True, help="NAME=http://host:port (repeat)")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    data = open(a.image, "rb").read()

    for item in a.url:
        name, base = item.split("=", 1)
        rtt, srv = [], []
        s = requests.Session()
        s.post(base + "/infer", data=data, timeout=60)      # untimed warm-up
        for i in range(a.n):
            t = time.perf_counter()
            r = s.post(base + "/infer", data=data, timeout=10).json()
            rtt.append((time.perf_counter() - t) * 1000)
            srv.append(r["server_ms"])
        rtt.sort()
        print(f"{name:8s} round-trip ms: mean={st.mean(rtt):.1f} median={st.median(rtt):.1f} "
              f"p95={rtt[int(0.95 * len(rtt)) - 1]:.1f} | server compute mean={st.mean(srv):.1f} "
              f"| network share ~{st.mean(rtt) - st.mean(srv):.1f} ms | count={r['count']}")


if __name__ == "__main__":
    main()
