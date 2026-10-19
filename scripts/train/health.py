#!/usr/bin/env python3
"""Health of a training run from its (offline) wandb files, per block of steps: medians of the loss, x_mse, the gradient norm and the scale of the
encoder streams (rms of a, s, z), and warnings for the signs of a diverging run seen in earlier runs: a gradient norm that grew a lot over
its best, a worse x_mse than the best block, streams that grew by orders of magnitude, a loss that stays constant (collapsed network).

    python scripts/train/health.py logs/train/<run> [--block 2000] [--last 6]
"""
import argparse
import glob
import json
import statistics as st
import struct
import sys


def history(run: str):
    """History rows of every offline wandb run file of `run` (.wandb is a LevelDB-style log: 7-byte header, 32 KiB blocks of [crc 4, len 2, type 1] chunks)"""
    from wandb.proto import wandb_internal_pb2 as pb
    rows = []
    for f in sorted(glob.glob(run + "/wandb/offline-run-*/run-*.wandb")):
        data = open(f, "rb").read()[7:]
        pos, buf = 0, b""
        while pos + 7 <= len(data):
            if (pos + 7) % 32768 > 32768 - 7:
                pos += 32768 - (pos + 7) % 32768
                continue
            _, length, typ = struct.unpack("<IHB", data[pos:pos + 7])
            chunk = data[pos + 7:pos + 7 + length]
            pos += 7 + length
            if typ in (1, 2):
                buf = chunk
            elif typ in (3, 4):
                buf += chunk
            else:
                continue
            if typ in (1, 4):
                r = pb.Record()
                try:
                    r.ParseFromString(buf)
                except Exception:       # a record cut by a crash
                    continue
                if r.HasField("history"):
                    rows.append({i.key or ".".join(i.nested_key): json.loads(i.value_json) for i in r.history.item if not i.key.startswith("grad/")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--last", type=int, default=6, help="number of blocks printed")
    a = ap.parse_args()
    every = history(a.run)                      # the gradient statistics are logged in rows of their own
    rows = [r for r in every if "train/loss" in r]
    if not rows:
        sys.exit("no training history found in " + a.run)
    step = lambda r: r.get("trainer/global_step", r.get("_step"))
    last = step(rows[-1])
    med = lambda rs, k: st.median([r[k] for r in rs if k in r]) if any(k in r for r in rs) else float("nan")
    blocks = []
    for lo in range(0, last + 1, a.block):
        rs = [r for r in rows if lo <= step(r) < lo + a.block]
        gs = [r for r in every if lo <= step(r) < lo + a.block]
        if len(rs) >= 5:
            blocks.append((lo, med(rs, "train/loss"), med(rs, "train/x_mse"), med(rs, "train/smooth_lddt"), med(gs, "train/grad_norm"),
                           med(rs, "train/rms_a"), med(rs, "train/rms_s"), med(rs, "train/rms_z"), med(rs, "train/lr")))
    print(f"{a.run}: step {last}, skipped updates {rows[-1].get('train/skipped_steps', 0):.0f}")
    print(f"{'steps':>13s} {'loss':>7s} {'x_mse':>7s} {'lddt':>6s} {'gnorm':>9s} {'rms_a':>8s} {'rms_s':>10s} {'rms_z':>8s} {'lr':>9s}")
    for b in blocks[-a.last:]:
        print(f"{b[0]:6d}-{b[0] + a.block:6d} {b[1]:7.2f} {b[2]:7.3f} {b[3]:6.3f} {b[4]:9.2f} {b[5]:8.2f} {b[6]:10.2f} {b[7]:8.2f} {b[8]:9.2e}")
    warn = []
    if len(blocks) >= 3:
        now, early = blocks[-1], blocks[1:] if len(blocks) > 1 else blocks
        best = min(early, key=lambda b: b[2])
        if now[2] > 1.5 * best[2] and now[0] > 4000:
            warn.append(f"x_mse {now[2]:.3f} is {now[2] / best[2]:.1f}x the best block ({best[2]:.3f} at step {best[0]})")
        g0 = min(b[4] for b in early if b[4] == b[4])
        if now[4] > 20 * g0:
            warn.append(f"gradient norm {now[4]:.1f} is {now[4] / g0:.0f}x its lowest block ({g0:.2f})")
        if now[6] == now[6] and blocks[1][6] == blocks[1][6] and now[6] > 100 * blocks[1][6]:
            warn.append(f"stream s has grown {now[6] / blocks[1][6]:.0f}x (rms {now[6]:.0f})")
        tail = blocks[-3:]
        if max(b[1] for b in tail) - min(b[1] for b in tail) < 0.02 and now[2] > 0.5 and now[0] > 8000:
            warn.append("loss has not moved over three blocks at a high value (collapsed network?)")
    print("WARNINGS:\n  " + "\n  ".join(warn) if warn else "no warnings")


if __name__ == "__main__":
    main()
