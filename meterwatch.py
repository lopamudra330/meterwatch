"""
MeterWatch - anomaly detection on half-hourly smart-meter readings.

Detects the kinds of faults seen in real smart-metering networks:
  - dropout : communication loss, readings missing / reported as zero
  - flatline: meter stuck reporting the same value
  - spike   : implausible surge (faulty reading or tampering)

Compares a simple rule-based threshold against an Isolation Forest
model on daily load profiles (48 half-hour readings per day).

Usage:
  python meterwatch.py                      # synthetic demo data
  python meterwatch.py --data LCL_block.csv # London smart-meter data (LCL format)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

RNG = np.random.default_rng(42)
FAULTS = ["dropout", "flatline", "spike"]


# ---------- 1. Load data ----------
def synthetic_profiles(n_homes=40, n_days=60):
    """Realistic household load: night base, morning bump, evening peak."""
    t = np.arange(48) / 2  # hours
    shape = (0.15 + 0.25 * np.exp(-((t - 8) ** 2) / 2)
             + 0.6 * np.exp(-((t - 19) ** 2) / 4))
    rows = []
    for h in range(n_homes):
        scale = RNG.uniform(0.6, 1.6)
        for d in range(n_days):
            weekend = 1.15 if d % 7 in (5, 6) else 1.0
            day = shape * scale * weekend * RNG.normal(1, 0.08)
            day += RNG.normal(0, 0.03, 48)
            rows.append([f"MAC{h:03d}", d, *np.clip(day, 0.01, None)])
    return pd.DataFrame(rows, columns=["home", "day", *range(48)])


def load_lcl(path):
    """Read 'Smart meters in London' half-hourly CSV.

    Works with both common layouts:
      LCLid, stdorToU, DateTime, KWH/hh (per half hour)   (original LCL files)
      LCLid, tstp, energy(kWh/hh)                         (Kaggle block files)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    kwh = [c for c in df.columns if "kwh" in c.lower()][0]
    time = [c for c in df.columns if c.lower() in ("datetime", "tstp")][0]
    df[kwh] = pd.to_numeric(df[kwh], errors="coerce")
    df[time] = pd.to_datetime(df[time])
    df["day"] = df[time].dt.date
    df["slot"] = df[time].dt.hour * 2 + df[time].dt.minute // 30
    wide = df.pivot_table(index=["LCLid", "day"], columns="slot",
                          values=kwh, aggfunc="mean")
    wide = wide.reindex(columns=range(48)).dropna()  # keep complete days only
    wide = wide.reset_index().rename(columns={"LCLid": "home"})
    return wide[["home", "day", *range(48)]]


# ---------- 2. Inject labelled faults ----------
def inject_faults(df, rate=0.05):
    df = df.copy()
    df["fault"] = "normal"
    X = df[list(range(48))].to_numpy(dtype=float, copy=True)
    for i in RNG.choice(len(df), int(len(df) * rate), replace=False):
        kind = RNG.choice(FAULTS)
        start = RNG.integers(0, 36)
        span = RNG.integers(6, 12)
        if kind == "dropout":
            X[i, start:start + span] = 0.0
        elif kind == "flatline":
            X[i, start:] = X[i, start]
        else:
            X[i, start:start + 2] *= RNG.uniform(4, 8)
        df.at[i, "fault"] = kind
    df[list(range(48))] = X
    return df


# ---------- 3. Features ----------
def features(X):
    diffs = np.abs(np.diff(X, axis=1))
    return np.column_stack([
        X.mean(1), X.std(1), X.max(1),
        (X <= 0.005).sum(1),               # zero readings (dropouts)
        (diffs < 1e-6).sum(1),             # unchanged steps (flatline)
        diffs.max(1),                      # largest jump (spike)
        X.max(1) / (X.mean(1) + 1e-6),     # peak-to-average ratio
    ])


# ---------- 4. Detectors ----------
def rule_based(X):
    """Baseline: flag days with any zero reading or a very high peak."""
    high = X.max(1) > np.percentile(X.max(1), 95)
    return ((X <= 0.005).any(1) | high).astype(int)


def isolation_forest(X, contamination=0.05):
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42)
    return (model.fit_predict(features(X)) == -1).astype(int)


def score(y, pred):
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0))


# ---------- 5. Run ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="LCL half-hourly CSV (optional)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = load_lcl(args.data) if args.data else synthetic_profiles()
    df = inject_faults(df.reset_index(drop=True))
    X = df[list(range(48))].to_numpy(dtype=float)
    y = (df["fault"] != "normal").astype(int).to_numpy()

    preds = {"Rule-based threshold": rule_based(X),
             "Isolation Forest": isolation_forest(X)}
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    rows = []
    for name, p in preds.items():
        s = score(y, p)
        caught = {k: p[df["fault"] == k].mean() for k in FAULTS}
        rows.append({"model": name, **{k: round(v, 2) for k, v in s.items()},
                     **{f"caught_{k}": round(v, 2) for k, v in caught.items()}})
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False)
    print(f"\nDays analysed: {len(df)} | faulty days: {y.sum()}\n")
    print(table.to_string(index=False))

    # Plot: one example of each fault vs a normal day
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
    hours = np.arange(48) / 2
    for ax, kind in zip(axes, ["normal", *FAULTS]):
        i = df.index[df["fault"] == kind][0]
        flagged = preds["Isolation Forest"][i]
        ax.plot(hours, X[i], color="#c0392b" if flagged else "#2c7fb8")
        ax.set_title(f"{kind}  ({'flagged' if flagged else 'not flagged'})")
        ax.set_xlabel("hour of day")
    axes[0].set_ylabel("kWh per half hour")
    fig.suptitle("MeterWatch - Isolation Forest on daily smart-meter profiles")
    fig.tight_layout()
    fig.savefig(out / "fault_examples.png", dpi=120)
    print(f"\nSaved {out/'metrics.csv'} and {out/'fault_examples.png'}")


if __name__ == "__main__":
    main()
