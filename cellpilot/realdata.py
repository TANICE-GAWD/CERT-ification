
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cellpilot.model import FeedEvent
from cellpilot.schema import CultureRun, Measurement, Variable

DEFAULT_DIR = Path("data/ieks_9batches")

_COL_TO_VAR = {
    "VCD (10^9 cells/L)": (Variable.VCD, 1.0),
    "Glucose (mM)": (Variable.GLUCOSE, 1.0),
    "Lactate (mM)": (Variable.LACTATE, 1.0),
    "Glutamine (mM)": (Variable.GLUTAMINE, 1.0),
    "Ammonium (mM)": (Variable.AMMONIA, 1.0),
    "Product (mg/L)": (Variable.TITER, 1e-3),   
    "Volume (L)": (Variable.VOLUME, 1e3),        
}


def _batch_to_run(bid: int, g: pd.DataFrame) -> CultureRun:
    measurements: list[Measurement] = []
    for _, row in g.iterrows():
        t_h = float(row["Time (day)"]) * 24.0
        for col, (var, scale) in _COL_TO_VAR.items():
            if col in g.columns and pd.notna(row[col]):
                measurements.append(
                    Measurement(time_h=t_h, variable=var, value=float(row[col]) * scale)
                )
    return CultureRun(
        run_id=f"IEKS-{bid:02d}",
        measurements=measurements,
        cell_line="mammalian (ieks-9batches, simulated, 3rd-party)",
        notes="Independent fed-batch dataset; see cellpilot/realdata.py for provenance.",
    )


def _batch_feeds(bid: int, fb: pd.DataFrame) -> list[FeedEvent]:
    feeds: list[FeedEvent] = []
    for _, row in fb[fb["Batch id"] == bid].iterrows():
        feeds.append(
            FeedEvent(
                time_h=float(row["Time (day)"]) * 24.0,
                volume_ml=float(row["Feed volume (L)"]) * 1e3,
                glucose_mM=float(row.get("Glucose (mM)", 0.0)),
                glutamine_mM=float(row.get("Glutamine (mM)", 0.0)),
            )
        )
    return feeds


def load_ieks_batches(data_dir: str | Path = DEFAULT_DIR) -> list[tuple[CultureRun, list[FeedEvent]]]:
    """Load the 9 fed-batch runs as (CultureRun, feed schedule) pairs."""
    data_dir = Path(data_dir)
    meas = pd.read_csv(data_dir / "measurements.csv")
    feed_path = data_dir / "feed_bolus.csv"
    feeds_df = pd.read_csv(feed_path) if feed_path.exists() else pd.DataFrame()

    out: list[tuple[CultureRun, list[FeedEvent]]] = []
    for bid, g in meas.groupby("Batch id"):
        run = _batch_to_run(int(bid), g.sort_values("Time (day)"))
        feeds = _batch_feeds(int(bid), feeds_df) if not feeds_df.empty else []
        out.append((run, feeds))
    return out
