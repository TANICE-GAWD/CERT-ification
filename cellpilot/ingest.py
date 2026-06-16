

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


from cellpilot.schema import CultureRun, Measurement, Variable


_SYNONYMS: dict[Variable, list[str]] = {
    Variable.VCD: ["vcd", "viable cell density", "viable cells", "xv", "vcc", "live cells"],
    Variable.VIABILITY: ["viability", "viab", "percent viable", "%viable", "via"],
    Variable.GLUCOSE: ["glucose", "glc", "gluc"],
    Variable.GLUTAMINE: ["glutamine", "gln", "q"],
    Variable.LACTATE: ["lactate", "lac", "lactic acid"],
    Variable.AMMONIA: ["ammonia", "ammonium", "amm", "nh3", "nh4"],
    Variable.TITER: ["titer", "titre", "product", "mab", "igg"],
    Variable.VOLUME: ["volume", "vol", "working volume"],
}

_TIME_SYNONYMS = ["time", "time_h", "hour", "hours", "hr", "elapsed", "day", "culture time"]


def _norm(s: str) -> str:
    """Lowercase, strip units in parentheses/brackets, collapse non-alphanumerics."""
    s = s.lower().strip()
    s = re.sub(r"[\(\[].*?[\)\]]", " ", s)        
    s = re.sub(r"[^a-z0-9% ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _match_variable(header: str) -> Variable | None:
    """Best-effort heuristic match of one header to a canonical Variable."""
    h = _norm(header)
    if not h:
        return None
    for var, syns in _SYNONYMS.items():
        for syn in syns:
            if h == syn or h.startswith(syn + " ") or h.endswith(" " + syn) or syn in h.split():
                return var
    return None


def _find_time_column(columns: list[str]) -> str | None:
    for col in columns:
        h = _norm(col)
        if any(h == s or s in h.split() or h.startswith(s) for s in _TIME_SYNONYMS):
            return col
    return None


def map_columns(columns: list[str]) -> tuple[str | None, dict[str, Variable], list[str]]:
    """Heuristically map headers. Returns (time_column, {column: Variable}, unresolved)."""
    time_col = _find_time_column(columns)
    mapping: dict[str, Variable] = {}
    unresolved: list[str] = []
    for col in columns:
        if col == time_col:
            continue
        var = _match_variable(col)
        if var is None:
            unresolved.append(col)
        else:
            mapping[col] = var
    return time_col, mapping, unresolved


def map_columns_llm(columns: list[str], unresolved: list[str], model: str = "claude-opus-4-8") -> dict[str, Variable]:
    """LLM fallback for headers the heuristic could not resolve.

    Lazy-imports the Anthropic SDK so the core package has no hard dependency on it.
    Returns a {column: Variable} mapping for whatever the model could confidently
    assign; unknown columns are simply omitted. Requires ANTHROPIC_API_KEY.
    """
    if not unresolved:
        return {}
    try:
        import json

        import anthropic
    except ImportError as e:  
        raise RuntimeError("install the 'agent' extra (anthropic) to use the LLM mapper") from e

    allowed = [v.value for v in Variable]
    prompt = (
        "You map messy lab-data column headers to a fixed vocabulary of cell-culture "
        "variables. Return ONLY a JSON object mapping each input header to one of "
        f"{allowed}, or to null if none fits.\n"
        f"All headers in this file: {columns}\n"
        f"Headers to classify: {unresolved}"
    )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=512, messages=[{"role": "user", "content": prompt}]
    )
    text = resp.content[0].text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = json.loads(match.group(0)) if match else {}
    out: dict[str, Variable] = {}
    for col, name in raw.items():
        if name in allowed and col in unresolved:
            out[col] = Variable(name)
    return out


def _to_hours(values: pd.Series, time_col: str) -> pd.Series:
    """Convert a time column to hours; if header says 'day', scale by 24."""
    if "day" in _norm(time_col):
        return values.astype(float) * 24.0
    return values.astype(float)


def ingest_dataframe(
    df: pd.DataFrame, run_id: str, *, use_llm: bool = False, cell_line: str | None = None
) -> CultureRun:
    """Normalize an arbitrary wide DataFrame into a :class:`CultureRun`."""
    columns = list(df.columns.astype(str))
    df = df.copy()
    df.columns = columns

    time_col, mapping, unresolved = map_columns(columns)
    if use_llm and unresolved:
        mapping.update(map_columns_llm(columns, unresolved))
    if time_col is None:
        raise ValueError(f"could not identify a time column among {columns}")

    hours = _to_hours(df[time_col], time_col)
    measurements: list[Measurement] = []
    for col, var in mapping.items():
        series = pd.to_numeric(df[col], errors="coerce")
        
        if var is Variable.VIABILITY and series.dropna().max() is not None and series.dropna().max() > 1.5:
            series = series / 100.0
        for t, v in zip(hours, series):
            if pd.notna(t) and pd.notna(v):
                measurements.append(Measurement(time_h=float(t), variable=var, value=float(v)))

    return CultureRun(run_id=run_id, measurements=measurements, cell_line=cell_line)


def ingest_csv(path: str | Path, run_id: str | None = None, *, use_llm: bool = False) -> CultureRun:
    """Load a CSV/Excel file and normalize it. ``run_id`` defaults to the file stem."""
    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return ingest_dataframe(df, run_id=run_id or path.stem, use_llm=use_llm)
