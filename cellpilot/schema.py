"""Unified data model for cell-culture runs."""



from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class Variable(str, Enum):


    VCD = "vcd"            
    VIABILITY = "viability"  
    GLUCOSE = "glucose"      
    GLUTAMINE = "glutamine"  
    LACTATE = "lactate"      
    AMMONIA = "ammonia"      
    TITER = "titer"          
    VOLUME = "volume"        



UNITS: dict[Variable, str] = {
    Variable.VCD: "1e6 cells/mL",
    Variable.VIABILITY: "fraction",
    Variable.GLUCOSE: "mM",
    Variable.GLUTAMINE: "mM",
    Variable.LACTATE: "mM",
    Variable.AMMONIA: "mM",
    Variable.TITER: "g/L",
    Variable.VOLUME: "mL",
}


class Measurement(BaseModel):
    

    time_h: float = Field(ge=0, description="Hours since inoculation.")
    variable: Variable
    value: float

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("measurement value must be finite")
        return v


class CultureRun(BaseModel):
    

    run_id: str
    measurements: list[Measurement] = Field(default_factory=list)
    cell_line: str | None = None
    notes: str | None = None

    def to_frame(self) -> pd.DataFrame:
        
        rows = [
            {
                "run_id": self.run_id,
                "time_h": m.time_h,
                "variable": m.variable.value,
                "value": m.value,
            }
            for m in self.measurements
        ]
        return pd.DataFrame(rows, columns=["run_id", "time_h", "variable", "value"])

    def pivot(self) -> pd.DataFrame:
        
        long = self.to_frame()
        if long.empty:
            return pd.DataFrame()
        wide = (
            long.pivot_table(index="time_h", columns="variable", values="value", aggfunc="mean")
            .sort_index()
        )
        wide.columns.name = None
        return wide

    def times(self) -> np.ndarray:
        
        return np.array(sorted({m.time_h for m in self.measurements}))

    def series(self, variable: Variable) -> tuple[np.ndarray, np.ndarray]:
        
        pairs = sorted(
            (m.time_h, m.value) for m in self.measurements if m.variable == variable
        )
        if not pairs:
            return np.array([]), np.array([])
        t, y = zip(*pairs)
        return np.array(t), np.array(y)
