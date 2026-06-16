from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from cellpilot.schema import Variable


IDX = {"Xv": 0, "Xd": 1, "Glc": 2, "Gln": 3, "Lac": 4, "Amm": 5, "Titer": 6, "V": 7}


_STATE_TO_VAR = {
    "Xv": Variable.VCD,
    "Glc": Variable.GLUCOSE,
    "Gln": Variable.GLUTAMINE,
    "Lac": Variable.LACTATE,
    "Amm": Variable.AMMONIA,
    "Titer": Variable.TITER,
    "V": Variable.VOLUME,
}


@dataclass
class ModelParams:
    

    mu_max: float = 0.042      
    K_glc: float = 0.5         
    K_gln: float = 0.2         
    KI_lac: float = 200.0      
    KI_amm: float = 15.0       

    mu_d_min: float = 0.0012   
    kd_lac: float = 0.00025    
    kd_amm: float = 0.0016     

    
    Y_x_glc: float = 2.2       
    Y_x_gln: float = 6.5       
    m_glc: float = 0.004       
    m_gln: float = 0.001       

    Y_lac_glc: float = 1.4     
    Y_amm_gln: float = 0.6     
    kd_gln_chem: float = 0.002  

    q_titer: float = 2.5e-4    

    
    
    
    
    
    glc_switch_mM: float = 3.0   
    k_lac_uptake: float = 0.0    
    Y_x_lac: float = 0.0         


@dataclass
class FeedEvent:


    time_h: float
    volume_ml: float
    glucose_mM: float = 0.0
    glutamine_mM: float = 0.0


@dataclass
class InitialState:

    Xv: float = 0.3
    Glc: float = 25.0
    Gln: float = 5.0
    Lac: float = 0.0
    Amm: float = 0.5
    Titer: float = 0.0
    V: float = 50.0
    Xd: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array(
            [self.Xv, self.Xd, self.Glc, self.Gln, self.Lac, self.Amm, self.Titer, self.V],
            dtype=float,
        )


def specific_growth_rate(glc: float, gln: float, lac: float, amm: float, p: ModelParams) -> float:
    
    glc = max(glc, 0.0)
    gln = max(gln, 0.0)
    lac = max(lac, 0.0)
    amm = max(amm, 0.0)
    limitation = (glc / (p.K_glc + glc)) * (gln / (p.K_gln + gln))
    inhibition = (p.KI_lac / (p.KI_lac + lac)) * (p.KI_amm / (p.KI_amm + amm))
    return p.mu_max * limitation * inhibition


def specific_death_rate(lac: float, amm: float, p: ModelParams) -> float:
    return p.mu_d_min + p.kd_lac * max(lac, 0.0) + p.kd_amm * max(amm, 0.0)


def _rhs(t: float, y: np.ndarray, p: ModelParams) -> np.ndarray:
    
    Xv, Xd, Glc, Gln, Lac, Amm, Titer, V = y
    Xv = max(Xv, 0.0)

    mu = specific_growth_rate(Glc, Gln, Lac, Amm, p)
    mu_d = specific_death_rate(Lac, Amm, p)

    
    q_glc = mu / p.Y_x_glc + p.m_glc
    q_gln = mu / p.Y_x_gln + p.m_gln

    
    lac_uptake = 0.0
    if p.k_lac_uptake > 0.0 and Glc < p.glc_switch_mM and Lac > 0.0:
        lac_uptake = p.k_lac_uptake * (Lac / (2.0 + Lac)) * Xv

    dXv = (mu - mu_d) * Xv + p.Y_x_lac * lac_uptake
    dXd = mu_d * Xv
    dGlc = -q_glc * Xv
    dGln = -q_gln * Xv - p.kd_gln_chem * Gln
    dLac = p.Y_lac_glc * q_glc * Xv - lac_uptake
    dAmm = p.Y_amm_gln * q_gln * Xv + p.kd_gln_chem * Gln
    dTiter = p.q_titer * Xv
    dV = 0.0
    return np.array([dXv, dXd, dGlc, dGln, dLac, dAmm, dTiter, dV])


def _apply_feed(y: np.ndarray, feed: FeedEvent) -> np.ndarray:
    
    y = y.copy()
    V = y[IDX["V"]]
    Vnew = V + feed.volume_ml
    if Vnew <= 0:
        return y
    dil = V / Vnew

    
    for k in ("Xv", "Xd", "Lac", "Amm", "Titer"):
        y[IDX[k]] *= dil
    
    y[IDX["Glc"]] = (y[IDX["Glc"]] * V + feed.glucose_mM * feed.volume_ml) / Vnew
    y[IDX["Gln"]] = (y[IDX["Gln"]] * V + feed.glutamine_mM * feed.volume_ml) / Vnew
    y[IDX["V"]] = Vnew
    return y


def simulate(
    initial: InitialState,
    t_end: float,
    params: ModelParams | None = None,
    feeds: list[FeedEvent] | None = None,
    dt: float = 1.0,
) -> pd.DataFrame:

    params = params or ModelParams()
    feeds = sorted(feeds or [], key=lambda f: f.time_h)

    
    breakpoints = sorted({0.0, t_end, *(f.time_h for f in feeds if 0.0 < f.time_h < t_end)})
    y = initial.vector()
    frames: list[pd.DataFrame] = []

    for seg_start, seg_end in zip(breakpoints[:-1], breakpoints[1:]):
        
        for f in feeds:
            if np.isclose(f.time_h, seg_start) and seg_start > 0.0:
                y = _apply_feed(y, f)

        t_eval = np.arange(seg_start, seg_end + 1e-9, dt)
        if t_eval[-1] < seg_end:
            t_eval = np.append(t_eval, seg_end)
        sol = solve_ivp(
            _rhs, (seg_start, seg_end), y, t_eval=t_eval, args=(params,),
            method="LSODA", rtol=1e-6, atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"integration failed on [{seg_start}, {seg_end}]: {sol.message}")

        seg = pd.DataFrame(sol.y.T, columns=list(IDX.keys()))
        seg["time_h"] = sol.t
        frames.append(seg)
        y = sol.y[:, -1]

    
    
    traj = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("time_h", keep="last")
        .set_index("time_h")
    )
    traj[list(IDX.keys())] = traj[list(IDX.keys())].clip(lower=0.0)

    
    out = pd.DataFrame(index=traj.index)
    for state, var in _STATE_TO_VAR.items():
        out[var.value] = traj[state]
    total = traj["Xv"] + traj["Xd"]
    out[Variable.VIABILITY.value] = np.where(total > 0, traj["Xv"] / total, 1.0)
    out.index.name = "time_h"
    return out


def integral_vcd(traj: pd.DataFrame) -> float:

    t = traj.index.to_numpy()
    vcd = traj[Variable.VCD.value].to_numpy()
    return float(np.trapezoid(vcd, t))
