

from __future__ import annotations

import json
import os
from pathlib import Path

from cellpilot import tools
from cellpilot.schema import CultureRun


MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

SYSTEM_PROMPT = (
    "You are CellPilot, a cell-culture process assistant for bioprocess scientists. "
    "You analyze a fed-batch run and give clear, actionable guidance. RULES: "
    "(1) Every quantitative claim (VCD, metabolite levels, feed volumes, projected "
    "improvement) MUST come from a tool result — never estimate numbers yourself. "
    "(2) Always calibrate/diagnose before recommending. (3) Explain the mechanistic "
    "reason for each recommendation (e.g. why a glutamine feed helps). Be concise."
)

TOOL_SPECS = [
    {
        "name": "query_run",
        "description": "Summarize the observed run: duration, peak VCD, final viability, end metabolites.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "diagnose_state",
        "description": "Flag metabolic problems (glucose/glutamine depletion, lactate/ammonia accumulation) and when they occur.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "predict_trajectory",
        "description": "Calibrate the virtual cell model to the run and predict the VCD trajectory; optionally extend the horizon.",
        "input_schema": {
            "type": "object",
            "properties": {"extend_h": {"type": "number", "description": "Hours to extend beyond the last measurement."}},
        },
    },
    {
        "name": "recommend_feed",
        "description": "Recommend a feed schedule maximizing integral viable cell density under lactate/ammonia limits; reports projected improvement vs no-feed baseline.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "propose_next_experiment",
        "description": "Active learning: propose the next experiment (initial glucose/glutamine + feed design) to run to find the best process fastest, via Bayesian optimization over the calibrated virtual cell model.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _dispatch(name: str, args: dict, run: CultureRun, fit_cache: dict) -> dict:
    """Route a tool call to the corresponding pure function, caching the model fit."""
    if "fit" not in fit_cache and name in {"predict_trajectory", "recommend_feed", "propose_next_experiment"}:
        fit_cache["fit"] = tools.calibrate(run)
    fit = fit_cache.get("fit")

    if name == "query_run":
        return tools.query_run(run)
    if name == "diagnose_state":
        return tools.diagnose_state(run)
    if name == "predict_trajectory":
        return tools.predict_trajectory(run, extend_h=float(args.get("extend_h", 0.0)), fit=fit)
    if name == "recommend_feed":
        return tools.recommend_feed(run, fit=fit)
    if name == "propose_next_experiment":
        return tools.propose_next_experiment(run, fit=fit)
    raise ValueError(f"unknown tool: {name}")


def _compact(result: dict) -> dict:
    """Trim long arrays before sending tool results back to the model."""
    out = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > 12:
            out[k] = {"n": len(v), "first": v[:3], "last": v[-3:]}
        else:
            out[k] = v
    return out


def analyze_run(run: CultureRun, question: str | None = None, model: str = MODEL, max_turns: int = 8) -> str:
    """Run the agent loop over a culture run and return its final narrative answer."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("install the 'agent' extra (anthropic) and set AI_GATEWAY_API_KEY or ANTHROPIC_API_KEY") from e

    
    gateway_key = os.getenv("AI_GATEWAY_API_KEY")
    if gateway_key:
        client = anthropic.Anthropic(api_key=gateway_key, base_url="https://ai-gateway.vercel.sh")
    else:
        client = anthropic.Anthropic()
    user_msg = question or (
        f"Analyze run '{run.run_id}'. Diagnose any problems and recommend a feed "
        "strategy to maximize viable cell density. Explain your reasoning."
    )
    messages = [{"role": "user", "content": user_msg}]
    fit_cache: dict = {}

    for _ in range(max_turns):
        resp = client.messages.create(
            model=model, max_tokens=1500, system=SYSTEM_PROMPT, tools=TOOL_SPECS, messages=messages
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                result = _dispatch(block.name, block.input or {}, run, fit_cache)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(_compact(result)),
                    }
                )
        messages.append({"role": "user", "content": results})

    return "Reached max turns without a final answer."


def analyze_run_cached(run: CultureRun, cache_dir: str | Path, **kwargs) -> tuple[str, bool]:
    """analyze_run memoized to disk by run_id. Returns (answer, was_cached).

    The run_id (e.g. SYNTH-7, IEKS-01) is the key, so each seed/dataset/run calls the API
    once. ponytail: uploads all share run_id "upload" — delete the file to refresh those.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{run.run_id}.md"
    if f.exists():
        return f.read_text(), True
    answer = analyze_run(run, **kwargs)
    f.write_text(answer)
    return answer, False
