"""Helpers to translate natural language plot instructions into Altair charts."""
from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import altair as alt
import pandas as pd

from services import LLMConfig, LLMRequestError, call_llm_chat


@dataclass
class ChartPlan:
    chart_type: str
    x: str
    y: Optional[str] = None
    aggregate: Optional[str] = None
    color: Optional[str] = None


@dataclass
class LLMChartPlan:
    mark: str
    x_field: str
    y_field: Optional[str] = None
    x_aggregate: Optional[str] = None
    y_aggregate: Optional[str] = None
    color: Optional[str] = None
    tooltip: Optional[List[str]] = None
    thinking: str = ""
    title: Optional[str] = None
    description: Optional[str] = None


@dataclass
class LLMPlotOutcome:
    kind: str
    chart: Optional[alt.Chart] = None
    figure: Optional[plt.Figure] = None
    plan: Optional[LLMChartPlan] = None
    raw_spec: Optional[Dict[str, Any]] = None
    thinking: str = ""
    title: Optional[str] = None
    description: Optional[str] = None
    code: Optional[str] = None


CHART_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "bar": ("bar", "histogram", "column"),
    "line": ("line", "trend"),
    "area": ("area",),
    "scatter": ("scatter", "dot", "bubble"),
    "pie": ("pie", "donut", "doughnut"),
}

AGG_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "count": ("count", "number of", "how many"),
    "sum": ("sum", "total"),
    "mean": ("average", "mean"),
    "max": ("maximum", "max", "highest"),
    "min": ("minimum", "min", "lowest"),
}


def _find_first(matchers: Iterable[Tuple[str, Tuple[str, ...]]], text: str) -> Optional[str]:
    lowered = text.lower()
    for key, tokens in matchers:
        if any(token in lowered for token in tokens):
            return key
    return None


def parse_instruction(instruction: str, columns: List[str]) -> ChartPlan:
    if not instruction.strip():
        raise ValueError("Instruction is empty")

    lowered = instruction.lower()
    chart_type = _find_first(CHART_KEYWORDS.items(), lowered) or "bar"
    aggregate = _find_first(AGG_KEYWORDS.items(), lowered)

    mentioned: List[str] = []
    for col in columns:
        token = col.lower()
        if token and token in lowered:
            mentioned.append(col)

    mentioned = list(dict.fromkeys(mentioned))  # Preserve order, drop duplicates

    if aggregate == "count" and not mentioned:
        mentioned = columns[:1]

    if not mentioned and len(columns) >= 2:
        mentioned = columns[:2]
    elif not mentioned:
        raise ValueError("Could not match any column names in the instruction")

    x = mentioned[0]
    y = mentioned[1] if len(mentioned) > 1 else None

    if aggregate in {"sum", "mean", "max", "min"} and y is None:
        numeric_candidates = [col for col in columns if col != x]
        if numeric_candidates:
            y = numeric_candidates[0]
        else:
            y = x

    if aggregate == "count" and y is None:
        y = "count"

    return ChartPlan(chart_type=chart_type, x=x, y=y, aggregate=aggregate)


def build_chart(df: pd.DataFrame, plan: ChartPlan) -> alt.Chart:
    if plan.x not in df.columns:
        raise ValueError(f"Column '{plan.x}' not available in results")
    if plan.y and plan.y not in df.columns and plan.aggregate != "count":
        raise ValueError(f"Column '{plan.y}' not available in results")

    working = df.copy()

    if plan.aggregate == "count" and plan.y == "count":
        working = (
            working[plan.x]
            .astype(str)
            .value_counts(dropna=False)
            .reset_index()
            .rename(columns={"index": plan.x, plan.x: "count"})
        )
        y_field = "count"
    elif plan.aggregate in {"sum", "mean", "max", "min"} and plan.y:
        grouped = (
            working[[plan.x, plan.y]]
            .dropna()
            .groupby(plan.x, dropna=False)[plan.y]
            .agg(plan.aggregate)
            .reset_index()
        )
        y_field = f"{plan.y}_{plan.aggregate}"
        working = grouped.rename(columns={plan.y: y_field})
    else:
        y_field = plan.y or plan.x
        working = working[[plan.x, y_field]].dropna()

    chart = alt.Chart(working)

    if plan.chart_type == "bar":
        chart = chart.mark_bar().encode(x=alt.X(plan.x, sort="-y"), y=y_field)
    elif plan.chart_type == "line":
        chart = chart.mark_line(point=True).encode(x=plan.x, y=y_field)
    elif plan.chart_type == "area":
        chart = chart.mark_area().encode(x=plan.x, y=y_field)
    elif plan.chart_type == "scatter":
        chart = chart.mark_circle(size=80).encode(x=plan.x, y=y_field)
    elif plan.chart_type == "pie":
        chart = chart.mark_arc().encode(theta=alt.Theta(y_field, type="quantitative"), color=plan.x)
    else:
        raise ValueError(f"Unsupported chart type: {plan.chart_type}")

    if plan.color and plan.color in working.columns:
        chart = chart.encode(color=plan.color)

    return chart.properties(width="container", height=400)


# ---------------------------------------------------------------------------
# LLM-assisted plotting helpers
# ---------------------------------------------------------------------------


def _summarize_columns(df: pd.DataFrame, max_columns: int = 12, max_samples: int = 3) -> str:
    lines: List[str] = []
    for col in df.columns[:max_columns]:
        series = df[col]
        dtype = str(series.dtype)
        samples = series.dropna().unique()[:max_samples]
        sample_text = ", ".join(str(s) for s in samples)
        lines.append(f"- {col} ({dtype}) samples: {sample_text}")
    if len(df.columns) > max_columns:
        lines.append(f"- ... {len(df.columns) - max_columns} more columns")
    return "\n".join(lines)


def _sample_rows(df: pd.DataFrame, max_rows: int = 6) -> str:
    sample = df.head(max_rows).replace({pd.NA: None}).to_dict(orient="records")
    return json.dumps(sample, ensure_ascii=True)





THAI_FONT_FILES = [
    Path(__file__).with_name("fonts") / "NotoSansThai-Regular.ttf",
    Path(__file__).with_name("fonts") / "THSarabunNew.ttf",
]

THAI_FONT_FAMILIES = [
    "Noto Sans Thai",
    "TH Sarabun New",
    "Sarabun",
    "Leelawadee UI",
    "Tahoma",
]

_THAI_FONT_CONFIGURED = False


def _ensure_thai_font() -> None:
    global _THAI_FONT_CONFIGURED
    if _THAI_FONT_CONFIGURED:
        return
    available = {f.name for f in fm.fontManager.ttflist}
    fonts_to_add = []
    for font_path in THAI_FONT_FILES:
        if font_path.exists():
            try:
                prop = fm.FontProperties(fname=str(font_path))
            except Exception:
                continue
            name = prop.get_name()
            if name not in available:
                fonts_to_add.append((font_path, name))
            available.add(name)
    for font_path, _ in fonts_to_add:
        try:
            fm.fontManager.addfont(str(font_path))
        except Exception:
            pass
    available = {f.name for f in fm.fontManager.ttflist}
    for family in THAI_FONT_FAMILIES:
        if family in available:
            plt.rcParams.update({
                "font.family": family,
                "axes.unicode_minus": False,
            })
            _THAI_FONT_CONFIGURED = True
            return
    _THAI_FONT_CONFIGURED = True


CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _build_matplotlib_code_prompt(columns: List[str], instruction: str) -> str:
    column_list = ', '.join(columns) if columns else 'no columns detected'
    prompt = textwrap.dedent(
        f"""Given a pandas DataFrame named `data` (alias `df`) with columns:
{column_list}

Write Python code using pandas and matplotlib.pyplot (imported as `plt`) to create a single visualization that answers:
"{instruction.strip()}".

Rules
-----
1. Do not include import statements; assume pandas as `pd`, numpy as `np`, and matplotlib.pyplot as `plt` are already available.
2. Use pandas for data preparation and matplotlib for plotting (prefer `plt.subplots`).
3. Handle missing values with `dropna` or `fillna` before aggregations or plotting.
4. Only reference fields that exist in the provided columns or summary.
5. Produce exactly one relevant plot and avoid tabular/text output.
6. Include axis labels and a concise, descriptive title.
7. Assign the matplotlib Figure (or Axes) you want to display to a variable named `result`.
8. Call `plt.tight_layout()` before returning `result` and do not call `plt.show()`.
9. Return only valid Python code inside a single fenced block labelled ```python, with no commentary.
"""
    ).strip()
    return prompt







def _sanitize_plot_code(code: str) -> str:
    cleaned_lines: List[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if stripped.startswith("plt.show"):
            continue
        cleaned_lines.append(line)
    sanitized = "\n".join(cleaned_lines).strip()
    return sanitized



def _extract_first_code_block(text: str) -> str:
    if not text:
        return ""
    match = CODE_BLOCK_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json_dict(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                cleaned = part
                break
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain JSON")
    candidate = cleaned[start : end + 1]
    return json.loads(candidate)


def _normalize_aggregate(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"average", "avg"}:
        return "mean"
    if lowered in {"count_distinct", "distinct", "nunique"}:
        return "count_distinct"
    return lowered


def _apply_filters(df: pd.DataFrame, filters: Any) -> pd.DataFrame:
    if not filters:
        return df
    if isinstance(filters, dict):
        filters = [filters]
    if not isinstance(filters, list):
        return df
    filtered = df
    for rule in filters:
        if not isinstance(rule, dict):
            continue
        field = rule.get("field")
        if not field or field not in filtered.columns:
            continue
        op = (rule.get("op") or "==").lower()
        value = rule.get("value")
        series = filtered[field]
        try:
            if op in {"==", "eq"}:
                filtered = filtered[series == value]
            elif op in {"!=", "ne"}:
                filtered = filtered[series != value]
            elif op in {">", "gt"}:
                filtered = filtered[pd.to_numeric(series, errors="coerce") > float(value)]
            elif op in {">=", "gte"}:
                filtered = filtered[pd.to_numeric(series, errors="coerce") >= float(value)]
            elif op in {"<", "lt"}:
                filtered = filtered[pd.to_numeric(series, errors="coerce") < float(value)]
            elif op in {"<=", "lte"}:
                filtered = filtered[pd.to_numeric(series, errors="coerce") <= float(value)]
            elif op == "in" and isinstance(value, list):
                filtered = filtered[series.isin(value)]
        except (TypeError, ValueError):
            continue
    return filtered


def _altair_type(value: Optional[str], *, default: str) -> str:
    if not value:
        return default
    lowered = value.lower()
    mapping = {
        "nominal": "nominal",
        "ordinal": "ordinal",
        "quantitative": "quantitative",
        "temporal": "temporal",
        "time": "temporal",
        "date": "temporal",
        "datetime": "temporal",
    }
    return mapping.get(lowered, default)


def _prepare_chart_dataframe(
    df: pd.DataFrame,
    plan: LLMChartPlan,
) -> Tuple[pd.DataFrame, str, str]:
    working = df.copy()

    agg_x = _normalize_aggregate(plan.x_aggregate)
    agg_y = _normalize_aggregate(plan.y_aggregate)

    def _ensure_column(field: str) -> None:
        if field not in working.columns:
            raise ValueError(f"Column '{field}' not available in results")

    if agg_x:
        if not plan.y_field:
            raise ValueError('LLM plan missing categorical field for aggregated x-axis')
        _ensure_column(plan.y_field)
        base_cols = [plan.y_field]
        if plan.x_field in working.columns:
            base_cols.append(plan.x_field)
        base = working[base_cols].dropna(subset=[plan.y_field])
        if agg_x == 'count':
            grouped = base.groupby(plan.y_field, dropna=False).size().reset_index(name='count')
            measure_field = 'count'
        elif agg_x == 'count_distinct':
            if plan.x_field not in working.columns:
                raise ValueError(f"Column '{plan.x_field}' not available for count_distinct aggregation")
            measure_field = f"{plan.x_field}_distinct"
            grouped = (
                base.groupby(plan.y_field, dropna=False)[plan.x_field]
                .nunique()
                .reset_index(name=measure_field)
            )
        else:
            if plan.x_field not in working.columns:
                raise ValueError(f"Column '{plan.x_field}' not available for aggregation")
            measure_field = f"{plan.x_field}_{agg_x}"
            grouped = (
                base.groupby(plan.y_field, dropna=False)[plan.x_field]
                .agg(agg_x)
                .reset_index(name=measure_field)
            )
        return grouped, measure_field, plan.y_field

    if agg_y and plan.y_field:
        _ensure_column(plan.x_field)
        _ensure_column(plan.y_field)
        base = working[[plan.x_field, plan.y_field]].dropna()
        if agg_y == 'count':
            grouped = base.groupby(plan.x_field, dropna=False).size().reset_index(name='count')
            measure_field = 'count'
        elif agg_y == 'count_distinct':
            measure_field = f"{plan.y_field}_distinct"
            grouped = (
                base.groupby(plan.x_field, dropna=False)[plan.y_field]
                .nunique()
                .reset_index(name=measure_field)
            )
        else:
            measure_field = f"{plan.y_field}_{agg_y}"
            grouped = (
                base.groupby(plan.x_field, dropna=False)[plan.y_field]
                .agg(agg_y)
                .reset_index(name=measure_field)
            )
        return grouped, plan.x_field, measure_field

    if plan.y_field and plan.y_field in working.columns:
        subset = working[[plan.x_field, plan.y_field]].dropna()
        return subset, plan.x_field, plan.y_field

    _ensure_column(plan.x_field)
    subset = working[[plan.x_field]].dropna()
    return subset, plan.x_field, plan.x_field


def _stringify_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for col in converted.columns:
        converted[col] = converted[col].apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)
    return converted


def _execute_altair_code(code: str, df: pd.DataFrame) -> alt.Chart:
    allowed_modules = {"altair": alt, "alt": alt, "pandas": pd, "pd": pd, "math": math}

    def _safe_import(name: str, globals=None, locals=None, fromlist=None, level=0):
        target = allowed_modules.get(name)
        if target is None:
            raise ImportError(f"import not allowed: {name}")
        return target

    allowed_builtins = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "range": range,
        "enumerate": enumerate,
        "abs": abs,
        "__import__": _safe_import,
    }
    global_env = {"__builtins__": allowed_builtins, "alt": alt, "pd": pd, "math": math}
    local_env: Dict[str, Any] = {"data": df.copy(), "df": df.copy()}
    try:
        exec(compile(code, "<llm_plot>", "exec"), global_env, local_env)
    except Exception as exc:
        raise ValueError(f"Failed to execute LLM-generated code: {exc}") from exc
    chart = local_env.get("chart")
    if chart is None:
        raise ValueError("LLM-generated code must define a variable named 'chart'")
    if not isinstance(chart, alt.TopLevelMixin):
        raise ValueError("LLM-generated code did not produce an Altair chart")
    return chart



def _execute_llm_plot_code(code: str, df: pd.DataFrame) -> Tuple[Optional[plt.Figure], Optional[alt.TopLevelMixin], Any]:
    allowed_modules = {
        "altair": alt,
        "alt": alt,
        "pandas": pd,
        "pd": pd,
        "matplotlib": matplotlib,
        "matplotlib.pyplot": plt,
        "plt": plt,
        "numpy": np,
        "np": np,
        "math": math,
    }

    def _safe_import(name: str, globals=None, locals=None, fromlist=None, level=0):
        target = allowed_modules.get(name)
        if target is None:
            raise ImportError(f"import not allowed: {name}")
        return target

    allowed_builtins = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "range": range,
        "enumerate": enumerate,
        "abs": abs,
        "round": round,
        "__import__": _safe_import,
    }
    global_env = {
        "__builtins__": allowed_builtins,
        "pd": pd,
        "alt": alt,
        "plt": plt,
        "matplotlib": matplotlib,
        "np": np,
        "math": math,
    }
    local_env: Dict[str, Any] = {
        "data": df.copy(),
        "df": df.copy(),
    }
    _ensure_thai_font()
    try:
        exec(compile(code, "<llm_plot>", "exec"), global_env, local_env)
    except Exception as exc:
        raise ValueError(f"Failed to execute LLM-generated code: {exc}") from exc

    result = local_env.get("result")
    figure: Optional[plt.Figure] = None
    chart_obj: Optional[alt.TopLevelMixin] = None

    if isinstance(result, plt.Figure):
        figure = result
    elif isinstance(result, plt.Axes):
        figure = result.figure
    elif isinstance(result, alt.TopLevelMixin):
        chart_obj = result

    if figure is None:
        candidate = local_env.get("fig")
        if isinstance(candidate, plt.Figure):
            figure = candidate
        elif isinstance(candidate, plt.Axes):
            figure = candidate.figure

    if chart_obj is None:
        candidate_chart = local_env.get("chart")
        if isinstance(candidate_chart, alt.TopLevelMixin):
            chart_obj = candidate_chart

    if figure is None:
        fignums = plt.get_fignums()
        if fignums:
            figure = plt.figure(fignums[-1])

    return figure, chart_obj, result


def build_chart_from_llm_plan(
    df: pd.DataFrame,
    plan: LLMChartPlan,
    spec: Dict[str, Any],
) -> alt.Chart:
    encoding_raw = spec.get("encoding")
    encoding = encoding_raw if isinstance(encoding_raw, dict) else {}
    transform = spec.get("transform")

    filters = None
    if isinstance(transform, dict):
        filters = transform.get("filter")
    elif isinstance(transform, list):
        for step in transform:
            if isinstance(step, dict) and "filter" in step:
                filters = step["filter"]
                break

    filtered = _apply_filters(df, filters)
    prepared, final_x, final_y = _prepare_chart_dataframe(filtered, plan)
    prepared = _stringify_columns(prepared)

    chart = alt.Chart(prepared)
    mark = (plan.mark or "bar").lower()
    if mark == "bar":
        chart = chart.mark_bar()
    elif mark == "line":
        chart = chart.mark_line(point=True)
    elif mark == "area":
        chart = chart.mark_area()
    elif mark in {"scatter", "point"}:
        chart = chart.mark_circle(size=80)
    elif mark == "pie":
        chart = chart.mark_arc()
    else:
        chart = chart.mark_bar()

    x_cfg = encoding.get("x", {}) if isinstance(encoding, dict) else {}
    y_cfg = encoding.get("y", {}) if isinstance(encoding, dict) else {}
    color_cfg = encoding.get("color") if isinstance(encoding, dict) else None
    tooltip_cfg = encoding.get("tooltip") if isinstance(encoding, dict) else None

    x_type = _altair_type(x_cfg.get("type"), default="nominal")
    y_type = _altair_type(y_cfg.get("type"), default="quantitative")

    chart = chart.encode(
        x=alt.X(final_x, type=x_type),
        y=alt.Y(final_y, type=y_type),
    )

    if plan.color and plan.color in prepared.columns:
        color_type = "nominal"
        if isinstance(color_cfg, dict):
            color_type = _altair_type(color_cfg.get("type"), default="nominal")
        chart = chart.encode(color=alt.Color(plan.color, type=color_type))

    if tooltip_cfg:
        if isinstance(tooltip_cfg, list):
            tooltips = []
            for item in tooltip_cfg:
                if isinstance(item, dict):
                    field = item.get("field")
                    if field and field in prepared.columns:
                        t_type = _altair_type(item.get("type"), default="nominal")
                        tooltips.append(alt.Tooltip(field, type=t_type))
                elif isinstance(item, str) and item in prepared.columns:
                    tooltips.append(item)
            if tooltips:
                chart = chart.encode(tooltip=tooltips)
        elif isinstance(tooltip_cfg, dict):
            field = tooltip_cfg.get("field")
            if field and field in prepared.columns:
                t_type = _altair_type(tooltip_cfg.get("type"), default="nominal")
                chart = chart.encode(tooltip=alt.Tooltip(field, type=t_type))

    if plan.title:
        chart = chart.properties(title=plan.title)

    return chart.properties(width="container", height=400)


def generate_chart_with_llm(
    df: pd.DataFrame,
    instruction: str,
    llm_config: LLMConfig,
    *,
    temperature: float = 0.2,
) -> LLMPlotOutcome:
    if df.empty:
        raise ValueError("No data available to plot")
    if not instruction.strip():
        raise ValueError("Provide a plotting request for the LLM")

    schema_summary = _summarize_columns(df)
    sample_json = _sample_rows(df)
    plot_prompt = _build_matplotlib_code_prompt(df.columns.tolist(), instruction)

    system_prompt = textwrap.dedent("""You are an expert Python data visualization assistant.
You receive dataset metadata and a user's requested visualization.
Respond using JSON with this structure:
{
  "thinking": string,
  "title": string | null,
  "description": string | null,
  "output": {
    "kind": "altair_python" | "altair_spec" | "plan",
    "code": string?,
    "spec": object?,
    "mark": string?,
    "encoding": object?,
    "transform": object?
  }
}
Guidelines:
- The ONLY acceptable `kind` values are `altair_python`, `altair_spec`, or `plan`.
- NEVER use `html` as the `kind`.
- Prefer kind=altair_python and return a single ```python fenced block.
- Build the chart in a variable named `chart` and avoid display/save calls.
- Use pandas (DataFrame `data`/`df`) and Altair (imported as `alt`).
- Handle missing values with dropna/fillna before aggregations.
- Only reference columns that exist in the schema summary.
- Suggest informative titles/labels when helpful.
- Keep the thinking concise (<= 60 words).
""")

    row_count = len(df)
    summary_text = schema_summary or "(no columns summarised)"
    user_prompt = textwrap.dedent(
        f"""{plot_prompt}

Dataset rows available: {row_count}

Column summary:
{summary_text}

Sample rows (JSON):
{sample_json}
"""
    ).strip()

    llm_response = call_llm_chat(
        llm_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=900,
    )

    parsed = _extract_json_dict(llm_response)
    thinking = parsed.get("thinking") or parsed.get("analysis") or ""
    title = parsed.get("title")
    description = parsed.get("description")

    output_payload = parsed.get("output") if isinstance(parsed.get("output"), dict) else None
    chart_spec = parsed.get("chart") if isinstance(parsed.get("chart"), dict) else None

    if output_payload:
        kind = (output_payload.get("kind") or "plan").lower()
        if kind in {"altair_python", "python", "code"}:
            code = output_payload.get("code")
            if not code:
                raise ValueError("LLM output missing python code")
            code = _extract_first_code_block(code) or code
            code = _sanitize_plot_code(code)
            if not code:
                raise ValueError("LLM output did not contain executable code after sanitization")
            figure, chart_obj, raw_result = _execute_llm_plot_code(code, df)
            if figure is not None:
                return LLMPlotOutcome(
                    kind="matplotlib",
                    chart=chart_obj,
                    figure=figure,
                    raw_spec=parsed,
                    thinking=thinking,
                    title=title,
                    description=description,
                    code=code,
                )
            if chart_obj is not None:
                return LLMPlotOutcome(
                    kind="altair_python",
                    chart=chart_obj,
                    raw_spec=parsed,
                    thinking=thinking,
                    title=title,
                    description=description,
                    code=code,
                )
        spec_payload = output_payload.get("spec")
        if isinstance(spec_payload, dict):
            try:
                chart = alt.Chart.from_dict(spec_payload)
            except Exception:
                chart = None
            return LLMPlotOutcome(
                kind="altair_spec",
                chart=chart,
                raw_spec=spec_payload,
                thinking=thinking,
                title=title,
                description=description,
            )
        if not chart_spec and ("encoding" in output_payload or "mark" in output_payload):
            chart_spec = output_payload

    if not chart_spec:
        raise ValueError("LLM response missing chart specification")

    encoding = chart_spec.get("encoding") or {}
    x_cfg = encoding.get("x") or {}
    y_cfg = encoding.get("y") or {}
    color_cfg = encoding.get("color") or {}

    x_field = x_cfg.get("field")
    if not x_field:
        raise ValueError("LLM did not specify an x-axis field")

    raw_tooltip = encoding.get("tooltip") if isinstance(encoding, dict) else None
    tooltip_fields: Optional[List[str]] = None
    if isinstance(raw_tooltip, list):
        extracted: List[str] = []
        for item in raw_tooltip:
            if isinstance(item, dict):
                field = item.get("field")
                if field:
                    extracted.append(field)
            elif isinstance(item, str):
                extracted.append(item)
        if extracted:
            tooltip_fields = extracted
    elif isinstance(raw_tooltip, dict):
        field = raw_tooltip.get("field")
        if field:
            tooltip_fields = [field]

    plan = LLMChartPlan(
        mark=chart_spec.get("mark", "bar"),
        x_field=x_field,
        y_field=y_cfg.get("field"),
        x_aggregate=x_cfg.get("aggregate"),
        y_aggregate=y_cfg.get("aggregate"),
        color=color_cfg.get("field"),
        tooltip=tooltip_fields,
    )

    chart = build_chart_from_llm_plan(df, plan, chart_spec)
    return LLMPlotOutcome(
        kind="plan",
        chart=chart,
        plan=plan,
        raw_spec=parsed,
        thinking=thinking,
        title=title,
        description=description,
    )

