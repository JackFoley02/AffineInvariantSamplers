#!/usr/bin/env python3
"""Build seed-aggregated LaTeX tables from benchmark result summaries.

The input directory is expected to contain parameter directories such as
``2d``/``4d`` or ``1c``/``3.162c``.  Each parameter directory may contain a
``seeds_manifest.json`` plus per-seed ``results_summary.json`` files, a
``seeds/`` directory without a manifest, or one legacy ``results_summary.json``.

ESS/evaluation is deliberately recomputed from primitive summary fields:

    mean walker-coordinate ESS / evaluations per walker
  = mean walker-coordinate ESS / (ensemble evaluations / walkers).

The experiment diagnostics define the scalar ESS by first averaging ESS over
walkers for each coordinate and then averaging those coordinate means.  The
ratio is therefore normalized with respect to both walker and dimension count.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class TableDataError(ValueError):
    """Raised when result files cannot be aggregated without ambiguity."""


@dataclass(frozen=True)
class TableRow:
    sampler: str
    dimension: float | int | None
    condition_number: float | int | None
    distribution: str | None
    n_steps: float | int | None
    n_warmup: float | int | None
    burn_in: float | int | None
    n_walkers: float | int | None
    thinning: float | int | None
    leapfrog_steps: float | None
    acceptance_rate: float | None
    tau: float | None
    ess: float | None
    n_evaluations: float | None
    evaluation_type: str | None
    ess_per_eval: float | None
    runtime: float | None
    ess_per_second: float | None
    mean_x1: float | None
    variance_x1: float | None
    mean_x2: float | None
    variance_x2: float | None
    parameter_index: int
    seed_count: int


_PARAMETER_RE = re.compile(
    r"^(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<kind>[dc])?(?P<affine>_af)?$"
)


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _finite_float(value)) is not None]
    return float(statistics.median(finite)) if finite else None


def _same_value(values: Iterable[Any], field: str) -> Any:
    present = [value for value in values if value is not None]
    if not present:
        return None
    first = present[0]
    if any(value != first for value in present[1:]):
        raise TableDataError(
            f"inconsistent {field} across seeds: "
            + ", ".join(sorted({str(value) for value in present}))
        )
    return first


def _metadata_value(document: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in document and document[key] is not None:
            return document[key]
    return None


def _summary_value(
    document: dict[str, Any], sampler: str, *keys: str
) -> Any:
    summary = document["summary"][sampler]
    for key in keys:
        if key in summary and summary[key] is not None:
            return summary[key]
    return None


def _walkers(document: dict[str, Any], sampler: str) -> Any:
    summary_walkers = _summary_value(
        document, sampler, "actual_n_chains", "n_walkers"
    )
    if summary_walkers is not None:
        return summary_walkers
    if sampler == "Hamiltonian Walk Move":
        hwm_walkers = _metadata_value(
            document, "hwm_walkers_requested", "hwm_walkers"
        )
        if hwm_walkers is not None:
            return hwm_walkers
    return _metadata_value(document, "n_chains", "n_walkers")


def _evaluation_type(document: dict[str, Any], sampler: str) -> str | None:
    value = _summary_value(document, sampler, "evaluation_type")
    if value is None:
        if sampler == "Stretch Move":
            return "standard"
        return None
    normalized = str(value).strip().lower()
    if normalized in {"function", "standard", "log density", "log-density"}:
        return "standard"
    if normalized in {"gradient", "grad"}:
        return "gradient"
    return normalized


def _leapfrog_steps(document: dict[str, Any], sampler: str) -> Any:
    if sampler == "Stretch Move":
        return 1
    return _summary_value(
        document,
        sampler,
        "n_leapfrog",
        "leapfrog_steps",
        "mean_L",
    )


def _evaluation_count(document: dict[str, Any], sampler: str) -> float | None:
    stored = _finite_float(_summary_value(document, sampler, "n_evaluations"))
    if stored is not None:
        return stored

    walkers = _finite_float(_walkers(document, sampler))
    n_steps = _finite_float(_metadata_value(document, "n_samples", "n_steps"))
    thinning = _finite_float(_metadata_value(document, "n_thin", "thinning"))
    if walkers is None or n_steps is None or thinning is None:
        return None

    transitions = n_steps * thinning
    if _evaluation_type(document, sampler) == "standard":
        return walkers * transitions

    leapfrog = _finite_float(
        _summary_value(document, sampler, "n_leapfrog", "leapfrog_steps")
    )
    if leapfrog is None:
        return None
    evaluations_per_transition = (
        leapfrog if sampler == "Dense-mass NUTS" else leapfrog + 1.0
    )
    return walkers * transitions * evaluations_per_transition


def _normalized_ess_per_eval(
    document: dict[str, Any], sampler: str
) -> float | None:
    """Return a dimension- and walker-normalized per-seed efficiency."""
    mean_walker_coordinate_ess = _finite_float(
        _summary_value(document, sampler, "ess")
    )
    walkers = _finite_float(_walkers(document, sampler))
    ensemble_evaluations = _evaluation_count(document, sampler)
    if (
        mean_walker_coordinate_ess is None
        or walkers is None
        or walkers <= 0.0
        or ensemble_evaluations is None
        or ensemble_evaluations <= 0.0
    ):
        return None
    evaluations_per_walker = ensemble_evaluations / walkers
    return mean_walker_coordinate_ess / evaluations_per_walker


def _warmup(document: dict[str, Any], sampler: str) -> Any:
    value = _summary_value(document, sampler, "n_warmup", "warmup_steps")
    if value is not None:
        return value
    value = _metadata_value(document, "n_warmup", "warmup_steps")
    if value is not None:
        return value
    if _evaluation_type(document, sampler) == "standard":
        return 0
    return None


def _sequence_value(value: Any, index: int) -> float | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    if index >= len(value):
        return None
    return _finite_float(value[index])


@functools.lru_cache(maxsize=None)
def _legacy_array_moments(
    archive_path: Path, sampler: str, affine: bool
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Recover intrinsic-coordinate moments from a legacy light archive."""
    if not archive_path.is_file():
        return None
    key = sampler.replace(" ", "_")
    try:
        with np.load(archive_path) as arrays:
            if affine:
                series_key = next(
                    (
                        candidate
                        for candidate in (
                            f"{key}_series_thin10",
                            f"{key}_series",
                        )
                        if candidate in arrays
                    ),
                    None,
                )
                if series_key is None or not all(
                    name in arrays for name in ("A", "B")
                ):
                    return None
                series = np.asarray(arrays[series_key], dtype=float)
                inverse = np.linalg.inv(np.asarray(arrays["A"], dtype=float))
                offset = np.asarray(arrays["B"], dtype=float)
                intrinsic = (series - offset) @ inverse.T
                flat = intrinsic.reshape(-1, intrinsic.shape[-1])
                means = np.mean(flat, axis=0)
                variances = np.var(flat, axis=0, ddof=1)
            else:
                mean_key = f"{key}_mean"
                covariance_key = f"{key}_cov"
                if not all(name in arrays for name in (mean_key, covariance_key)):
                    return None
                means = np.asarray(arrays[mean_key], dtype=float)
                variances = np.diag(
                    np.asarray(arrays[covariance_key], dtype=float)
                )
    except (OSError, ValueError, np.linalg.LinAlgError):
        return None
    return tuple(means.tolist()), tuple(variances.tolist())


def _parameter_moments(
    document: dict[str, Any], sampler: str
) -> tuple[tuple[float | None, float | None], ...]:
    """Return the first two intrinsic-coordinate (mean, variance) pairs."""
    stored_means = _summary_value(document, sampler, "parameter_mean")
    stored_variances = _summary_value(
        document, sampler, "parameter_variance"
    )
    if stored_means is None or stored_variances is None:
        source_path = document.get("_source_path")
        recovered = (
            _legacy_array_moments(
                Path(source_path).with_name("results_arrays_light.npz"),
                sampler,
                bool(document.get("transform_affine", document.get("af", False))),
            )
            if source_path is not None
            else None
        )
        if recovered is not None:
            stored_means, stored_variances = recovered

    return tuple(
        (
            _sequence_value(stored_means, index),
            _sequence_value(stored_variances, index),
        )
        for index in range(2)
    )


def _parse_parameter_tokens(raw_tokens: Sequence[str]) -> list[str]:
    result: list[str] = []
    for raw in raw_tokens:
        result.extend(token.strip() for token in raw.split(",") if token.strip())
    if not result:
        raise TableDataError("at least one testing parameter is required")
    return result


def _parameter_directory_candidates(
    results_dir: Path, token: str, parameter_kind: str
) -> list[Path]:
    exact = results_dir / token
    if exact.is_dir():
        return [exact]

    match = _PARAMETER_RE.fullmatch(token)
    if not match:
        return []
    requested_number = float(match.group("number"))
    requested_suffix = match.group("kind")
    requested_affine = match.group("affine")
    allowed_suffixes = {
        "dimension": {"d"},
        "condition": {"c"},
        "auto": {"d", "c"},
    }[parameter_kind]
    if requested_suffix:
        allowed_suffixes &= {requested_suffix}

    candidates: list[Path] = []
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        child_match = _PARAMETER_RE.fullmatch(child.name)
        if not child_match or child_match.group("kind") not in allowed_suffixes:
            continue
        if requested_affine != child_match.group("affine") and requested_affine:
            continue
        if math.isclose(
            float(child_match.group("number")),
            requested_number,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            candidates.append(child)

    if not requested_affine:
        non_affine = [
            path
            for path in candidates
            if not _PARAMETER_RE.fullmatch(path.name).group("affine")
        ]
        if non_affine:
            candidates = non_affine
    return sorted(candidates)


def resolve_parameter_directories(
    results_dir: Path, raw_tokens: Sequence[str], parameter_kind: str
) -> list[Path]:
    if not results_dir.is_dir():
        raise TableDataError(f"results directory does not exist: {results_dir}")

    resolved: list[Path] = []
    for token in _parse_parameter_tokens(raw_tokens):
        candidates = _parameter_directory_candidates(
            results_dir, token, parameter_kind
        )
        if not candidates:
            raise TableDataError(
                f"no parameter directory in {results_dir} matches {token!r}"
            )
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise TableDataError(
                f"parameter {token!r} is ambiguous ({names}); "
                "set --parameter-kind or include the d/c suffix"
            )
        if candidates[0] in resolved:
            raise TableDataError(
                f"parameter {token!r} resolves to duplicate directory "
                f"{candidates[0].name}"
            )
        resolved.append(candidates[0])
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise TableDataError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict) or not isinstance(
        document.get("summary"), dict
    ):
        raise TableDataError(f"{path} has no summary object")
    document["_source_path"] = path
    return document


def load_parameter_documents(parameter_dir: Path) -> list[dict[str, Any]]:
    manifest_path = parameter_dir / "seeds_manifest.json"
    paths: list[Path] = []
    if manifest_path.exists():
        try:
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise TableDataError(f"cannot read {manifest_path}: {error}") from error
        runs = manifest.get("runs", []) if isinstance(manifest, dict) else []
        for run in runs:
            if not isinstance(run, dict) or "directory" not in run:
                raise TableDataError(f"invalid run entry in {manifest_path}")
            paths.append(parameter_dir / run["directory"] / "results_summary.json")
    if not paths:
        paths = sorted(
            (parameter_dir / "seeds").glob("seed_*/results_summary.json")
        )
    if not paths and (parameter_dir / "results_summary.json").exists():
        paths = [parameter_dir / "results_summary.json"]
    if not paths:
        raise TableDataError(
            f"no results_summary.json files found beneath {parameter_dir}"
        )

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise TableDataError(
            "manifest references missing result files: "
            + ", ".join(str(path) for path in missing)
        )
    documents = [_read_json(path) for path in paths]
    seed_values = [
        document.get("seed") for document in documents if document.get("seed") is not None
    ]
    if len(seed_values) != len(set(seed_values)):
        raise TableDataError(f"duplicate seed values beneath {parameter_dir}")
    return documents


def aggregate_parameter(
    documents: Sequence[dict[str, Any]], parameter_index: int
) -> list[TableRow]:
    if not documents:
        return []
    sampler_sets = [set(document["summary"]) for document in documents]
    if any(samplers != sampler_sets[0] for samplers in sampler_sets[1:]):
        raise TableDataError("sampler sets differ across seeds")

    rows: list[TableRow] = []
    for sampler in sorted(sampler_sets[0], key=str.casefold):
        dimension = _same_value(
            (_metadata_value(document, "dim", "dimension") for document in documents),
            "dimension",
        )
        condition = _same_value(
            (
                _metadata_value(document, "condition_number", "cond")
                if _metadata_value(document, "condition_number", "cond") is not None
                else _summary_value(document, sampler, "condition_number", "cond")
                for document in documents
            ),
            "condition number",
        )
        distribution = _same_value(
            (_summary_value(document, sampler, "distribution") for document in documents),
            "distribution",
        )
        n_steps = _same_value(
            (_metadata_value(document, "n_samples", "n_steps") for document in documents),
            "number of steps",
        )
        n_warmup = _same_value(
            (_warmup(document, sampler) for document in documents),
            f"{sampler} warmup steps",
        )
        burn_in = _same_value(
            (_metadata_value(document, "burn_in", "n_burn_in") for document in documents),
            "burn-in steps",
        )
        n_walkers = _same_value(
            (_walkers(document, sampler) for document in documents),
            f"{sampler} walker count",
        )
        thinning = _same_value(
            (_metadata_value(document, "n_thin", "thinning") for document in documents),
            "thinning factor",
        )
        evaluation_type = _same_value(
            (_evaluation_type(document, sampler) for document in documents),
            f"{sampler} evaluation type",
        )

        median_ess = _median(
            _summary_value(document, sampler, "ess") for document in documents
        )
        median_runtime = _median(
            _summary_value(document, sampler, "time", "runtime")
            for document in documents
        )
        ess_per_second = (
            median_ess / median_runtime
            if median_ess is not None
            and median_runtime is not None
            and median_runtime > 0.0
            else None
        )
        moments = [_parameter_moments(document, sampler) for document in documents]
        rows.append(
            TableRow(
                sampler=sampler,
                dimension=dimension,
                condition_number=condition,
                distribution=distribution,
                n_steps=n_steps,
                n_warmup=n_warmup,
                burn_in=burn_in,
                n_walkers=n_walkers,
                thinning=thinning,
                leapfrog_steps=_median(
                    _leapfrog_steps(document, sampler) for document in documents
                ),
                acceptance_rate=_median(
                    _summary_value(
                        document,
                        sampler,
                        "acceptance_rate_mean",
                        "acceptance_rate",
                    )
                    for document in documents
                ),
                tau=_median(
                    _summary_value(
                        document,
                        sampler,
                        "tau",
                        "integrated_autocorrelation_time",
                    )
                    for document in documents
                ),
                ess=median_ess,
                n_evaluations=_median(
                    _evaluation_count(document, sampler) for document in documents
                ),
                evaluation_type=evaluation_type,
                ess_per_eval=_median(
                    _normalized_ess_per_eval(document, sampler)
                    for document in documents
                ),
                runtime=median_runtime,
                ess_per_second=ess_per_second,
                mean_x1=_median(moment[0][0] for moment in moments),
                variance_x1=_median(moment[0][1] for moment in moments),
                mean_x2=_median(moment[1][0] for moment in moments),
                variance_x2=_median(moment[1][1] for moment in moments),
                parameter_index=parameter_index,
                seed_count=len(documents),
            )
        )
    return rows


def build_rows(
    parameter_dirs: Sequence[Path], row_order: str
) -> list[TableRow]:
    rows = [
        row
        for parameter_index, parameter_dir in enumerate(parameter_dirs)
        for row in aggregate_parameter(
            load_parameter_documents(parameter_dir), parameter_index
        )
    ]
    if row_order == "parameter":
        return sorted(rows, key=lambda row: (row.parameter_index, row.sampler.casefold()))
    return sorted(rows, key=lambda row: (row.sampler.casefold(), row.parameter_index))


def _tex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _format_number(value: Any, style: str = "general") -> str:
    number = _finite_float(value)
    if number is None:
        return "--"
    if style == "integer":
        return f"{number:.0f}"
    if style == "acceptance":
        return f"{number:.3f}"
    if style == "scientific":
        return f"{number:.3e}"
    if style == "runtime":
        return f"{number:.3f}"
    if abs(number - round(number)) < 1e-12 and abs(number) < 1e6:
        return f"{number:.0f}"
    if number != 0.0 and (abs(number) < 1e-3 or abs(number) >= 1e6):
        return f"{number:.3e}"
    return f"{number:.3f}"


_SAMPLER_ABBREVIATIONS = {
    "Dense-mass NUTS": "NUTS",
    "Hamiltonian Walk Move": "HWM",
    "Stretch Move": "SM",
    "HMC": "HMC",
}


def _sampler_label(name: str) -> str:
    return _SAMPLER_ABBREVIATIONS.get(name, name)


def _row_cells(row: TableRow) -> list[str]:
    return [
        _tex_escape(_sampler_label(row.sampler)),
        _format_number(row.dimension),
        _format_number(row.condition_number),
        _tex_escape(row.distribution) if row.distribution is not None else "--",
        _format_number(row.n_steps, "integer"),
        _format_number(row.n_warmup, "integer"),
        _format_number(row.burn_in, "integer"),
        _format_number(row.n_walkers, "integer"),
        _format_number(row.thinning, "integer"),
        _format_number(row.leapfrog_steps),
        _format_number(row.acceptance_rate, "acceptance"),
        _format_number(row.tau),
        _format_number(row.ess),
        _format_number(row.n_evaluations, "scientific"),
        _tex_escape(row.evaluation_type) if row.evaluation_type else "--",
        _format_number(row.ess_per_eval, "scientific"),
        _format_number(row.runtime, "runtime"),
        _format_number(row.ess_per_second),
        _format_number(row.mean_x1),
        _format_number(row.variance_x1),
        _format_number(row.mean_x2),
        _format_number(row.variance_x2),
    ]


_COLUMN_SPEC = (
    r"@{}>{\raggedright\arraybackslash}p{1.05cm}"
    r">{\raggedleft\arraybackslash}p{0.65cm}"
    r">{\raggedleft\arraybackslash}p{0.90cm}"
    r">{\raggedright\arraybackslash}p{1.65cm}"
    r">{\raggedleft\arraybackslash}p{0.85cm}"
    r">{\raggedleft\arraybackslash}p{0.90cm}"
    r">{\raggedleft\arraybackslash}p{0.85cm}"
    r">{\raggedleft\arraybackslash}p{0.85cm}"
    r">{\raggedleft\arraybackslash}p{0.55cm}"
    r">{\raggedleft\arraybackslash}p{0.90cm}"
    r">{\raggedleft\arraybackslash}p{0.85cm}"
    r">{\raggedleft\arraybackslash}p{0.90cm}"
    r">{\raggedleft\arraybackslash}p{1.00cm}"
    r">{\raggedleft\arraybackslash}p{1.35cm}"
    r">{\raggedright\arraybackslash}p{1.00cm}"
    r">{\raggedleft\arraybackslash}p{1.30cm}"
    r">{\raggedleft\arraybackslash}p{0.95cm}"
    r">{\raggedleft\arraybackslash}p{0.95cm}"
    r">{\raggedleft\arraybackslash}p{0.80cm}"
    r">{\raggedleft\arraybackslash}p{1.00cm}"
    r">{\raggedleft\arraybackslash}p{0.80cm}"
    r">{\raggedleft\arraybackslash}p{1.00cm}@{}"
)

_HEADERS = [
    "Sampler",
    "Dim.",
    "Cond.",
    "Distribution",
    "Steps",
    "Warmup",
    "Burn-in",
    "Walkers",
    "Thin",
    r"\shortstack{Leapfrog\\steps}",
    "Accept.",
    "IAT",
    "ESS",
    "Evaluations",
    "Type",
    "ESS/eval",
    r"\shortstack{Runtime\\(s)}",
    "ESS/s",
    r"$\mu_1$",
    r"$\mathrm{Var}(x_1)$",
    r"$\mu_2$",
    r"$\mathrm{Var}(x_2)$",
]


def render_latex(
    rows: Sequence[TableRow],
    caption: str,
    label: str,
    standalone: bool = True,
) -> str:
    if not rows:
        raise TableDataError("cannot render a table with no rows")
    seed_counts = sorted({row.seed_count for row in rows})
    seed_text = (
        str(seed_counts[0])
        if len(seed_counts) == 1
        else f"{seed_counts[0]}--{seed_counts[-1]}"
    )
    lines: list[str] = []
    if standalone:
        lines.extend(
            [
                r"\documentclass[10pt]{article}",
                r"\usepackage[margin=0.45in]{geometry}",
                r"\usepackage{array}",
                r"\usepackage{booktabs}",
                r"\usepackage{longtable}",
                r"\usepackage{pdflscape}",
                r"\begin{document}",
            ]
        )
    else:
        lines.append(
            r"% Requires: \usepackage{array,booktabs,longtable,pdflscape}"
        )
    lines.extend(
        [
            r"\begin{landscape}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            rf"\begin{{longtable}}{{{_COLUMN_SPEC}}}",
            rf"\caption{{{_tex_escape(caption)}}}\label{{{_tex_escape(label)}}}\\",
            r"\toprule",
            " & ".join(_HEADERS) + r" \\",
            r"\midrule",
            r"\endfirsthead",
            rf"\multicolumn{{22}}{{c}}{{\tablename\ \thetable{{}} continued}}\\",
            r"\toprule",
            " & ".join(_HEADERS) + r" \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            r"\multicolumn{22}{r}{Continued on next page}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )
    previous_group: tuple[Any, ...] | None = None
    parameter_major = all(
        rows[index].parameter_index <= rows[index + 1].parameter_index
        for index in range(len(rows) - 1)
    )
    for row in rows:
        group = (
            (row.parameter_index,) if parameter_major else (row.sampler.casefold(),)
        )
        if previous_group is not None and group != previous_group:
            lines.append(r"\addlinespace")
        lines.append(" & ".join(_row_cells(row)) + r" \\")
        previous_group = group
    lines.extend(
        [
            r"\end{longtable}",
            r"\par\footnotesize",
            (
                rf"Performance entries are medians across {_tex_escape(seed_text)} "
                r"seed(s). ESS is the mean of walker ESS values and then the mean "
                r"over dimensions. ESS/eval is the per-seed ratio of that ESS to "
                r"ensemble evaluations divided by the actual walker count; the "
                r"table reports its seed median. ESS/second is median ESS divided "
                r"by median runtime. Evaluation counts cover retained, post-burn-in "
                r"sampling transitions; runtime covers the complete sampler call. "
                r"Leapfrog steps are seed medians; Stretch Move is assigned one. "
                r"Parameter moments are component-wise seed medians in the "
                r"intrinsic (pre-affine-transform) coordinate system. Sampler "
                r"labels: NUTS = Dense-mass NUTS; HWM = Hamiltonian Walk Move; "
                r"SM = Stretch Move."
            ),
            r"\end{landscape}",
        ]
    )
    if standalone:
        lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def _default_output(results_dir: Path) -> Path:
    return results_dir / "benchmark_table.tex"


def compile_latex(tex_path: Path, compiler: str = "pdflatex") -> Path:
    """Compile a standalone table twice and return its PDF path.

    Auxiliary LaTeX files are kept in a temporary build directory and removed
    automatically.  Only the final PDF is placed beside ``tex_path``.
    """
    executable = shutil.which(compiler)
    if executable is None:
        raise TableDataError(
            f"LaTeX compiler {compiler!r} was not found. Install it or rerun "
            "with --no-compile to generate only the .tex source."
        )
    if tex_path.suffix.lower() != ".tex":
        raise TableDataError(
            f"compiled output must use a .tex filename, got: {tex_path}"
        )

    pdf_path = tex_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory(
        prefix=f".{tex_path.stem}_build_", dir=tex_path.parent
    ) as temporary:
        build_dir = Path(temporary)
        command = [
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={build_dir}",
            str(tex_path),
        ]
        for _ in range(2):
            completed = subprocess.run(
                command,
                cwd=tex_path.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if completed.returncode:
                tail = "\n".join(completed.stdout.splitlines()[-25:])
                raise TableDataError(
                    f"{compiler} could not compile {tex_path}:\n{tail}"
                )

        compiled_pdf = build_dir / f"{tex_path.stem}.pdf"
        if not compiled_pdf.is_file():
            raise TableDataError(
                f"{compiler} completed without producing {compiled_pdf.name}"
            )
        compiled_pdf.replace(pdf_path)
    return pdf_path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a LaTeX benchmark table by aggregating results_summary.json "
            "files across seeds."
        )
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Results root containing directories such as 2d or 3.162c.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        required=True,
        metavar="VALUE",
        help="Testing parameters; space- or comma-separated, with optional d/c suffix.",
    )
    parser.add_argument(
        "--parameter-kind",
        choices=("auto", "dimension", "condition"),
        default="auto",
        help="How to interpret unsuffixed parameter values (default: auto).",
    )
    parser.add_argument(
        "--row-order",
        "--sort-by",
        choices=("parameter", "sampler"),
        default="parameter",
        help="Group rows by testing parameter or sampler (default: parameter).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .tex path (default: RESULTS_DIR/benchmark_table.tex).",
    )
    parser.add_argument(
        "--fragment",
        action="store_true",
        help=(
            "Write only the table fragment instead of a standalone LaTeX "
            "document; fragment output is not compiled."
        ),
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile",
        dest="compile_pdf",
        action="store_true",
        help="Compile the standalone .tex table to PDF (default).",
    )
    compile_group.add_argument(
        "--no-compile",
        dest="compile_pdf",
        action="store_false",
        help="Generate the .tex source without compiling a PDF.",
    )
    parser.set_defaults(compile_pdf=True)
    parser.add_argument(
        "--latex-compiler",
        default="pdflatex",
        metavar="COMMAND",
        help="pdflatex-compatible compiler command (default: pdflatex).",
    )
    parser.add_argument(
        "--caption",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:sampler-benchmarks",
        help="LaTeX label (default: tab:sampler-benchmarks).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        results_dir = args.results_dir.expanduser().resolve()
        parameter_dirs = resolve_parameter_directories(
            results_dir, args.parameters, args.parameter_kind
        )
        rows = build_rows(parameter_dirs, args.row_order)
        caption = args.caption or (
            f"Sampler benchmark summary for {results_dir.name}"
        )
        output = (
            args.output.expanduser().resolve()
            if args.output
            else _default_output(results_dir)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render_latex(
                rows,
                caption=caption,
                label=args.label,
                standalone=not args.fragment,
            ),
            encoding="utf-8",
        )
        pdf_output = (
            compile_latex(output, args.latex_compiler)
            if args.compile_pdf and not args.fragment
            else None
        )
    except (OSError, TableDataError) as error:
        parser.error(str(error))
    print(
        f"Wrote {len(rows)} rows from "
        f"{sum(row.seed_count for row in rows)} sampler-seed results to {output}"
    )
    if pdf_output is not None:
        print(f"Compiled PDF to {pdf_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
