#!/usr/bin/env python3
"""Synchronize the reviewed Markdown manuscript into IEEEtran LaTeX.

The Markdown file is the editing source. Pandoc performs the mechanical inline
and display-math conversion; this script restores the IEEE title/abstract,
section levels, citations, compact figure captions/labels, and bibliography.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "wbc_ieee_v4.md"
TARGET = HERE / "wbc_v4.tex"


PREAMBLE = r"""\documentclass[journal]{IEEEtran}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{graphicx}
\usepackage{array,booktabs,longtable}
\usepackage{placeins}
\usepackage{url}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{assumption}{Assumption}
\newtheorem*{remark}{Remark}

\begin{document}

\title{Interaction Dynamics: A Configuration-Invariant Predictive Model for\\
Humanoid Locomotion under Terrain and External Disturbances}
\author{Yongyan Cao%
\thanks{Y. Cao. E-mail: yongyancao@gmail.com.}}
\markboth{Manuscript}{Cao: Configuration-Invariant Interaction Dynamics for Humanoid Locomotion}
\maketitle
"""


FIGURES = {
    "figures/multirate_architecture.png": (
        "figure*", "0.96\\textwidth", "Interaction prediction with high-rate whole-body realization.", "fig:arch"
    ),
    "figures/prediction_realization_concept.png": (
        "figure", "\\columnwidth", "Fixed-model prediction and robot-specific realization.", "fig:concept"
    ),
    "figures/uneven_ground_prediction.png": (
        "figure*", "0.92\\textwidth", "Nominal and conditioned-residual prediction errors.", "fig:prediction"
    ),
    "figures/uneven_ground_tracking.png": (
        "figure*", "0.92\\textwidth", "Terrain tracking outcomes under the shared fixed plan.", "fig:terrain_tracking"
    ),
    "figures/uneven_ground_timeseries.png": (
        "figure", "\\columnwidth", "Representative future-obstacle interaction and tracking response.", "fig:terrain_response"
    ),
    "figures/external_push_summary.png": (
        "figure*", "0.92\\textwidth", "Post-push peak error and recovery to the 12 mm band.", "fig:push_summary"
    ),
    "figures/external_push_response.png": (
        "figure", "\\columnwidth", "Representative measured-phase push response.", "fig:push_response"
    ),
    "figures/uneven_ground_timing.png": (
        "figure", "\\columnwidth", "Unoptimized non-real-time timing relative to simulated periods.", "fig:timing"
    ),
}


SECTION_LABELS = {
    "Introduction": "sec:introduction",
    "Related Work": "sec:related",
    "Locomotion Interaction Dynamics": "sec:dynamics",
    "Canonical Task Model and Normalization": "sec:canonical",
    "Interaction Estimation and Prediction": "sec:estimation",
    "Constrained Interaction-Dynamics MPC": "sec:mpc",
    "Whole-Body Realization": "sec:realizer",
    "Properties and Scope": "sec:properties",
    "Environmental-Interaction Experiments": "sec:experiments",
    "Limitations": "sec:limitations",
    "Conclusion": "sec:conclusion",
}


TABLES = (
    r"""\begin{table*}[!t]
\centering
\footnotesize
\begin{tabular}{lll}
\toprule
Terrain & Definition & Purpose \\
\midrule
Flat & Nominal surface & Estimator and tracking control \\
Unilateral depression & One planned foothold 20 mm below nominal & Delayed contact and reduced early support force \\
Unilateral obstacle & One planned foothold 20 mm above nominal & Early impact and load transfer \\
Frozen rough sequence & Left patch +15 mm; right patch -20 mm & Repeated interaction mismatch \\
\bottomrule
\end{tabular}
\end{table*}""",
    r"""\begin{table*}[!t]
\centering
\caption{Uneven-ground tracking outcomes (cell medians over ten seeds).}
\label{tab:terrain}
\footnotesize
\begin{tabular}{llrrrr}
\toprule
Terrain & Controller & CoM RMS (mm) & CoM peak (mm) & Roll/pitch RMS (mrad) & Falls/10 \\
\midrule
Flat & Impedance & 4.782 & 52.204 & 66.89 & 10 \\
 & Nominal MPC & \textbf{4.110} & 11.434 & 41.89 & 0 \\
 & ID-MPC & 4.508 & \textbf{10.636} & \textbf{39.81} & 0 \\
Depression & Impedance & 5.839 & 48.472 & \textbf{66.36} & 10 \\
 & Nominal MPC & 5.821 & 47.506 & 68.53 & 10 \\
 & ID-MPC & \textbf{5.807} & \textbf{42.679} & 68.88 & 10 \\
Obstacle & Impedance & \textbf{3.989} & 11.564 & \textbf{34.32} & 0 \\
 & Nominal MPC & 4.572 & 11.434 & 40.99 & 0 \\
 & ID-MPC & 4.567 & \textbf{10.636} & 40.06 & 0 \\
Rough & Impedance & \textbf{6.072} & 48.900 & \textbf{64.44} & 10 \\
 & Nominal MPC & 6.166 & 48.325 & 66.92 & 10 \\
 & ID-MPC & 6.082 & \textbf{43.517} & 66.57 & 10 \\
\bottomrule
\end{tabular}
\end{table*}""",
    r"""\begin{table*}[!t]
\centering
\caption{External-push response (cell medians over ten seeds).}
\label{tab:push}
\footnotesize
\begin{tabular}{lrrrr}
\toprule
Condition & Peak error: imp./nom./ID (mm) & ID vs. nominal & Recovery: imp./nom./ID (s) & Falls: imp./nom./ID \\
\midrule
Lateral, double support & 60.08 / 59.28 / \textbf{50.96} & $-14.0\%$ & -- / -- / -- & 10 / 10 / 10 \\
Lateral, single support & 15.83 / 16.00 / \textbf{12.38} & $-22.6\%$ & 0.764 / 0.754 / \textbf{0.279} & 0 / 0 / 0 \\
Forward, double support & 19.91 / 19.39 / \textbf{18.18} & $-6.2\%$ & 0.752 / 0.728 / \textbf{0.661} & 0 / 0 / 0 \\
Forward, single support & 15.74 / 15.82 / \textbf{13.66} & $-13.7\%$ & -- / -- / -- & 0 / 0 / 0 \\
\bottomrule
\end{tabular}
\end{table*}""",
)


def pandoc_latex() -> str:
    return subprocess.check_output(
        ["pandoc", str(SOURCE), "-f", "markdown", "-t", "latex", "--wrap=none"],
        text=True,
    )


def replace_figure(match: re.Match[str]) -> str:
    block = match.group(0)
    path = next((path for path in FIGURES if path in block), None)
    if path is None:
        raise RuntimeError(f"unrecognized figure block: {block[:120]}")
    env, width, caption, label = FIGURES[path]
    return (
        f"\\begin{{{env}}}[!t]\n"
        "\\centering\n"
        f"\\includegraphics[width={width}]{{{path}}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\end{{{env}}}"
    )


def transform(fragment: str) -> str:
    fragment = re.sub(r"\\begin\{center\}\\rule\{0\.5\\linewidth\}\{0\.5pt\}\\end\{center\}\s*", "", fragment)
    fragment = fragment.replace("\\tightlist\n", "")
    fragment = re.sub(r"\{\[\}(\d+)\{\]\}", lambda m: f"\\cite{{ref{m.group(1)}}}", fragment)

    if "\\begin{longtable}" in fragment:
        table_index = 0

        def table(_: re.Match[str]) -> str:
            nonlocal table_index
            if table_index >= len(TABLES):
                raise RuntimeError("more Markdown tables than synchronized table definitions")
            value = TABLES[table_index]
            table_index += 1
            return value

        fragment = re.sub(
            r"\{\\def\\LTcaptype\{none\}.*?\\end\{longtable\}\s*\}",
            table,
            fragment,
            flags=re.S,
        )
        if table_index != len(TABLES):
            raise RuntimeError(f"expected {len(TABLES)} Markdown tables, found {table_index}")

    def section(match: re.Match[str]) -> str:
        title = match.group(2)
        label = SECTION_LABELS[title]
        barrier = "\\FloatBarrier\n" if title == "Conclusion" else ""
        return f"{barrier}\\section{{{title}}}\\label{{{label}}}"

    fragment = re.sub(
        r"\\subsection\{([IVX]+)\. ([^}]*)\}\\label\{[^}]*\}", section, fragment
    )
    fragment = re.sub(
        r"\\subsubsection\{[A-Z]\. ([^}]*)\}\\label\{[^}]*\}",
        lambda m: f"\\subsection{{{m.group(1)}}}",
        fragment,
    )
    fragment = re.sub(r"\\begin\{figure\}\n.*?\\end\{figure\}", replace_figure, fragment, flags=re.S)
    fragment = re.sub(r"\n\\textbf\{Fig\.\s*\d+\.\}\s*[^\n]*\n", "\n", fragment)
    fragment = fragment.replace(
        "Walking is a continuous physical interaction",
        "\\IEEEPARstart{W}{alking} is a continuous physical interaction",
        1,
    )
    return fragment.strip()


def main() -> None:
    old = TARGET.read_text()
    bibliography_match = re.search(
        r"\\begin\{thebibliography\}\{17\}.*?\\end\{thebibliography\}", old, re.S
    )
    if bibliography_match is None:
        raise RuntimeError("existing bibliography not found")
    bibliography = bibliography_match.group(0)

    converted = pandoc_latex()
    abstract_match = re.search(
        r"\\subsection\{Abstract\}\\label\{abstract\}\s*(.*?)\s*\\textbf\{Index Terms\}",
        converted,
        re.S,
    )
    if abstract_match is None:
        raise RuntimeError("converted abstract not found")
    abstract = transform(abstract_match.group(1))
    body_start = converted.index("\\subsection{I. Introduction}")
    body_end = converted.index("\\subsection{References}")
    body = transform(converted[body_start:body_end])

    output = (
        PREAMBLE
        + "\n\\begin{abstract}\n"
        + abstract
        + "\n\\end{abstract}\n\n"
        + "\\begin{IEEEkeywords}\n"
        + "Interaction dynamics, uneven-terrain locomotion, external-push rejection, "
        + "humanoid robots, model predictive control, disturbance estimation, whole-body control.\n"
        + "\\end{IEEEkeywords}\n\n"
        + body
        + "\n\n\\IEEEtriggeratref{9}\n"
        + bibliography
        + "\n\n\\end{document}\n"
    )
    TARGET.write_text(output)
    print(f"synchronized {SOURCE.name} -> {TARGET.name}")


if __name__ == "__main__":
    main()
