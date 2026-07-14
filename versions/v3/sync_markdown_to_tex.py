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
SOURCE = HERE / "wbc_ieee.md"
TARGET = HERE / "wbc_v3.tex"


PREAMBLE = r"""\documentclass[journal]{IEEEtran}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{graphicx}
\usepackage{array}
\usepackage{url}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{assumption}{Assumption}
\newtheorem*{remark}{Remark}

\begin{document}

\title{Interaction Dynamics for Floating-Base\\ Whole-Body Manipulation}
\author{Yongyan Cao%
\thanks{Y. Cao. E-mail: yongyancao@gmail.com.}}
\markboth{Manuscript}{Cao: Interaction Dynamics for Floating-Base Whole-Body Manipulation}
\maketitle
"""


FIGURES = {
    "figures/interaction_dynamics_ports_architecture.png": (
        "figure*", "0.90\\textwidth", "Dual-MPC interaction-dynamics architecture.", "fig:arch"
    ),
    "figures/prediction_realization_concept.png": (
        "figure", "\\columnwidth", "Prediction--realization interface.", "fig:concept"
    ),
    "figures/authority_sets.png": (
        "figure", "\\columnwidth", "Current-state planar body-authority boxes.", "fig:authority_sets"
    ),
    "figures/authority_transition.png": (
        "figure*", "0.92\\textwidth", "Measured authority-gated support transition.", "fig:authority_transition"
    ),
    "figures/authority_fidelity_boundary.png": (
        "figure*", "0.90\\textwidth", "Queried-authority consistency and conditional offset-free boundary.", "fig:authority_fidelity"
    ),
    "figures/authority_closed_loop.png": (
        "figure", "\\columnwidth", "Non-real-time per-update authority query.", "fig:authority_closed_loop"
    ),
}


SECTION_LABELS = {
    "Introduction": "sec:introduction",
    "Related Work": "sec:related",
    "Floating-Base Interaction Dynamics": "sec:dynamics",
    "Body Interaction Port": "sec:body",
    "Task Interaction Port": "sec:task",
    "Whole-Body Interaction Realizer": "sec:realizer",
    "External-Wrench and Internal-Momentum Preview": "sec:preview",
    "Kalman Estimation and Contact Events": "sec:kalman",
    "Relation to the Fixed-Base Theory": "sec:theory",
    "Evaluation of Realization-Informed Feasible Authority": "sec:eval",
    "Limitations": "sec:limitations",
    "Conclusion": "sec:conclusion",
}


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
    fragment = re.sub(r"\{\[\}(\d+)\{\]\}", lambda m: f"\\cite{{b{m.group(1)}}}", fragment)

    def section(match: re.Match[str]) -> str:
        title = match.group(2)
        label = SECTION_LABELS[title]
        return f"\\section{{{title}}}\\label{{{label}}}"

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
        "Humanoid robots regulate two physical interfaces at once:",
        "\\IEEEPARstart{H}{umanoid} robots regulate two physical interfaces at once:",
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
        + "Interaction dynamics, centroidal MPC, whole-body control, floating-base robots, "
        + "loco-manipulation, physical human-robot interaction, model predictive control.\n"
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
