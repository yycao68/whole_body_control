# Markdown–LaTeX Sentence Sync Check

Compared:

- `whole_body_control/wbc_ieee.md`
- `whole_body_control/arXiv/body.tex` (+ `main.tex`, `arxiv_main.tex` for the title)

Date: 2026-07-04, after the interaction-dynamics reframe.

## Canonical Choice

The LaTeX file is the compact double-anonymous submission version and, for this
paper, is also the *more current* of the two: several honest-renumbering updates
(v2) landed in the `.tex` but never propagated to the readable `.md`. When the two
differed in a technical claim or a number, I synchronized to the `.tex` value.
When the `.md` carried extra explanatory prose with no different claim, I left it
expanded (author-facing readable version). Intentional differences (anonymization,
numbered vs. BibTeX citations, shorter LaTeX captions) were left as-is.

## Structural Corrections Applied (both consistent)

| Location | Mismatch found | Resolution |
|---|---|---|
| Section layout | `.md` had 12 sections (separate **IX. Complete Joint Torque Equation** and **X. Architectural Comparison**); `.tex` merges them into one section **IX. Complete Torque Equation and Architectural Comparison**, giving 11 sections. Everything from IX on was misnumbered between versions. | `.md` merged to match `.tex`: IX now has subsections A (Complete Joint Torque Equation) and B (Architectural Comparison); Simulation → **X**, Conclusion → **XI**. |
| Intro roadmap (§I) | `.md` roadmap listed the old 12-section order. | Rewritten to the merged 11-section order, matching the `.tex` roadmap. |
| Contribution 7 cross-ref | `.md` cited **"(Section IX)"** for the *empirical* stability result — but §IX is the torque equation; the divergence result is in the simulation section. `.tex` correctly points to `sec:simulation`. | `.md` → **"(Section X)"** (simulations). |
| §VII cross-ref | `.md` "In the simulations of Section XI" | → **Section X** (post-merge simulations). |

## Sentence/Number Corrections Applied

| Location | Mismatch found | Correct synchronized wording |
|---|---|---|
| **Fig. 1 caption** (Scenario A) | `.md` caption was stale v0/v1 text contradicting its own Table III: claimed D6/D7 "sub-0.05 mm", "all other controllers plateau at ≥5.8 mm", and **"D3 (fixed-base MPC) diverges to 12 mm"** — but Table III has D3 at 0.21/0.15 mm and D6/D7 at 0.43/0.40 mm. | Rewritten to match Table III and the `.tex` caption: in static double support **all** predictive controllers (incl. fixed-base D3) reach sub-mm SS; contact-consistency does not distinguish them here; D6/D7 = 0.43/0.40 mm. |
| **Fig. 3 caption** (Scenario C, G1) | `.md` cited a residual SS of **"3.9 mm"**, a biped SS of **"0.037 mm"**, and **"2.5× lower SS than D1"** — none match Table V (D7 SS 2.41, D3 2.28) / Table III (biped D7 SS 0.40), and the ~4× figure in the body text. | Rewritten: D7 (2.41 mm) and D3 (2.28 mm) comparable; all predictive ~4× over D1 (9.70 mm); kept the accurate ~5 Hz position-actuator-bandwidth explanation. |
| **§V-E Kalman** | `.tex` explains that `Q_w` retains a small block on `x_e` (2 orders below the `d̂` block) to regularize against frozen-matrix/SRBD residuals, and that offset-free follows from the integrating structure of `A_aug` not the noise partition. `.md` had the numeric `Q_w` but not the explanation. | Added the "Unlike the strict random-walk model of [23]…" sentence to `.md`. |
| **§II-C Related Work** | `.md` (post-reframe) dropped the concrete predecessor result and one extension clause present in `.tex`. | Added "sub-0.05 mm under a sustained 15 N force on a 7-DOF manipulator, versus 44.8 mm for classical impedance," and the "Kalman additionally absorbs leg-momentum variations and SRBD approximation errors" clause; structure now mirrors `.tex`. |
| **§X-A Platform** | `.tex` lists friction-cone half-angle `μ = 0.6`; `.md` omitted it. | Added `μ = 0.6` to the `.md` platform parameters. |

## Technical Claims Confirmed Already Synced

| Area | Status |
|---|---|
| All benchmark tables (III–VII) | Numbers identical in both files (Scenario A/B/C/E/F). |
| Reframe sections (title, abstract, index terms, intro, contribution 1, §II-C series sentence, conclusion through-line) | Synced in the prior reframe pass; re-verified consistent. |
| §III Dynamics (eqs. 1–12) | Identical equations/claims; `.tex` slightly more compact. |
| §IV OSF / SK05, §V residual plant, Prop. 1 (constant `A_d`), QP, augmented state | Identical claims and equations. |
| Theorem 1 (Impedance Equivalence), Theorem 2 (Zero SS), transient bound (26) | Identical statements and proofs. |
| Fig. 2 (Scenario B) and Fig. 4 (Scenario F) captions | Already consistent with their tables. |

## Intentional Differences Left

- `.tex` is double-anonymous ("a line of prior work"); `.md` keeps "the authors' prior work" and author/affiliation.
- `.md` keeps longer explanatory prose the compact `.tex` omits: the Unitree G1/R1 `unitree_sdk2` hardware-interface paragraph, the anti-windup analogue sentence in §V-E, the transient-bound walking/running tuning discussion, and the multi-rate execution detail. Same claims, more words.
- Citations: `.md` numbered [1]–[23]; `.tex` BibTeX keys. The fixed-base predecessor is `cao2026impedance` (= `.md` [23]).
- Figure/table captions are equivalent in claim content but not sentence-identical (LaTeX captions are shorter).

## Verification

- `.md`: 11 sections (I–XI + References), all `Section <n>` cross-refs resolve, no stale figure numbers remain.
- `.tex`: no changes this pass (all corrections were `.md`-side). Last build stands:
  `latexmk -pdf main.tex` → **11 pages, no undefined/multiply-defined references.**

Rebuild with:

```bash
cd whole_body_control/arXiv
latexmk -pdf -interaction=nonstopmode main.tex
```
