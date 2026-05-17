# Deviation Log

Per pre-registration §3 discipline: post-lock deviations must be logged here
with the deviation, the reason, and the commit hash of the deviation.

---

## Entry 001 — 2026-05-17 — Phase 2 axis-3 substitute

**Pre-reg lock:** `aa38aea`

**Pre-registered:** §2L axis 3 — "Source replication: swap xeno-canto for Macaulay Library on a subset (≥ 20% of recordings, balanced across species). Compare coefficient."

**Deviation:** Substituted a **recordist-source split within xeno-canto** for the Macaulay swap as the source-replication robustness axis, run at this commit.

**Reason:** Macaulay Library bulk API access requires a Cornell Lab research-access application (free for research, gated by institutional vetting, weeks-to-months timeline). At the time of this analysis run, that access has not been obtained. The recordist-source split is the "best available substitute": stratifies our existing 1,344 xeno-canto recordings into top-16-recordists (n=626) vs long-tail (n=604), refits Arm 1 model on each, compares. This tests whether the spectral signal is a single-recordist-equipment artifact, which is the same threat model as Macaulay source-swap (different microphones, recording styles, archives).

**The canonical Macaulay swap remains queued.** When Cornell research access lands, the canonical axis-3 will be re-run and the substitute can be either retained as a supplementary robustness check or superseded.

**Result of the substitute:** PASS on both subgroups (top: 0.265 [0.176, 0.354]; long-tail: 0.152 [0.070, 0.243]). Spectral signal is not recordist-equipment-specific.

**Files written by this deviation:**
- `_scripts/run_arm1_recordist_split.py`
- `analysis/arm1_recordist_split/recordist_split_results.json`
- Updated `analysis/REPORT_PHASE2_2026_05_17.md` with the substitute result + this deviation reference

**Decision rule used for the substitute (matches the locked pre-reg rule for axis 3):** PASS if frac_cell q025 > 0.05 on BOTH subgroups for the spectral outcome. PASS achieved on both subgroups.
