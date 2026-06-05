# Literature Review — HFRVLA

> Last updated: 2026-06-05
> Scope: paper-facing related-work notes for the current Fast Wrist Residual
> mainline. Use `paper/src/references.bib` as the citation source of truth.

---

## Current Position

HFRVLA should be framed as:

> A wrist-centric, LeRobot-native study of per-step residual correction for
> frozen SmolVLA action chunks.

The current paper should not claim a learned gate, contact-aware correction,
event-driven replanning, or real-hardware validation. Those ideas were part of
older HFRVLA notes and are out of the current mainline.

---

## Closest Work

| Work | What it contributes | Relationship to HFRVLA |
|---|---|---|
| A2C2: *Leave No Observation Behind* | Frozen/off-the-shelf VLA action chunks plus a lightweight per-step correction head conditioned on latest observation, base action, time feature, and policy context. | Closest conceptual anchor. HFRVLA keeps the frozen-chunk residual idea but studies a wrist-camera correction path over SmolVLA/LeRobot. |
| DynamicVLA | Treats dynamic manipulation as a perception-execution gap and addresses it through continuous inference and action streaming. | Useful problem framing for stale observations and long chunk execution. HFRVLA changes the correction module, not the VLA scheduler as the main method. |
| Reactive Diffusion Policy | Slow-fast manipulation policy with high-frequency feedback for contact-rich settings. | Supports the slow-fast motivation. HFRVLA avoids tactile/force inputs and asks how far wrist vision plus slow-planner context can go. |
| SmolVLA | Efficient open VLA backbone with action chunking and LeRobot integration. | Frozen slow planner used by HFRVLA. |
| LeRobot | Dataset/training/evaluation infrastructure for robot learning. | HFRVLA is implemented as a LeRobot policy plugin with recorder, fast-cache builder, packaging, and eval registry. |
| LIBERO | Benchmark suite for long-horizon robot manipulation tasks. | Current simulation benchmark. |
| DINOv3 | Frozen dense visual representation. | Provides the wrist patch features used by the fast residual module. |

---

## Novelty Boundary

Residual correction on top of frozen VLA chunks is not enough by itself to be
novel after A2C2. The defensible HFRVLA angle is narrower:

- High-frequency new visual evidence comes from wrist-camera DINO patches.
- Global task context is inherited from the frozen slow planner through
  `a_base`, `z_goal`, `z_phase`, robot state, and chunk index.
- The system is LeRobot-native and keeps SmolVLA frozen.
- The evaluation explicitly separates planning chunk size from execution/replan
  interval.

Avoid saying the whole policy is wrist-only. The accurate claim is that the
new per-step visual correction signal is wrist-centric.

---

## Citation Discipline

Do not copy BibTeX entries from old notes. The verified bibliography currently
lives in:

```text
paper/src/references.bib
```

If a new related work item is added, fetch or verify its BibTeX programmatically
before citing it in `paper/src/main.tex`.
