# Literature Review — Hierarchical Fast-Reactive VLA (HFRVLA)

> Last updated: 2026-05-12
> Scope: 2024–2026 work on residual / reactive / fast-slow correction layers over Vision-Language-Action models, with priority on closing the LIBERO + real-hardware gap.

---

## 1. Closest Competitors (direct overlap)

| # | Paper | Venue / Date | Sensor | Freq | Real HW? | Frozen base? | Gate? | Residual? | Replan trigger? |
|---|-------|--------------|--------|------|----------|--------------|-------|-----------|-----------------|
| 1 | **A2C2** — *Leave No Observation Behind: Real-time Correction for VLA Action Chunks* (Sendai et al.) | arXiv:2509.23224, Sep 2025 | wrist+third RGB | per-step (4.7 ms head) | **No (sim only)** | **Yes** | **No** | **Yes** | No |
| 2 | **Residual Policy Adaptation for VLA in Precision Assembly** | 2025 (ResearchGate 401122606) | RGB | per-step | No (lab assembly only) | **Yes** | No | **Yes** | No |
| 3 | **DuoCore-FS** — *Asynchronous Fast-Slow VLA Policies* (Astribot, Zou et al.) | arXiv:2512.20188, Dec 2025 | multi-view RGB | 30 Hz | **Yes (whole-body)** | No (joint) | implicit | latent buffer | implicit async |
| 4 | **FAVLA** — *Force-Adaptive Fast–Slow VLA* (Li et al.) | arXiv:2602.23648, Feb 2026 | F/T sensor | variable hi-freq | unclear | unclear | No | **force adapter** | No |
| 5 | **Self-Correcting VLA: Online Action Refinement via Sparse World Imagination** (Liu et al.) | arXiv:2602.21633, Feb 2026 | RGB + predictive head | per-step | **Yes (14 % gain real-world)** | unclear | No | refinement module | implicit (world-model error) |
| 6 | **Reactive Diffusion Policy (RDP / PhaForce)** (Xie et al.) | RSS 2025 / arXiv:2503.02881 | tactile + RGB | hi-rate residual | **Yes (86 % SR)** | **No (joint)** | **Phase predictor** | **Yes** | **Yes (phase-routed)** |
| 7 | **DreamTacVLA** | arXiv:2512.23864, Dec 2025 | RGB (frozen ResNet-18) + tactile dream | per-step | **Yes** | **Yes** | No | refinement | No |
| 8 | **WristWorld** | arXiv:2510.07313, 2025 | wrist 4D world model | n/a (generator) | sim-extend | n/a | n/a | n/a | n/a |

> Verdict: residual-on-frozen-VLA is **no longer novel as an idea**. Differentiation must come from (a) wrist-only sensing, (b) confidence gate semantics, (c) event-driven re-planning, (d) real-hardware demonstration on LeRobot.

---

## 2. Hierarchical / Fast-Slow Context

| Paper | Venue | What is the "fast" path? | What we differ |
|-------|-------|--------------------------|----------------|
| **Hi Robot** (Liu et al.) | ICML 2025 | Fast = action policy; Slow = LLM planner | Hi-Robot's fast is NOT closed-loop visual residual; we close the wrist-vision loop |
| **HiRT** (Hierarchical Robot Transformers) | 2025 | Multi-rate transformer | HiRT has no explicit residual + gate |
| **RT-H** (Belkhale et al.) | RSS 2024, arXiv:2403.01823 | Language-motion intermediate | RT-H operates at language layer, not motor layer |
| **HAMSTER** (Ge et al.) | ICLR 2025 | Coarse-to-fine 3D | HAMSTER is offline planner + 3D, no fast residual |
| **LiLo-VLA** | arXiv:2602.21531 | Motion planner + object-centric | Long-horizon, replanning by failure detection |
| **Helix** (Figure AI) | Industry, 2025 | 200 Hz visuomotor S1 + slow S2 | Closed-source; concept aligned but no public arch |

---

## 3. Foundation VLA Backbones (build-on candidates)

| Paper | Best for us | Notes |
|-------|-------------|-------|
| **SmolVLA** (HF) | **Primary**: small (~450 M), open, LeRobot-native, action chunking + flow matching | Our System 2 |
| OpenVLA / OpenVLA-OFT | Reference baseline | 97.1 % LIBERO SR, action chunk + parallel decoding |
| π0 / π0.5 (Physical Intelligence) | Reference | Strong but closed |
| GR-2 (NVIDIA) | Reference | Video pretraining, large |
| X-VLA | Reference for **frozen-backbone prompting** | 1 % params tuned matches π0 on LIBERO |

---

## 4. Failure-Centric & Adaptive-Horizon Work

| Paper | Date | What it shows |
|-------|------|---------------|
| **RoboFAC** | arXiv:2505.12224 | VLAs trained from success demos lack failure-centric supervision |
| **Yell At Your Robot** | RSS 2024 | Human language correction during execution recovers tasks |
| **RL-for-VLA** | 2025 | SFT VLAs compound errors under distribution shift; RL fixes |
| **Real-Time Action Chunking Flow** (π team) | 2025 | Asynchronous decoding via flow matching |
| **AdaWorldPolicy** | arXiv:2602.20057 | World-model prediction error → online updates |

---

## 5. Honest Novelty Gap for HFRVLA

After this review, HFRVLA's **defensible contribution** narrows to the **intersection** of the following — no prior work covers all four simultaneously:

- (i) **Wrist-camera-only** fast loop (no third-person, no tactile, no force) → distinct from A2C2 (multi-view), RDP / FAVLA / DreamTacVLA (tactile or force)
- (ii) **Learned confidence gate** with phase-aware semantics → distinct from A2C2 (pure additive)
- (iii) **Event-driven re-planning** signal from fast module to slow VLA → distinct from A2C2 (fixed horizon), Hi Robot (fixed schedule)
- (iv) **Real-hardware demonstration on the LeRobot + SmolVLA ecosystem** → A2C2's explicit limitation; existing real-hardware work uses π or RT-style proprietary stacks

**Three risk papers that may scoop us before submission:**
1. A successor to **A2C2** that adds a gate or moves to real hardware
2. **DuoCore-FS** (Astribot, Dec 2025) — already real hardware; if Astribot releases a residual-style follow-up, our angle is gone
3. **Self-Correcting VLA** (Feb 2026) — already real-world tested; if its team adds wrist-camera reactivity, that's the same paper

---

## 6. Recommended Baselines for Experiments

**Primary baselines (must compare)**:
- SmolVLA alone (frozen, no correction)
- SmolVLA + A2C2-style correction head (re-implement)
- SmolVLA + ResNet-based residual (no DINO, no gate)

**Secondary baselines (recommended)**:
- OpenVLA-OFT (when computationally feasible) — strong LIBERO performance ceiling
- Reactive Diffusion Policy with vision-only ablation
- X-VLA frozen-backbone reference

**Real-hardware baselines**:
- SmolVLA fine-tuned only (no correction)
- HFRVLA (ours)

---

## 7. BibTeX Entries (top-10)

```bibtex
@article{a2c2_2025,
  title   = {Leave No Observation Behind: Real-time Correction for VLA Action Chunks},
  author  = {Sendai, Kohei and Alvarez, Maxime and Matsushima, Tatsuya and Matsuo, Yutaka and Iwasawa, Yusuke},
  journal = {arXiv preprint arXiv:2509.23224},
  year    = {2025}
}

@article{duocore_fs_2025,
  title   = {Asynchronous Fast-Slow Vision-Language-Action Policies for Whole-Body Robotic Manipulation},
  author  = {Zou, Teqiang and others},
  journal = {arXiv preprint arXiv:2512.20188},
  year    = {2025},
  note    = {Astribot}
}

@article{favla_2026,
  title   = {FAVLA: A Force-Adaptive Fast--Slow VLA model for Contact-Rich Robotic Manipulation},
  author  = {Li, Yao and Tang, Peiyuan and Zhang, Wuyang and others},
  journal = {arXiv preprint arXiv:2602.23648},
  year    = {2026}
}

@article{self_correcting_vla_2026,
  title   = {Self-Correcting VLA: Online Action Refinement via Sparse World Imagination},
  author  = {Liu and Tan and Zhu and others},
  journal = {arXiv preprint arXiv:2602.21633},
  year    = {2026}
}

@inproceedings{rdp_2025,
  title     = {Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation},
  author    = {Xie, Han and others},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2503.02881}
}

@inproceedings{hi_robot_2025,
  title     = {Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models},
  author    = {Liu and others},
  booktitle = {ICML},
  year      = {2025}
}

@inproceedings{rt_h_2024,
  title     = {RT-H: Action Hierarchies Using Language},
  author    = {Belkhale, Suneel and others},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2024}
}

@article{smolvla_2025,
  title  = {SmolVLA: Efficient Vision-Language-Action Model trained on LeRobot Community Data},
  author = {Hugging Face team},
  year   = {2025},
  url    = {https://huggingface.co/blog/smolvla}
}

@article{openvla_oft_2025,
  title  = {OpenVLA-OFT: Fine-Tuning Vision-Language-Action Models for Speed and Success},
  author = {Kim and others},
  year   = {2025}
}

@article{wristworld_2025,
  title  = {WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation},
  year   = {2025},
  url    = {https://arxiv.org/abs/2510.07313}
}
```

---

## 8. Positioning Statement (draft)

> While recent work (A2C2; Sendai et al., 2025) has demonstrated that lightweight per-step correction heads on frozen VLAs improve simulated success rates, none has simultaneously (i) closed the loop with wrist-camera observations alone, (ii) learned a phase-aware confidence gate that determines when to apply correction, (iii) emitted an event-driven re-planning signal that adaptively triggers the slow VLA, and (iv) demonstrated all of the above on commodity LeRobot hardware. **HFRVLA fills this intersection** and frames the small reactive module as a "cerebellum-like" companion to the cortical-scale VLA — a deployable real-robot system, not only a simulation result.

---

*Source links*: see `paper/src/bib/` for full BibTeX with URLs.
