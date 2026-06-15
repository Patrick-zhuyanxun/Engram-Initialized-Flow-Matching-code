# Long Paper Build Source

Updated: 2026-06-15

This folder contains the current buildable bilingual long-paper source:

```text
main.tex
main_zh.tex
```

Use `paper/targets/long_paper/README.md` for the target dashboard and claim
boundaries. Keep English and Traditional Chinese sources synchronized when
changing method framing, result claims, limitations, citations, or section
structure.

Build from `paper/`:

```bash
bash build_bilingual.sh
```

Shared figures live in `figures/`; bibliography entries live in
`references.bib`. Verify numerical claims against
`experiments/eval_registry/eval_results_master.csv` before editing.
