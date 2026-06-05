# Bilingual Paper Sync Memory

This project now maintains two paper source files:

- English: `paper/src/main.tex`
- Traditional Chinese: `paper/src/main_zh.tex`

Any paper-facing update to the problem statement, method framing, evaluation
protocol, limitations, citations, or final claims should update both versions in
the same change. The English and Chinese versions should remain structurally
parallel, even if individual sentences are localized rather than translated
word-for-word.

Current claim boundary:

- Keep the manuscript at the high-level architecture and evaluation-scaffold
  stage until the full experiment set is complete.
- Do not promote provisional success-rate trends into final paper claims.
- Keep detailed success-rate charts and tables in the HTML dashboard until the
  user decides the experiment set is final.
- Preserve the current method framing: wrist-camera fast residual correction on
  top of frozen `HuggingFaceVLA/smolvla_libero` context, deployed as
  `a_final = a_base + alpha * clip(delta_a)`.

Build both PDFs with:

```bash
cd /home/hucenrotia/Patrick/VLA_research/Hierachical_fast_reactive/paper
bash build_bilingual.sh
```

Expected outputs:

- `paper/build/en/main.pdf`
- `paper/build/zh/main_zh.pdf`
