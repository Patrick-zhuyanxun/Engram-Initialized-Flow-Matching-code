# NYCU Thesis LaTeX Project

This is the HFRVLA thesis draft rebuilt from the local NYCU template archive:

```text
paper/templates/nycu_thesis/NYCU_thesis_template__1_.zip
```

The project preserves the template's layout:

- `main.tex` stores thesis metadata and imports the template environment.
- `covers/` contains the NYCU cover, title-page, watermark, and TOC logic.
- `Sections/` contains one TeX file per thesis component.
- `figures/` contains HFRVLA paper figures copied from `paper/src/figures/`.
- `ref.bib` mirrors the verified HFRVLA bibliography from `paper/src/references.bib`.

Build from this folder:

```bash
bash build.sh
```

The generated PDF is written to:

```text
build/main.pdf
```

Current status: complete thesis-format draft, not final administrative metadata.
Advisor, department, oral-defense date, and final title should be confirmed
before formal submission.
