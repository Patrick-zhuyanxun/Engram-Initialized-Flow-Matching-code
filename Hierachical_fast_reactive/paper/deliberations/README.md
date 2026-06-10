# Paper Deliberations

This folder stores durable outputs from academic skills, subagents, reviewer
simulations, citation audits, claim audits, and structured debates. These files
are process evidence. They are not manuscripts.

## Layout

| Path | Purpose |
|---|---|
| `shared/` | Deliberations that apply across multiple paper targets. |
| `shared/literature_searches/` | Search strategies, source screening logs, and literature matrices. |
| `shared/citation_audits/` | DOI/arXiv/CrossRef/Zotero verification reports. |
| `shared/claim_audits/` | Claim-to-source alignment checks. |
| `shared/decision_logs/` | Cross-target research decisions and rationale. |
| `<target>/review_rounds/` | Reviewer/writer rounds for a specific target. |
| `<target>/revision_logs/` | Records of edits made after review or debate. |

## Required Note Header

Each note should start with:

```markdown
# <Short Title>

- Date:
- Target:
- Skill / agent / tool:
- Input materials:
- Output type:
- Status:
```

Then record findings, disagreements, final decision, edits made, and unresolved
risks. If a deliberation changes a manuscript claim, update the corresponding
`paper/targets/<target>/source_map.md` or shared `paper/notes/*.md`.

## Current Target Folders

| Target | Folder |
|---|---|
| NYCU thesis | `nycu_thesis/` |
