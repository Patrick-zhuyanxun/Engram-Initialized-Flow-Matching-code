# Round 3 Methodology Review

Agent: `019eac02-6418-77f2-8488-04daa8efd36b` (Confucius)

Role: methodology reviewer, read-only.

## Verdict

Conditional pass. The only blocking issue was n50 alpha/clip provenance: the
registry manifest had assigned the generated-checkpoint n50 sweep to the b512
FWR-v2 metadata profile even though the sweep output directory name implied the
generated checkpoint. The reviewer required direct log verification or explicit
downgrading.

## Evidence Checked

- `outputs/eval_hfrvla_fwr_chunk_generated_seq2_b1024p3_50k_n50_alpha_clip_spatial_50eps_b3/sweep_background.log`
  records `--policy.path=checkpoints/hfrvla_fwr_chunk_generated_seq2_b1024p3_50k_packaged`.
- The registry row is therefore a generated FWR-v2 checkpoint sweep, not a b512
  checkpoint sweep.

## Revision Actions Taken After Round 3

- Updated `experiments/eval_registry/sources.csv` so
  `n50_alpha_clip_spatial_50eps` uses
  `hfrvla_fwr_generated_b1024p3_50k`.
- Regenerated `eval_results_master.csv`; n50 rows now show generated checkpoint
  provenance and leave unverified LR/WD/batch metadata as `not recorded`.
- Added FWR-v1 teacher-forced previous-residual training versus inference
  predicted-residual mismatch text.
- Softened matched-chunk wording from a causal phrase to a numerical trend.
- Added an explicit reference to the n50 alpha/clip calibration table.
