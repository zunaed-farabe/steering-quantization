# Verification record

The repository has been exercised locally on CPU with synthetic fixtures and a tiny randomly initialized Transformers Llama model. The 29 core tests check scenario-group isolation, immutable-data manifests, correct training targets, answer-position balance, answer parsing, hand-computed BBQ scores, intervention location, zero-strength equivalence, hook cleanup, frozen validation selection, resume checks, paired bootstrap units, and the full command workflow. Three additional tests cover handoff archives, partial-result packaging, and credential-free diagnostics: all 32 pass. A standalone CLI smoke run additionally completed extraction, validation, frozen test evaluation, capability evaluation, and report generation. The isolated environment passes `pip check`.

The 128-example pilot extraction sample spans all 27 training scenarios. It creates 576 contrastive pairs with exactly 192 positive and 192 negative occurrences of each answer letter A/B/C. Every positive answer matches the gold label. The detailed source hashes, split manifest, and pair audit are in `data-audit.json`.

Official BBQ gender data was downloaded from commit `bea11bd97d79217245b5871acd247b9d6eb24598`. The identity-stratified grouped split has 3,192 extraction-pool examples (27 scenarios), 1,240 validation examples (11 scenarios), and 1,240 test examples (12 scenarios), 5,672 in total. Every split has equal ambiguous/disambiguated and negative/non-negative counts, no shared scenario groups, and no repeated context+question text across splits. Official target annotations cover all validation/test examples; 16 training-pool examples have missing targets. Raw files and prepared data are ignored by Git and retain source/file hashes locally.

## Changes after the original verification

The gender data was re-split with identity stratification. The original unstratified split put 85% name-based items in training but 43% trans/non-trans items in test. In the same revision, the GPU check was extended to INT8/NF4 and all vector sources, and the following were added: a go/no-go gate (`scripts/check_gate.py`), real kernel checks in `scripts/check_gpu.py`, an unattended resumable runner (`scripts/run_campaign.sh`, launched in tmux by `run_experiment.sh`), and explicit UTF-8/`\n` file I/O. Later in the same revision:

- Selection always carries the prompt-only control to test.
- The capability loader accepts an optional `unknown` option index, and capability reports are broken down by category.
- The 2,000-question capability suite was built by `scripts/build_capability_qa.py`. It passes the project's `load_capability` loader, contains no duplicate questions and no demographic terms in questions or options, and rebuilds byte-identically (SHA-256 `8c194ca2…`, full hash in its manifest). The new split, its manifest and `data-audit.json` were generated with the repository's own `prepare`, `load_split` and `contrastive_pairs` functions. The assertions of `tests/test_data.py` were replayed directly against the new splitter and passed. **The full pytest suite and smoke test have not been re-run since these changes** (no torch environment was available). `run_campaign.sh` runs both first on the GPU machine and stops if anything fails.

Not yet executed: Qwen2.5-7B inference, real bias-reduction measurements, CUDA LLM.int8()/NF4 execution, the capability suite on a real model, or activation quantization. These are not implied by passing offline tests. The selected GPU environment is still undecided.

The three QA examples bundled in `examples/` are schema examples, not a validation set. Synthetic smoke reports are plumbing diagnostics, not study results. See `local-environment.txt` for the CPU test dependencies and `protocol.md` for experimental boundaries.

## Dose finding and merged Gemma support

Merged from the Gemma handoff: the Gemma2 adapter, checkpoint stop IDs, zero intervention in selection, and the executed-source snapshot in result archives. Added: the BF16 dose-finding configs, runner and pass rule (`docs/dose-finding.md`). The handoff's `gender-run1` split was not copied because its three files are byte-identical to `data/prepared/gender/`.

The laptop used for this change has no PyTorch or Transformers, so **the full suite has not been run on this revision**. `run_dose.sh` runs it first on the GPU machine. Checked here, with stand-in modules for those libraries: the data, metrics, dose-rule, packaging and selection tests pass. Not exercised: real Gemma2 forward passes, stop tokens on a real checkpoint, and the end-to-end pipeline tests. `run_dose.sh` was dry-run with fake commands through a failure and a resume. The pass rule was applied to run 1's real Qwen validation predictions and returned FAIL, 0 of 6 settings, as expected.

## Run 3

Added: the stereotype contrast and its geometry diagnostic, letter-plus-option parsing, baseline-only scans that load no vectors, the bias-scan report, the verdict's direction check, prepared splits for seven more BBQ categories (pinned upstream revision, hashes in `data/raw/bbq/source.json`), the run 3 runner, and committing and pushing the results ZIP at the end of every campaign. Logs and locked configs are no longer tracked.

Checked on the laptop, with stand-in modules for PyTorch and Transformers: the run 3, dose, metrics, data and packaging tests pass (69), apart from the diagnostic test that cannot serialize a stand-in. Against run 2's real predictions, the parser change accepts 344 of Gemma's 432 invalid answers, each a letter followed by exactly that letter's option text, and rejects the rest. The runner was dry-run with stand-in commands through a failure in the Gemma Race sweep and a resume, which skipped all finished stages. The push step was run against a local Git remote: it committed only the archive, left other edits alone, reused the archive on a rerun, rebased when the remote had moved ahead, and ended DONE with instructions when the remote was unreachable.

Not run here: `tests/test_run3_model.py` and the other tiny-model tests (they need PyTorch; the runner runs the whole suite first on the GPU machine), and anything on the real models.

