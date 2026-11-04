# Results so far

What has been run, what it showed, and what it means for the study. Everything here is exploratory: one seed, one bias category (gender), and no correction for multiple comparisons.

**Short version.** In both runs, the bias vector never lowered stereotyped answers at full precision (BF16) more than a random direction did without hurting accuracy. With no working effect at full precision, there is nothing for quantization to preserve, so the quantization question cannot be answered yet with this vector on this data.

## Runs at a glance

| Run | Date (UTC) | Model | What ran | Result archive |
|---|---|---|---|---|
| Run 1: main campaign | 2026-10-31 | Qwen2.5-7B-Instruct | BF16, INT8, NF4; layer 13; strengths −0.05 to +0.1; validation, test and capability | `steering-results-20261031T081615Z.zip` |
| Run 2: dose finding | 2026-11-03 | Qwen2.5-7B-Instruct and Gemma-2-9B-IT | BF16 only; 3 layers per model; strengths ±0.1 to ±0.6; validation only | `dose-results-20261103T214804Z.zip` |

Both runs use the same gender BBQ split (`data/prepared/gender`): 1,240 validation and 1,240 test questions, half with an answer in the context and half without. The bias vector is the mean difference between the correct answer and a wrong named answer, built from 128 training examples (`src/steerquant/steering.py`, `src/steerquant/data.py:225`). Strength is a fraction of the layer's mean activation norm.

## Checks that passed

- **Offline tests:** 33 passed before run 1, and 49 before run 2. The 49 include a tiny Gemma2 model, its stop tokens, and the run 2 pass rule.
- **Small GPU check** (run 1): loading, hooks, answer parsing and stop tokens worked at BF16, INT8 and NF4.
- **Gemma access:** the token could read `google/gemma-2-9b-it`.
- **Answer format:** unsteered answers were valid. Invalid rates were 0.0–0.7% in run 1 and 0% for both models in run 2, with no truncation.
- **Reproducibility:** Qwen's unsteered BF16 validation scores in run 2 match run 1 exactly. Ambiguous accuracy was 0.984, disambiguated accuracy 0.660, and invalid answers 0.000.

## Run 1: Qwen, BF16, INT8 and NF4

**Steering did almost nothing.** At strengths up to 0.1, the bias vector changed at most 7 of 1,240 validation answers. The raw vector's norm is 3.72, against a mean activation norm of 51.4, so these doses were tiny. Frozen selection picked strength 0.01 at BF16, which is effectively no steering.

**The vectors survived quantization.** Each quantized model's own vector pointed almost the same way as the BF16 vector: the cosine similarity was 0.992 for INT8 and 0.962 for NF4.

**INT8 made the unsteered model more cautious.** Here are the unsteered accuracy scores on questions with an answer in the context:

| Split | BF16 | INT8 | NF4 |
|---|---:|---:|---:|
| Validation | 0.660 | 0.642 | 0.671 |
| Test | 0.877 | 0.829 | 0.887 |

On test, INT8 is 4.8 points below BF16, with a 95% interval of −12.9 to −0.2. INT8 shifted answers toward "unknown". NF4 did not.

**The prompt instruction was the only intervention that worked.** On Qwen validation, it cut stereotyped answers from 37 to 17 (3.0% to 1.4%) and raised disambiguated accuracy from 0.660 to 0.729.

**Capability was intact.** These are the unsteered accuracy scores:

| Task | BF16 | INT8 | NF4 |
|---|---:|---:|---:|
| MMLU (1,000) | 0.700 | 0.694 | 0.699 |
| SQuAD, answerable (500) | 0.954 | 0.946 | 0.948 |
| SQuAD, unanswerable (500) | 0.456 | 0.486 | 0.412 |

**Conclusion of run 1:** the doses were too small to test anything. Run 2 was set up to check whether stronger doses help.

## Run 2: BF16 dose finding on Qwen and Gemma

Each model ran 50 conditions on validation:
- 3 layers × 8 strengths × 2 sources (the bias vector and a random direction);
- plus the unsteered baseline and the prompt instruction.

The layers were 8, 13 and 20 for Qwen, and 12, 20 and 28 for Gemma, at about 30%, 50% and 70% depth.

### The pass rule

The pass rule was fixed before running (`docs/dose-finding.md`, `src/steerquant/dose.py`). A setting passes only if all of these hold:
- **It lowers stereotyped answers** by at least 1 point.
- **It beats random.** The 95% interval of the bias vector minus random, at the same layer and strength, is below zero.
- **It stays within the limits:**
  - both accuracies drop by at most 2 points;
  - unnecessary "unknown" answers rise by at most 2 points;
  - invalid and truncated answers each rise by at most 1 point.

### Verdicts

| | Qwen | Gemma |
|---|---|---|
| **Verdict** | **FAIL**, 0 of 24 settings | **FAIL**, 0 of 24 settings |
| Unsteered stereotyped answers | 37 of 1,240 (3.0%) | 24 of 1,240 (1.9%) |
| Unsteered accuracy, ambiguous / disambiguated | 0.984 / 0.660 | 0.992 / 0.837 |
| Unsteered unnecessary "unknown" | 26.6% | 10.2% |
| Best bias-vector setting | layer 13, +0.6: 37 → 18 stereotyped | layer 20, +0.6: 24 → 12 stereotyped |
| Why that setting failed | disambiguated accuracy 0.660 → 0.605; "unknown" 26.6% → 35.6%; not better than random | disambiguated accuracy 0.837 → 0.719; "unknown" 10.2% → 25.0% |
| Prompt instruction | 37 → 17 stereotyped; accuracy up | 24 → 21 stereotyped |

The full tables are in `reports/dose-qwen/verdict.md` and `reports/dose-gemma/verdict.md` inside the archive.

### What run 2 showed

1. **A low dose is no longer the explanation.** Strong settings change many answers. For example, Gemma at layer 28, −0.6 changed 325 of 1,240, and Qwen at layer 13, +0.6 changed 74.
2. **The vector mostly controls how often the model says "unknown".** Wherever it lowered stereotyped answers, it did so by turning answers into "unknown", including correct ones: 75 on Gemma at layer 20, +0.6, and 41 on Qwen at layer 13, +0.6.

   This follows from how the vector is built:
   - In the questions without an answer in the context, the correct answer is always "unknown", so half the training signal is "say unknown".
   - In the questions with an answer, the correct answer names the stereotyped group about half the time, so the stereotype signal largely cancels out.
3. **There is very little gender bias to remove.** Both models answer "unknown" correctly on 98–99% of the questions without an answer. On the questions with an answer, correct answers are only 6.1 points more frequent for Qwen, and 2.3 points for Gemma, when the right answer fits the stereotype than when it goes against it.
4. **The two settings that beat random look like chance.** On Gemma, layer 12 at −0.6 and layer 20 at +0.6 beat random, but layer 12 lowered stereotyped answers by only 0.4 points, and layer 20 broke the accuracy and "unknown" limits. With 48 comparisons, about one such result is expected by chance.

### Side observations

These came from a breakdown after the run, so they are not tested by the pass rule.
- **On Gemma, layer 20 at −0.3 made the model more accurate.** Disambiguated accuracy rose 0.837 → 0.879, because the model said "unknown" less often (10.2% → 5.8%). Stereotyped answers did not change, and random at the same setting gave 0.830.
- **Gemma sometimes answers with the letter and its text**, such as "B. Can't answer". The strict parser counts that as invalid. This accounts for the 23 invalid answers under Gemma's prompt instruction, and for most of the 325 changes at layer 28, −0.6.

## Conclusions

1. **The original question can't be answered yet.** The current vector has no useful effect at full precision on gender BBQ, for either model, so INT8 and NF4 runs would only measure noise. Under the pass rule, the study stops before quantization.
2. **The vector is an abstention ("say unknown") direction, not a bias direction.** That is a property of how it was built, not of the models.
3. **INT8 alone shifts Qwen toward "unknown"** (run 1). This effect is real, but separate from steering.
4. **The prompt instruction is a stronger bias reduction than any vector setting tried** on Qwen.

## Next step: run 3

Run 3 is built but has not run yet. Its plan and rules are fixed in `docs/run3.md`:
1. **A stereotype-specific vector.** It contrasts the other named answer with the stereotype-consistent one, and leaves "unknown" out.
2. **Dose tests on Race and Gender,** with the same grid and pass rule as run 2. Only these two categories have enough validation scenarios (10 and 11) for the scenario bootstrap.
3. **A descriptive bias scan** of the other six BBQ categories. They have 3–5 validation scenarios each.

There is a fallback study if that fails: the current vector does shift "unknown" answers strongly at BF16, so whether that effect survives quantization is testable. It would need its own pass rule, written before running.

## The Gemma plan from Kazi's handoff

Kazi's `GEMMA_RUNBOOK.md` planned four steps.

| Step | Kazi's plan | Status |
|---|---|---|
| 1. Small GPU check | 16 training examples, 32 questions, layer 20, strength 0.025 | **Covered.** The run 2 tests include a tiny Gemma2 model, and the full Gemma run loaded, generated and stopped correctly. The small check itself was not run separately. |
| 2. BF16 reference effect, then a checkpoint | Layers 12, 20, 28; strengths −0.05 to +0.1; bias vector, native and random; validation only. Stop before quantization if there is no useful BF16 effect | **Done, with stronger doses.** Same model, layers and data (Kazi's `gender-run1` split is byte-identical to ours). Strengths were ±0.1 to ±0.6 instead, because run 1 showed that 0.1 and below barely changes answers. Native equals the bias vector at BF16, so it was not repeated. Result: no useful effect, so the checkpoint says stop. |
| 3. INT8 and NF4 | Run quantized models | **Not run**, by step 2's checkpoint. |
| 4. Selection, test and capability | Freeze settings, evaluate test and capability | **Not run**, by step 2's checkpoint. |

Kazi's weaker strengths (−0.05 to +0.1) were not run on Gemma. They are unlikely to matter: at ±0.1, Gemma's bias vector changed only 2–19 answers, and stronger doses still failed.
