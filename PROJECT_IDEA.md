# Project idea: does quantization preserve bias-reduction steering?

## Motivation

Activation steering changes an LLM's behavior by adding a vector to an intermediate hidden state, without updating the model's weights. Quantization reduces the numerical precision used to store/run a model. They are often combined in practice, but preserving ordinary model accuracy does not necessarily preserve the model's response to an activation intervention.

Our study asks whether a vector that improves evidence-based demographic answering in the original model remains useful after quantization. We will distinguish a changed baseline, a changed steering direction, and a changed response to an otherwise similar direction.

This is an empirical question. We do not assume that quantization must make steering weaker, or that a vector with a high cosine similarity must produce the same behavior.

## First task: bias reduction

We chose bias reduction instead of sentiment or sycophancy. Our operational target is:

> Reduce unsupported demographic stereotyping while preserving the ability to answer from evidence.

The first benchmark is the gender-related subset of **BBQ**, an English QA benchmark grounded primarily in U.S. social contexts. BBQ has two useful evidence conditions:

| Evidence in the context | Desired response |
|---|---|
| Insufficient to identify the answer | Choose the explicit unknown/cannot-determine option |
| Sufficient to identify the answer | Choose the evidence-supported person/group, including when it is the stereotyped group |

An intervention that always refuses or always chooses unknown has not achieved our objective. A lower aggregate bias score alone is not enough: opposite group-level effects can cancel.

## Main questions

1. **Transfer:** Does a BF16-extracted vector retain its effect in INT8/NF4 deployment?
2. **Recovery:** If its effect changes, can strength recalibration or extracting a vector again at the receiver's precision recover useful control?
3. **Specificity and utility:** Does apparent debiasing reflect evidence use, excessive uncertainty, invalid output, or general capability loss?
4. **Geometry:** Does vector cosine similarity track its functional effect?

## First experiment

Model: `Qwen/Qwen2.5-7B-Instruct`, with the exact model commit locked before execution.

| Receiver | Unsteered | BF16 vector transferred | Vector extracted at receiver precision |
|---|---|---|---|
| BF16 | Yes | Reference steering | Same reference vector |
| bitsandbytes LLM.int8() | Yes | Yes | Yes |
| bitsandbytes NF4 | Yes | Yes | Yes |

Each receiver also has a prompt-only control and norm-matched random-vector controls. The same original-model activation scale is used across receivers and vector sources.

The first campaign compares weight-quantized deployment schemes. LLM.int8() uses mixed-precision outlier handling and internally quantized matmul inputs, so it is not a pure uniform W8A16 control. NF4 and INT8 also differ in algorithm. We must not attribute every difference solely to the number of bits. Separately controlled activation quantization belongs to the next milestone.

## How the steering direction is constructed

Use contrastive mean differences, with gold-respecting answers:

- Under ambiguity, compare the correct unknown response against incorrect named attributions.
- With sufficient evidence, compare the gold answer against the other named attribution. Never label an evidence-supported stereotyped-group answer as wrong merely because of group identity.
- Rotate answer positions over A/B/C to reduce letter-position artifacts.
- Average within each example and then balance the two evidence conditions.
- Unit-normalize the resulting vector.

This direction may capture evidence following or uncertainty as well as social bias. We therefore describe it as an evidence-based answering direction until controls establish a more specific interpretation.

Intervention at zero-based decoder block output 13:

`h_last <- h_last + strength × original_activation_norm × unit_vector`

The last prompt token is modified during prefill, so the first answer can change. During cached generation, the current last token is modified at each step.

## Data and tuning

The same scenario/template group never crosses training, validation, or test. Grouping is by category and BBQ `question_index`. Groups are stratified by identity type (proper names, gender terms, trans/non-trans), so all three splits have a similar identity mix. Current gender splits contain 27 training, 11 validation, and 12 test scenarios (3,192/1,240/1,240 questions), with many variants per scenario. Effective independent sample size is therefore much smaller than the raw question count. About five 200-question name scenarios dominate each held-out split.

We use training examples to extract vectors, validation to sweep strengths and choose settings, and the untouched test split to measure those frozen settings. Test execution requires the saved selection file. The main evaluation includes both:

- Reference settings copied unchanged to quantized receivers.
- Strengths recalibrated separately using each receiver's validation performance.

The provided strength grid and accuracy-loss budgets are exploratory pilot choices. They require justification before a confirmatory campaign.

## Outcomes and interpretation

Report accuracy separately for ambiguous and disambiguated questions, unsupported stereotypical errors, unnecessary unknown answers, invalid outputs, token-cap truncation, subgroup-pair results, signed BBQ scores, and individual answer transitions.

Compare each intervention against that receiver's own unsteered baseline and retain absolute scores. Quantization can shift the baseline before steering is applied.

Reports provide paired confidence intervals by resampling whole scenario groups. They are conditional on one extraction seed and are not equivalence tests, multiplicity-corrected significance tests, or pooled model-level evidence. Repeat across seeds/categories/models before generalizing.

## What would be interesting?

- Similar vectors but different behavioral effects: geometric similarity does not establish preserved controllability.
- Changed raw transfer but successful strength recalibration: deployment calibration matters.
- Native extraction succeeds where original-vector transfer fails: extraction and application interact with quantization.
- Lower signed bias accompanied by unnecessary unknown/invalid answers: apparent improvement is not sufficient evidence of useful debiasing.
- Similar aggregate bias but different affected examples/groups: mean metrics hide redistribution of errors.
- Little measurable change across schemes: preservation is informative, but equivalence requires a predefined margin and an adequately powered analysis.

## Next stages

After the first gender pilot: repeat extraction across seeds; add age/race and another model family; define a sufficiently large held-out capability and neutral-uncertainty suite; add an independent generic-uncertainty steering control. Then extend to controlled activation quantization and matched hidden-state downstream-response experiments.

The first GPU execution is a pipeline/scientific pilot, not a completed publishable study.

## Relationship to existing work

We inspected *Shifting Perspectives* and chose to write the experimental core independently, using it as a methodological/reproduction reference. Our core question is quantization's effect on usable bias-reduction steering, not whether activation steering can mitigate bias at all.

Relevant sources:

- [BBQ](https://arxiv.org/abs/2110.08193) — benchmark, evidence conditions, and bias measurements.
- [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) — mean-difference steering method.
- [Shifting Perspectives](https://arxiv.org/abs/2503.05371) — bias-mitigation steering reference; its primary method uses PCA.
- [Steering Under Compression](https://arxiv.org/abs/2609.06473) — closely related weight-quantization work; the paper leaves vector cross-application and activation quantization for further work.
- [Investigating Social Bias Changes in Quantized Language Models](https://arxiv.org/abs/2602.06181) — quantization can change affected examples/groups while aggregate bias scores remain similar.

The exact novelty claim remains subject to a fuller literature review before writing the paper.
