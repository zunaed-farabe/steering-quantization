# Capability / neutral-uncertainty QA

`qa.jsonl` contains 2,000 held-out, demographically neutral multiple-choice questions. They are evaluated only at the settings frozen on BBQ validation, to check whether steering costs general ability or makes the model abstain where it should answer.

| `category` | Questions | Groups (bootstrap unit) | Options | Correct answer |
|---|---|---|---|---|
| `mmlu` | 1,000 | 52 MMLU subjects, 17–20 each | 4, original order | The MMLU key |
| `squad_answerable` | 500 | 28 SQuAD articles, 17–18 each | 3 | The annotated span; "Cannot be determined" is an unnecessary abstention |
| `squad_unanswerable` | 500 | 28 SQuAD articles, 17–18 each | 3 | "Cannot be determined" |

## Construction (`scripts/build_capability_qa.py`, seed 42)

- **MMLU** test split, from the authors' original archive. Excluded subjects about social attitudes and identity: human_sexuality, moral_disputes, moral_scenarios, sociology, world_religions. Duplicate questions (same text and options, sometimes listed under two subjects) keep only their first copy. Questions are spread evenly over the remaining 52 subjects.
- **SQuAD 2.0 dev.** Excluded articles centred on ethnic, religious or social groups: Black_Death, Economic_inequality, Huguenot, Imperialism, Islamism, Normans, Yuan_dynasty.
  - Answerable questions offer the most common annotated span, a distractor and "Cannot be determined".
  - Unanswerable questions offer SQuAD's plausible-but-wrong span, a distractor and "Cannot be determined". The 15 unanswerable questions without a plausible span are skipped.
  - The distractor is a span answering another question in the same passage that does not overlap any answer variant.
  - All six option orders are cycled, so the correct letter and the "Cannot be determined" position are balanced over A/B/C.
- **Neutrality filter (both sources):** a question is dropped if its question or options match the demographic-term pattern in `manifest.json`. Passages can still mention people; the questions are not about demographic groups.
- Each row has `id`, `group`, `category`, `question`, `answers`, zero-based `gold`, and for SQuAD a `context` and the zero-based `unknown` option index.

The script downloads the sources into `data/downloads/` (not committed) and verifies their pinned SHA-256 hashes. Re-running it reproduces `qa.jsonl` byte for byte; its hash is in `manifest.json`.

## Licenses and attribution

- MMLU: Hendrycks et al., *Measuring Massive Multitask Language Understanding* (ICLR 2021). MIT license, see `LICENSE-MMLU.txt`.
- SQuAD 2.0: Rajpurkar, Jia and Liang, *Know What You Don't Know: Unanswerable Questions for SQuAD* (ACL 2018). CC BY-SA 4.0. The `squad_*` rows are adapted from it and carry the same license.

## Caveats

Both benchmarks are public and may appear in Qwen's training data. That inflates absolute scores but matters less here, because the study compares each steered condition with the same model's unsteered baseline. SQuAD 2.0 unanswerable questions are adversarial, so the plausible span can look right. The comparison of interest is the change in accuracy and abstention, not the absolute level.
