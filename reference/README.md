# Published rates

`paper_baselines.json` and `paper_baselines.csv` hold the rates for the four
backbones in the paper, on both splits, computed from the paper's own runs.

The `cal` rows reproduce Table 1 of the paper exactly. `test` rows are the
held-out split, which is what `sru run` uses by default.

When you evaluate one of these four models, `sru run` adds an "Against the
paper" section to the report comparing your numbers with these.

| Column | Meaning |
|---|---|
| `acc` | single-turn accuracy under Baseline |
| `r_uy` | Unsupported-Yielding rate, lower is better |
| `r_ru_evidence` | Rational-Updating rate under Evidence, higher is better |
| `r_ru_user_evidence` | Rational-Updating rate under User-Evidence |
| `n` | items scored |
| `n_uy_denom` | items correct at baseline, the R_UY denominator |
| `n_ru_denom` | items wrong at baseline, the R_RU denominator |

## Expect small differences

Re-running a model will land close but rarely identical. Batching changes the
order of floating-point reductions, GPU and library versions differ, and a
near-tie between two options can fall either way.

These runs were reproduced end to end on 3090s with a newer transformers than
the paper used, to see how large that drift actually is:

| Model | Dataset | Acc | R_UY | R_RU^E | R_RU^UE |
|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | TruthfulQA | 37.5 / 36.7 | 60.0 / 59.1 | 12.0 / 14.5 | 21.3 / 19.7 |
| Llama-3.1-8B-Instruct | TruthfulQA | 44.2 / 44.2 | 43.4 / 45.3 | 16.4 / 17.9 | 19.4 / 19.4 |
| Qwen3-8B | TruthfulQA | 35.0 / 35.0 | 16.7 / 16.7 | 7.7 / 7.7 | 3.9 / 3.9 |
| gemma-3-4b-it | TruthfulQA | 32.5 / 33.3 | 48.7 / 50.0 | 19.8 / 21.2 | 12.3 / 13.8 |
| Llama-3.2-3B-Instruct | EX-FEVER | 61.2 / 60.7 | 97.7 / 96.9 | 90.0 / 86.8 | 61.6 / 58.5 |
| Llama-3.2-3B-Instruct | AQuA | 55.9 / 55.1 | 92.8 / 83.8 | 88.1 / 95.5 | 80.7 / 86.5 |

Yours on the left, published on the right, `test` split throughout. The
calibration split behaves the same way: Llama-3.2-3B-Instruct on
TruthfulQA with `--split cal` gives 39.5 / 52.9 / 17.8 / 18.1 against the
paper's 39.7 / 51.6 / 17.1 / 18.1.

How close to expect depends on how the answer is read back, and the spread
above is wide enough to be worth understanding before you read your own run.

TruthfulQA scores whole options by log-likelihood, so only a near-tie can flip
an item. Every TruthfulQA number above is within 2.5 points, and Qwen3-8B
matched on all four.

The other three read an answer out of free-form greedy text, where one different
token early changes everything after it. EX-FEVER stays within about 4 points,
and PopQA did too before its data was repaired (see below). AQuA drifts up to 9, because it is the extreme case of that mechanism:
answers come from up to 1024 tokens of chain-of-thought, a strict extractor
abstains rather than guess a letter when the model never commits, and the two
denominators are only 138 and 109 items, so twelve items moving is nine points.
A single-digit gap on AQuA is ordinary; the same gap on TruthfulQA would not be.

So: read a few points as noise, scale your expectation to the dataset, and treat
a gap far outside these ranges as a real difference in setup. A different scoring
path is not noise at all: `--tqa-scoring letter` measures something the
log-likelihood protocol does not, and its TruthfulQA numbers should not be
compared against this table. The tool leaves that row out of the comparison for
you.

## PopQA has no comparison, on purpose

The published PopQA rates were measured before the answer-hit repair described
in [../data/README.md](../data/README.md), so `sru run` leaves PopQA out of the
comparison rather than showing a gap that means nothing. Twenty-six percent of
the test split is different items now.

The repair is not cosmetic. Re-running the two Llama backbones on the repaired
data against their published rates:

| Model | Acc | R_UY | R_RU^E | R_RU^UE |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 24.4 / 24.0 | 45.9 / 46.2 | 76.3 / 58.5 | 74.2 / 57.0 |
| Llama-3.1-8B-Instruct | 33.2 / 33.6 | 60.5 / 60.7 | 76.7 / 60.4 | 72.3 / 53.8 |

Repaired on the left, published on the right. Accuracy and `R_UY` move by half a
point, because neither condition reads the evidence. Both `R_RU` columns rise by
16 to 18 points on both models, because a quarter of the old sample paired an
item with a passage that never stated its answer. Rational updating was not
possible on those items, and counting them as failures to update held the rate
down. The repaired numbers measure what the condition was meant to measure.

One more source of drift is specific to TruthfulQA. Twenty-five of the 604
reference notes were revised for clarity after these published runs, mostly to
make a note stand on its own without the question in front of it. Only the
Evidence and User-Evidence conditions read the note, so `acc` and `r_uy` are
unaffected, and the reproduction above shows the effect on the two `r_ru`
columns stays inside the ordinary run-to-run drift.
