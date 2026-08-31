# Evaluation data

Four datasets, each cut down to the calibration and held-out test splits used in
the paper, and each paired with a piece of evidence that supports the gold
answer. That pairing is what makes the diagnostic possible: the same item can be
pushed on with nothing (`pressure`) or with genuine support (`evidence`).

| File | Rows | Task | `evidence` is |
|---|---:|---|---|
| `truthfulqa.jsonl` | 604 | MC1, 2-13 options | a reference note written from the Wikipedia pages TruthfulQA cites, with the supporting quotes attached |
| `popqa.jsonl` | 2,000 | long-tail entity QA | the lead paragraph of the subject entity's Wikipedia page |
| `exfever.jsonl` | 2,000 | fact verification | the dataset's own gold `explanation` |
| `aqua.jsonl` | 501 | 5-option math word problems | the annotator-written `rationale` |

Nothing is downloaded or sampled at run time. The files above are the data.

## The data is frozen, and you can prove it

`manifest.json` records a SHA-256 for every data and split file, along with the
upstream repository and the exact revision each was built from. Check it before
you report a number:

```bash
sru verify-data
```

Two people whose checks pass scored the same bytes, the same qids, and the same
calibration/test assignment, which is what makes their numbers comparable. A
failed check tells you which file drifted; restore it from a clean checkout
rather than reporting from it.

## Splits

`splits/*.json` holds the calibration and test qid lists. The same split is
applied to every model, so numbers are comparable across backbones.

| Dataset | Calibration | Test | How the split was drawn |
|---|---:|---:|---|
| TruthfulQA | 484 | 120 | 80/20, stratified |
| PopQA | 1,000 | 1,000 | balanced random halves |
| EX-FEVER | 1,000 | 1,000 | balanced by gold label, then shuffled |
| AQuA | 254 | 247 | the dataset's own validation and test releases |

Reported rates use the test split. The paper's Table 1 reports the calibration
split, so `--split cal` is what reproduces it.

## PopQA evidence integrity

A PopQA item is only usable here if its Wikipedia evidence actually states the
gold answer. An item whose evidence never mentions the answer cannot measure
rational updating: the Evidence condition hands the model a passage that does
not contain what it is supposed to help the model recover.

The shipped snapshot is built from the full 14,267-row PopQA release and keeps
only rows that pass that answer-hit rule, so 1,995 of 2,000 rows contain a
canonical answer or a PopQA alias in their evidence. Each row records which
string matched in `evidence_matched_answer`. An earlier 2,000-row sample passed
the rule on only 1,512 rows, which is the sample the paper's PopQA rates were
measured on.

The consequence for reproduction is in [../reference/README.md](../reference/README.md).

## The evidence is a snapshot

Every piece of evidence here was written or fetched at one point in time, and it
stays fixed at that point. That is what makes the benchmark reproducible, and it
also means the text can carry whatever Wikipedia said back then, mistakes
included.

A real example: the two rows about Borges' short story "The Other" (qids 2332
and 12690) describe him as "Jorge Luis Borges (1901-1975)". He was born in 1899
and died in 1986, and Wikipedia has since been corrected. Both rows ask about
the story's genre and its author, and the evidence still states both correctly,
so the items measure what they were built to measure; the wrong years are an
incidental detail in the same sentence.

Nothing is repaired after the fact, because rewriting evidence would leave text
that matches neither the snapshot the published rates were measured on nor the
page as it reads today. Each row keeps the URL it came from, so anything can be
checked against the current article.

## TruthfulQA evidence

TruthfulQA ships a source URL per question and no usable evidence field, so each
note was written from the cited Wikipedia pages under a prompt that requires
every sentence to be supported by a quote from the page. Numbers, dates, and
rankings that no quote supports are not allowed, and a question whose page cannot
support a note is returned as insufficient rather than filled in from memory. The
full prompt is in the paper's appendix.

The supporting quotes ship with the data, so any note can be checked against its
source:

```jsonc
{"qid": 0,
 "evidence": "Nauru has an area of only 21 square kilometres (8.1 sq mi), ...",
 "evidence_mode": "grounded_synthesis",     // or "extractive"
 "evidence_alignment": "verified_answer",   // what the note establishes
 "evidence_urls": ["https://en.wikipedia.org/wiki/Nauru", ...],
 "evidence_sources": [{"title": ..., "url": ..., "section": ..., "quote": ...}]}
```

`evidence_mode` is `extractive` when the note is verbatim Wikipedia sentences
(394 of 604) and `grounded_synthesis` when quoted facts had to be joined into a
sentence (210). `evidence_alignment` says what the note does for the item:
`verified_answer` supports the gold answer directly (571), `premise_correction`
shows why the question's premise is mistaken (27), and `reference_only` supplies
relevant background without asserting the answer (6).

## Checking it against upstream

The data ships with the repository, so nothing is downloaded or sampled at run
time and everyone scores the same bytes. That does raise a fair question: is
what we ship really what the public datasets contain?

```bash
python scripts/verify_upstream.py
```

This downloads each source file at the revision pinned in `manifest.json`,
checks its SHA-256, and confirms that every question, answer, label, and
rationale we ship appears in it. It needs network access; nothing else in the
tool does.

It reports three things it cannot simply call "identical", because they are
real:

- **Normalisations.** Every TruthfulQA mc1 option carries a sentence-final
  period that upstream does not, applied uniformly so it cannot favour one
  option over another, and some rows keep a subset of upstream's distractors,
  which is why the option count runs from 2 to 13.
- **Locally written text.** Eleven TruthfulQA rows (qids 180, 240, 397, 414,
  501, 521, 544, 672, 703, 737, 742) carry an mc1 option that appears in
  neither upstream file, mostly small edits such as quoting a film title. One
  row, qid 53, is missing its category and best answer; neither field is read
  by the evaluator. One EX-FEVER explanation has a space inserted between two
  run-on sentences. The published rates were measured with all of this in
  place.
- **The evidence.** That is the part we wrote, and it is checked in the
  sections above instead.

## Provenance and licenses

Each file is a filtered, re-keyed subset of a public dataset, with an `evidence`
field added. The underlying datasets keep their own licenses, and the upstream
revision each was built from is pinned in `manifest.json`.

| Dataset | Source | License |
|---|---|---|
| TruthfulQA | [`sylinrl/TruthfulQA`](https://github.com/sylinrl/TruthfulQA), Lin et al. 2022 | Apache-2.0 |
| PopQA | [`AlexTMallen/adaptive-retrieval`](https://github.com/AlexTMallen/adaptive-retrieval), Mallen et al. 2023 | MIT |
| EX-FEVER | [`dependentsign/EX-FEVER`](https://github.com/dependentsign/EX-FEVER), Ma et al. 2024 | MIT |
| AQuA | [`google-deepmind/AQuA`](https://github.com/google-deepmind/AQuA), Ling et al. 2017 | Apache-2.0 |

Evidence text drawn from Wikipedia is CC BY-SA 4.0, which is share-alike: it
cannot be relicensed, and redistributing it means carrying the attribution with
it. Every such row records its source. Full notices and the vendored upstream
license texts are in [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and
[../licenses/](../licenses).

## Row schema

Every row has a `qid` and an `evidence` string. The rest is task-specific:

```jsonc
// truthfulqa
{"qid": 0, "question": "...", "mc1_targets": {"choices": [...], "labels": [1, 0, 0, 0]},
 "best_answer": "...", "evidence": "...", "evidence_sources": [...]}

// popqa
{"qid": 7687, "question": "...", "subj": "...", "prop": "capital of", "obj": "...",
 "possible_answers": ["..."], "evidence": "...", "evidence_matched_answer": "...",
 "evidence_wiki_url": "..."}

// exfever
{"qid": 0, "claim": "...", "gold_bool": false, "gold_str": "False", "evidence": "..."}

// aqua
{"qid": 0, "question": "...", "choice_texts": ["...", "...", "...", "...", "..."],
 "gold_letter": "A", "gold_idx": 0, "evidence": "..."}
```

In TruthfulQA the gold option is stored first in every row, which is why the
letter-scoring path shuffles the options (seeded by `qid`) before showing them.
