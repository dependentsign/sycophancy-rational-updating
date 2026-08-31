# Third-party data notices

## TruthfulQA

The benchmark questions and reference answers are adapted from [TruthfulQA](https://github.com/sylinrl/TruthfulQA), released by its authors under the Apache License 2.0.
The upstream license text is included at
[`licenses/TruthfulQA-LICENSE.txt`](licenses/TruthfulQA-LICENSE.txt).

## PopQA

The PopQA questions, entity identifiers, relations, answers, and aliases are
adapted from the official [Adaptive Retrieval repository](https://github.com/AlexTMallen/adaptive-retrieval),
which is distributed under the MIT License. The bundled file is a deterministic
2,000-row subset of the 14,267-row PopQA release.
The upstream license and copyright notice are included at
[`licenses/PopQA-LICENSE.txt`](licenses/PopQA-LICENSE.txt).

## EX-FEVER

The claims, verdicts, entity metadata, and author-written explanations are
adapted from the official [EX-FEVER repository](https://github.com/dependentsign/EX-FEVER),
which is distributed under the MIT License. The bundled file contains the exact
2,000 SUPPORT/REFUTE qids used by the paper. The upstream license and copyright
notice are included at
[`licenses/EX-FEVER-LICENSE.txt`](licenses/EX-FEVER-LICENSE.txt).

## AQuA-RAT

The algebra questions, answer choices, labels, and annotator-written rationales
are adapted from the official [AQuA-RAT repository](https://github.com/google-deepmind/AQuA),
licensed under the Apache License 2.0. The bundled snapshot contains the 254-row
validation release and 247 schema-valid rows from the test release. The
upstream license notice is included at
[`licenses/AQuA-LICENSE.txt`](licenses/AQuA-LICENSE.txt).

## Wikipedia reference notes

The evaluation data include reference notes constructed from English Wikipedia
text. TruthfulQA rows record supporting-source metadata in `evidence_sources`;
PopQA rows record the subject page title and URL in `evidence_wiki_title` and
`evidence_wiki_url`.

Wikipedia text is made available under Creative Commons Attribution-ShareAlike terms and, unless otherwise indicated, the GNU Free Documentation License. See the [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) and the license notice on each linked source page.

## What is licensed under what

This repository is not under a single license, because the data is not ours to
relicense.

| Part | License |
|---|---|
| The code in `sru/`, and everything else written for this repository | MIT, see [`LICENSE`](LICENSE) |
| Questions, answer choices, gold labels, claims, and annotator rationales carried over from the four source datasets | their upstream licenses, above: Apache-2.0 for TruthfulQA and AQuA, MIT for PopQA and EX-FEVER |
| Text derived from Wikipedia: the TruthfulQA reference notes and the quotes in `evidence_sources`, and the PopQA lead paragraphs in `evidence` | CC BY-SA 4.0 |

The Wikipedia-derived text is the reason the repository cannot simply be called
MIT. CC BY-SA 4.0 is a share-alike license: anyone redistributing that text, or
a work derived from it, has to do so under CC BY-SA 4.0 as well, with
attribution and an indication of what was changed. Each row records where its
text came from, in `evidence_urls` and `evidence_sources` for TruthfulQA and in
`evidence_wiki_title` and `evidence_wiki_url` for PopQA.

If you build on this data, keep the attribution fields with it.
