# What Breaks in Open-Source Games? Reproduction code

`reproduce.py` recomputes every number in the paper *What Breaks in Open-Source Games? Symptom Categories of Reported Bugs in OSSGameBench and Testing Strategies to Detect Them*. It covers the whole pipeline, from the raw OSSGameBench tables to the last table. Each block of output is headed with the paper section or table it supports.

## 1. Environment

The script was run with Python 3.11.9 on Linux and needs these three packages:

    python3.11 -m venv .venv && . .venv/bin/activate
    pip install pandas==3.0.1 numpy==2.2.6 pyarrow==20.0.0

## 2. Data

Download `OSSGameBench.zip` from the Dropbox link given in the course assignment. Zenodo record 18359123 holds only the authors' scripts and appendix, not the data. The snapshot used here has these checksums:

- outer zip: 65,816,582,051 bytes, SHA-256 `12ca0bd412a6cf842d813a33cd754c623b984b327e34acac1a0ba32d564940b2`;
- inner zip: SHA-256 `9653dfed70ccab6e949b6960cb4690427a017c1ea575d08e1979bec7f860dd38`.

The archive contains a nested zip, so unzip twice. The commit tables are not needed.

    unzip OSSGameBench.zip -d data/raw
    unzip data/raw/OSSGameBench.zip 'OSSGameBench/*.parquet' -x 'OSSGameBench/commits/*' -d data/tables

The script reads four tables: `project.parquet`, `issue.parquet`, `pr.parquet` and `issue_pr.parquet`.

## 3. Run

    python -I reproduce.py data/tables/OSSGameBench

It takes under a minute and writes nothing to disk. It runs these steps in order:

| Step | Paper |
| --- | --- |
| Dataset counts, and the 20 duplicate-copy repositories that the dataset authors also exclude | 3.1 |
| Bug rules: label rule, merged-pull-request rule, and the seeded 200-issue audit sample | 3.2 |
| Analysis sets: at least 20 bug issues per repository, at most 200 sampled per repository | 3.2 |
| Keyword classifier, with a check that its rules match the SHA-256 recorded when they were frozen; development and held-out samples | 3.3 |
| Repository types: manual labels, blind second rater and topic rule | 3.4 |
| Rule precision with Wilson intervals; Cohen's κ for the model raters and the human check; classifier accuracy | 4.1, Table 1 |
| Pooled and median per-repository symptom shares | 4.2, Table 2 |
| Tool-minus-game differences with a repository bootstrap (5,000 resamples), Holm-adjusted p and Bonferroni 99% intervals, under the primary analysis and three sensitivity analyses | 4.3, Table 3 |
| Correction for classifier error (Algorithm 1) | Tables 2 and 3 |
| Checks on the fallback category: keyword-matched issues only, a Fisher exact test on fallback precision, and reweighting by text length | 4.3 |
| Adjustment for programming language (least squares with language-group dummies) | 4.3, Table 3 |
| How many issues carry a label naming their symptom | 4.4 |

All random draws use seed 42, so repeated runs print identical numbers.

## 4. Labels

The manual labels cannot be recomputed. They are embedded at the top of `reproduce.py` as strings, one character per item, in the order of the seeded samples:

- the bug audit, from two model raters;
- the development and held-out symptom labels, and a blind second rater's held-out labels;
- the game/tool type of all 391 repositories, and a blind second rater's types;
- the author's human check of 20 audit issues and 30 repositories.

Before using any label set, the script checks that its seeded draw selects exactly the labelled items, by comparing a SHA-256 of the issue or repository keys. Section 3.6 of the paper states which model or person produced each label set and what each rater could see.
