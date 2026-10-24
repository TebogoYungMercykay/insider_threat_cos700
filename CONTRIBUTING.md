# Contributing

![Title slide of the project presentation](docs/assets/readme.png)

> Thank you for your interest in improving the per-user insider threat detection framework.

Bug reports, questions and suggestions all help, and this guide explains how to make them count.

## Before You Start

This repository is part of an assessed University of Pretoria project (COS700, 2026). Issues and suggestions are welcome at any time, but code changes from others may not be merged until the project has been assessed, because the work is examined as the author's own.

## Reporting Issues

Open a GitHub issue that says what you ran, what you expected and what happened. Include the command, which dataset it ran on (the full population or the 400-user cohort) and the relevant part of the console output. Security issues go through the [Security Policy](SECURITY.md) instead.

## Making a Change

1. Branch from `develop`, as `feat/...`, `fix/...` or `docs/...`.
2. Follow the [Coding Standards](docs/coding-standards.md), in particular its correctness rules.
3. Run `pytest -m "not slow"` and make sure every test passes. Add a regression test for any bug you fix.
4. If the change affects results, regenerate them with the scripts in [Running the Experiments](docs/running-the-experiments.md), never by editing `results/`, and say which tables changed.
5. Open a pull request into `develop` with a one-line summary and a few bullets of changes. `develop` is merged into `master` after review.

New to pull requests? See [GitHub's guide to pull requests](https://docs.github.com/en/pull-requests).

## Commit Messages

Use a conventional prefix, `feat:`, `fix:`, `docs:`, `chore:`, `test:` or `refactor:`, followed by a short imperative description in lower case, for example `docs: use plain punctuation instead of dashes`.

## Community Guidelines

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Help

Open a GitHub issue with your question. The [Setup Guide](docs/setup-guide.md) and its troubleshooting section answer the most common ones.

---

_COS700 Research Project, University of Pretoria, 2026._
