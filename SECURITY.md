# Security Policy

![Title slide of the project presentation](docs/assets/readme.png)

> Responsible disclosure for the per-user insider threat detection research prototype.

## Scope

This repository is a university research prototype. The code in it, the Docker image built from it, [`deploy.sh`](deploy.sh) and the live dashboard at [cos700research.tebogoyungmercykay.co.za](https://cos700research.tebogoyungmercykay.co.za) are in scope. The dashboard serves models trained on the CERT Insider Threat Test Dataset r6.2, which is synthetic, so it holds no real personal data. The dataset itself is published by Carnegie Mellon University and is out of scope here.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report them privately through GitHub instead, from the **Security** tab of this repository, using **Report a vulnerability**. Only the maintainer can see the report.

## What to Include

- A description of the vulnerability and its potential impact
- Steps to reproduce it, or a proof of concept if it is safe to share
- The affected file, component or page of the dashboard
- Any mitigation or fix you have in mind

## What to Expect

This is a one-person student project, so responses are best effort. You can expect an acknowledgement within a week and an update once the issue has been assessed. Valid issues are fixed on `master` and redeployed, and you are credited in the fix if you wish.

## Keeping Secrets Out

Never commit AWS credentials, keys, `.env` files or other secrets. The trained models and the dataset are gitignored and reach the server through S3, so nothing in the repository needs credentials to build or run.

## Safe Harbour

Good-faith research that follows this policy and avoids privacy violations, data destruction and service disruption is welcome. Please do not load test the live dashboard or try to deny service to it, since it runs on a single small instance.

---

_COS700 Research Project, University of Pretoria, 2026._
