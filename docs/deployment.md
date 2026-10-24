# Deployment

How the dashboard is packaged and how the live site at [cos700research.tebogoyungmercykay.co.za](https://cos700research.tebogoyungmercykay.co.za) is put together.

## What runs where

The dashboard runs in a Docker container built from the [`Dockerfile`](../Dockerfile). The image starts from `python:3.14-slim`, installs `requirements.txt` with the pinned versions in `constraints.txt`, copies the repository, and serves Streamlit headless on port 8501. The image carries the code, `results/` and the report figures, which the Findings page reads.

The trained models are not in the image or in git. `python train.py` writes them to `data/artifacts/` (the `*.pkl` files and `meta.json`), which is gitignored, so they travel separately and are mounted into the container read-only at `/app/data/artifacts`.

## Run the container locally

Train first, then build and run the image from the repository root.

```bash
python train.py
docker build -t insider-threat .
docker run --rm -p 8501:8501 -v "$PWD/data/artifacts:/app/data/artifacts:ro" insider-threat
```

The dashboard is then at http://localhost:8501.

## Deploy to AWS with deploy.sh

[`deploy.sh`](../deploy.sh) builds the image, ships the models and starts or updates one EC2 host. It needs the AWS CLI configured with credentials that can use ECR, S3, EC2 and SSM (set `AWS_PROFILE` to the profile you deploy with), Docker, the trained artifacts in `data/artifacts/`, and an IAM instance profile named `insider-threat-ec2` that lets the instance pull from ECR, read the S3 bucket and receive SSM commands.

```bash
REGION=<your-region> ./deploy.sh
```

Two optional variables change its defaults. `INSTANCE_TYPE` defaults to `t3.medium`, and accounts on the AWS free plan can only launch free-tier types, so the live site runs on a `c7i-flex.large`. `ALLOWED_CIDR` defaults to your current public IP, which is the only address the new security group admits on port 8501.

Each run does the same five things.

1. Creates the ECR repository `insider-threat` if it does not exist, then builds the image for `linux/amd64` and pushes it.
2. Creates the bucket `insider-threat-data-<account id>` if needed and syncs the `*.pkl` files and `meta.json` from `data/artifacts/` into it.
3. Looks for a running instance tagged `insider-threat`.
4. If there is none, it creates the security group if needed, launches Amazon Linux 2023 with user data that installs Docker, pulls the image and the artifacts and starts the container with `--restart unless-stopped`.
5. If the instance already exists, it sends the same pull, sync and restart commands through SSM, so a redeploy needs no SSH.

It prints the instance's address when it finishes. Allow a few minutes on the first boot.

## The live site

The live dashboard adds three pieces that `deploy.sh` does not create, set up once by hand. CloudFront sits in front of the instance and serves HTTPS with a certificate from AWS Certificate Manager, the domain points at the CloudFront distribution with a CNAME record, and the instance's security group admits port 8501 only from CloudFront's managed prefix list, so the instance cannot be reached directly.

If the instance is stopped and started, its public DNS name changes, and the CloudFront origin must be updated to the new name. `deploy.sh` does not do this.

## Taking it down

Terminate the instance, then delete the CloudFront distribution, the security group, the S3 bucket and the ECR repository, so nothing keeps running or billing.
