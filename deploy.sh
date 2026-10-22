#!/usr/bin/env bash
# Build + push the image, upload model artifacts, then launch or update the EC2 host.
# Run from the repo root after `python train.py`:   REGION=af-south-1 ./deploy.sh
set -euo pipefail

: "${REGION:?set REGION, e.g. REGION=us-east-1}"
export AWS_DEFAULT_REGION="$REGION"

REPO=insider-threat
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
ALLOWED_CIDR="${ALLOWED_CIDR:-$(curl -s https://checkip.amazonaws.com)/32}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="insider-threat-data-${ACCOUNT_ID}"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:latest"

# --- image -------------------------------------------------------------------
aws ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" >/dev/null
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"

# --- model artifacts (gitignored, so they ship via S3) -------------------------
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = us-east-1 ]; then
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" \
      --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
fi
aws s3 sync data/artifacts "s3://${BUCKET}/artifacts" --delete \
  --exclude "*" --include "*.pkl" --include "meta.json"

# --- what the instance runs on every deploy -----------------------------------
RUN="set -e
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY
docker pull $IMAGE
mkdir -p /opt/insider/artifacts
aws s3 sync s3://$BUCKET/artifacts /opt/insider/artifacts --region $REGION --delete
docker rm -f insider 2>/dev/null || true
docker run -d --name insider --restart unless-stopped -p 8501:8501 \
  -v /opt/insider/artifacts:/app/data/artifacts:ro $IMAGE"

# --- host --------------------------------------------------------------------
ID="$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$REPO" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)"

if [ "$ID" = None ]; then
  VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' --output text)"
  SG="$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$REPO" "Name=vpc-id,Values=$VPC" \
    --query 'SecurityGroups[0].GroupId' --output text)"
  if [ "$SG" = None ]; then
    SG="$(aws ec2 create-security-group --group-name "$REPO" --vpc-id "$VPC" \
      --description "insider-threat dashboard" --query GroupId --output text)"
    aws ec2 authorize-security-group-ingress --group-id "$SG" \
      --protocol tcp --port 8501 --cidr "$ALLOWED_CIDR" >/dev/null
  fi
  AMI="$(aws ssm get-parameter \
    --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --query Parameter.Value --output text)"
  USERDATA="#!/bin/bash
dnf install -y docker
systemctl enable --now docker
$RUN"
  ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --security-group-ids "$SG" --iam-instance-profile Name=insider-threat-ec2 \
    --metadata-options HttpTokens=required --user-data "$USERDATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$REPO}]" \
    --query 'Instances[0].InstanceId' --output text)"
  aws ec2 wait instance-running --instance-ids "$ID"
else
  PARAMS="$(python3 -c 'import json,sys; print(json.dumps({"commands":[sys.stdin.read()]}))' <<<"$RUN")"
  aws ssm send-command --instance-ids "$ID" --document-name AWS-RunShellScript \
    --parameters "$PARAMS" >/dev/null
fi

IP="$(aws ec2 describe-instances --instance-ids "$ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "Dashboard: http://${IP}:8501  (allow a few minutes for first boot; open to ${ALLOWED_CIDR} only)"
