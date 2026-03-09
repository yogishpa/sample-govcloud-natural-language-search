# NL Search Chatbot — Deployment Guide

RAG-based natural language search chatbot using FastAPI, Streamlit, Bedrock Knowledge Bases, OpenSearch Serverless, and ECS Fargate.

## Architecture

```
User → ALB (HTTP/HTTPS) → ECS Fargate
                            ├── Backend (FastAPI, port 8000)
                            │     ├── Bedrock KB Retrieve API → OpenSearch Serverless
                            │     └── Bedrock Claude (cross-region inference)
                            └── Frontend (Streamlit, port 8501)

S3 Bucket → Bedrock Knowledge Base → OpenSearch Serverless (vector index)
EventBridge (hourly) → Lambda → KB Sync
```

## Directory Structure

```
dist/
├── backend/          # FastAPI backend service
│   ├── api/          # Routes, models, middleware
│   ├── core/         # Config, logging, security
│   ├── services/     # Search, LLM, session services
│   ├── tests/        # Unit + property-based tests
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── frontend/         # Streamlit chat UI
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── infra/            # CloudFormation templates
│   ├── network-stack.yaml        # VPC, subnets, NAT, SGs, VPC endpoints
│   ├── opensearch-stack.yaml     # OpenSearch Serverless collection
│   ├── knowledgebase-stack.yaml  # Bedrock KB, S3 bucket, sync Lambda
│   ├── ecs-platform-stack.yaml   # ECS cluster, ALB, IAM roles (stable)
│   ├── ecs-services-stack.yaml   # Task defs, Fargate services (deployable)
│   └── lambda/kb_sync.py         # KB sync Lambda source
└── README.md
```

## Stack Deployment Order

Deploy stacks in this order (each depends on outputs from the previous):

1. **network-stack** — VPC, subnets, security groups, VPC endpoints
2. **opensearch-stack** — OpenSearch Serverless vector collection
3. **knowledgebase-stack** — Bedrock KB, S3 data source, sync Lambda
4. **ecs-platform-stack** — ECS cluster, ALB, IAM roles, log groups
5. **ecs-services-stack** — Task definitions, Fargate services

Stacks 1-4 are stable and rarely need updates. Stack 5 is redeployed on every code change.

## Prerequisites

- AWS CLI v2 configured with appropriate credentials
- Docker (for building container images)
- Python 3.12+ (for running tests locally)
- Bedrock model access enabled for:
  - Claude Sonnet 4 (or Claude 3.5 Sonnet v2) via cross-region inference
  - Amazon Titan Text Embeddings V2


## Deployment — Commercial (us-east-1)

### Step 1: Deploy Network Stack

```bash
aws cloudformation create-stack \
  --stack-name nl-search-network \
  --template-body file://infra/network-stack.yaml \
  --parameters ParameterKey=Environment,ParameterValue=commercial \
  --tags Key=Application,Value=nl-search-chatbot
```

Wait for CREATE_COMPLETE (~3-5 min for VPC endpoints).

### Step 2: Deploy OpenSearch Stack

```bash
# Get VPC endpoint ID from network stack
VPCE_ID=$(aws cloudformation describe-stacks --stack-name nl-search-network \
  --query 'Stacks[0].Outputs[?OutputKey==`OpenSearchServerlessEndpointId`].OutputValue' --output text)

aws cloudformation create-stack \
  --stack-name nl-search-opensearch \
  --template-body file://infra/opensearch-stack.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=commercial \
    ParameterKey=VpcEndpointId,ParameterValue=$VPCE_ID \
  --tags Key=Application,Value=nl-search-chatbot
```

Wait for CREATE_COMPLETE (~2 min).

### Step 2b: Create Vector Index

The OpenSearch Serverless collection needs a vector index before the KB can use it. Temporarily enable public access on the network policy, create the index, then revert.

```bash
# Get collection endpoint
COLLECTION_ENDPOINT=$(aws cloudformation describe-stacks --stack-name nl-search-opensearch \
  --query 'Stacks[0].Outputs[?OutputKey==`CollectionEndpoint`].OutputValue' --output text)

# Temporarily allow public access for index creation
# (update the network policy via AWS console or CLI, create index, then revert)

# Create index using Python:
pip install 'opensearch-py<3' requests-aws4auth
python3 -c "
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

session = boto3.Session()
creds = session.get_credentials().get_frozen_credentials()
awsauth = AWS4Auth(creds.access_key, creds.secret_key, 'us-east-1', 'aoss', session_token=creds.token)

host = '${COLLECTION_ENDPOINT}'.replace('https://', '')
client = OpenSearch(hosts=[{'host': host, 'port': 443}], http_auth=awsauth,
                    use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection, timeout=60)

client.indices.create('bedrock-knowledge-base-default-index', body={
    'settings': {'index': {'knn': True, 'knn.algo_param.ef_search': 512}},
    'mappings': {'properties': {
        'bedrock-knowledge-base-default-vector': {
            'type': 'knn_vector', 'dimension': 1024,
            'method': {'engine': 'faiss', 'name': 'hnsw', 'space_type': 'l2',
                       'parameters': {'ef_construction': 512, 'm': 16}}},
        'AMAZON_BEDROCK_TEXT_CHUNK': {'type': 'text'},
        'AMAZON_BEDROCK_METADATA': {'type': 'text', 'index': False}
    }}
})
print('Index created')
"
```

### Step 3: Deploy Knowledge Base Stack

```bash
COLLECTION_ARN=$(aws cloudformation describe-stacks --stack-name nl-search-opensearch \
  --query 'Stacks[0].Outputs[?OutputKey==`CollectionArn`].OutputValue' --output text)

aws cloudformation create-stack \
  --stack-name nl-search-knowledgebase \
  --template-body file://infra/knowledgebase-stack.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=commercial \
    ParameterKey=CollectionArn,ParameterValue=$COLLECTION_ARN \
  --capabilities CAPABILITY_NAMED_IAM \
  --tags Key=Application,Value=nl-search-chatbot
```

**Important**: Update the OpenSearch data access policy to include the KB role ARN as a principal.

### Step 4: Build and Push Docker Images

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

# Create ECR repos
aws ecr create-repository --repository-name nl-search-commercial-backend --image-scanning-configuration scanOnPush=true
aws ecr create-repository --repository-name nl-search-commercial-frontend --image-scanning-configuration scanOnPush=true

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build and push (use --platform linux/amd64 on Apple Silicon)
docker build --platform linux/amd64 -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-commercial-backend:latest backend/
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-commercial-backend:latest

docker build --platform linux/amd64 -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-commercial-frontend:latest frontend/
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-commercial-frontend:latest
```

### Step 5: Deploy ECS Platform Stack

```bash
VPC_ID=$(aws cloudformation describe-stacks --stack-name nl-search-network --query 'Stacks[0].Outputs[?OutputKey==`VpcId`].OutputValue' --output text)
PUB1=$(aws cloudformation describe-stacks --stack-name nl-search-network --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnet1Id`].OutputValue' --output text)
PUB2=$(aws cloudformation describe-stacks --stack-name nl-search-network --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnet2Id`].OutputValue' --output text)
ALB_SG=$(aws cloudformation describe-stacks --stack-name nl-search-network --query 'Stacks[0].Outputs[?OutputKey==`ALBSecurityGroupId`].OutputValue' --output text)
KB_ID=$(aws cloudformation describe-stacks --stack-name nl-search-knowledgebase --query 'Stacks[0].Outputs[?OutputKey==`KnowledgeBaseId`].OutputValue' --output text)

aws cloudformation create-stack \
  --stack-name nl-search-platform \
  --template-body file://infra/ecs-platform-stack.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=commercial \
    ParameterKey=VpcId,ParameterValue=$VPC_ID \
    ParameterKey=PublicSubnet1Id,ParameterValue=$PUB1 \
    ParameterKey=PublicSubnet2Id,ParameterValue=$PUB2 \
    ParameterKey=ALBSecurityGroupId,ParameterValue=$ALB_SG \
    ParameterKey=BedrockKbId,ParameterValue=$KB_ID \
    ParameterKey=BedrockRegion,ParameterValue=us-east-1 \
  --capabilities CAPABILITY_NAMED_IAM \
  --tags Key=Application,Value=nl-search-chatbot
```

### Step 6: Deploy ECS Services Stack

```bash
OPENSEARCH_ENDPOINT=$(aws cloudformation describe-stacks --stack-name nl-search-opensearch --query 'Stacks[0].Outputs[?OutputKey==`CollectionEndpoint`].OutputValue' --output text)

# Get the inference profile ARN for Claude Sonnet 4
INFERENCE_ARN=$(aws bedrock list-inference-profiles --query 'inferenceProfileSummaries[?contains(inferenceProfileName, `Sonnet 4`) && status==`ACTIVE`].inferenceProfileArn | [0]' --output text)

aws cloudformation create-stack \
  --stack-name nl-search-services \
  --template-body file://infra/ecs-services-stack.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=commercial \
    ParameterKey=PlatformStackName,ParameterValue=nl-search-platform \
    ParameterKey=NetworkStackName,ParameterValue=nl-search-network \
    ParameterKey=BackendImageUri,ParameterValue=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-commercial-backend:latest \
    ParameterKey=FrontendImageUri,ParameterValue=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-commercial-frontend:latest \
    ParameterKey=OpenSearchEndpoint,ParameterValue=$OPENSEARCH_ENDPOINT \
    ParameterKey=BedrockKbId,ParameterValue=$KB_ID \
    ParameterKey=BedrockRegion,ParameterValue=us-east-1 \
    ParameterKey=InferenceProfileArn,ParameterValue=$INFERENCE_ARN \
  --tags Key=Application,Value=nl-search-chatbot
```

### Step 7: Upload Documents and Sync

```bash
# Upload documents to the corpus bucket
aws s3 cp my-documents/ s3://nl-search-commercial-corpus-$ACCOUNT_ID/ --recursive

# Trigger initial sync
aws lambda invoke --function-name nl-search-commercial-kb-sync /dev/stdout
```

### Redeploying After Code Changes

Only the services stack needs updating:

```bash
# Rebuild and push images
docker build --platform linux/amd64 -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-commercial-backend:latest backend/
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-commercial-backend:latest

# Update services stack (CloudFormation detects new image digest)
aws cloudformation update-stack \
  --stack-name nl-search-services \
  --template-body file://infra/ecs-services-stack.yaml \
  --parameters \
    ParameterKey=BackendImageUri,ParameterValue=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-commercial-backend:$(date +%s) \
    ParameterKey=FrontendImageUri,UsePreviousValue=true \
    ParameterKey=Environment,UsePreviousValue=true \
    ParameterKey=PlatformStackName,UsePreviousValue=true \
    ParameterKey=NetworkStackName,UsePreviousValue=true \
    ParameterKey=OpenSearchEndpoint,UsePreviousValue=true \
    ParameterKey=BedrockKbId,UsePreviousValue=true \
    ParameterKey=BedrockRegion,UsePreviousValue=true \
    ParameterKey=InferenceProfileArn,UsePreviousValue=true
```


## Deployment — GovCloud US East (us-gov-east-1)

### Changes Required for GovCloud

The CloudFormation templates are parameterized with `Environment=govcloud` which automatically handles:

| What Changes | Commercial | GovCloud |
|---|---|---|
| ARN prefix | `arn:aws` | `arn:aws-us-gov` |
| Region | `us-east-1` | `us-gov-east-1` |
| LLM access | Direct or cross-region inference | Cross-region inference required |
| Embedding model | Titan Text Embeddings V2 (native) | Titan Text Embeddings V2 (native) |
| ECR endpoint | `{acct}.dkr.ecr.us-east-1.amazonaws.com` | `{acct}.dkr.ecr.us-gov-east-1.amazonaws.com` |
| VPC endpoint services | `com.amazonaws.us-east-1.*` | `com.amazonaws.us-gov-east-1.*` |

### GovCloud-Specific Prerequisites

1. **Bedrock model access**: Enable Claude 3.5 Sonnet (or Claude 3.7 Sonnet) via cross-region inference in us-gov-east-1. Titan Text Embeddings V2 is available natively.
2. **OpenSearch Serverless**: Available in us-gov-east-1. Verify supported AZs for AOSS VPC endpoints (may differ from commercial).
3. **ECR**: Available in us-gov-east-1. Container images must be built and pushed to the GovCloud ECR.
4. **Network**: The network stack hardcodes AZs `us-east-1b` and `us-east-1c`. For GovCloud, update these to valid us-gov-east-1 AZs.

### Required Template Changes

#### 1. Network Stack — Update Availability Zones

Edit `infra/network-stack.yaml` and change the hardcoded AZs to GovCloud AZs:

```yaml
# Before (commercial):
AvailabilityZone: us-east-1b
AvailabilityZone: us-east-1c

# After (GovCloud):
AvailabilityZone: us-gov-east-1a
AvailabilityZone: us-gov-east-1b
```

Alternatively, parameterize the AZs:

```yaml
Parameters:
  AZ1:
    Type: AWS::EC2::AvailabilityZone::Name
    Default: us-east-1b
  AZ2:
    Type: AWS::EC2::AvailabilityZone::Name
    Default: us-east-1c
```

Then use `!Ref AZ1` / `!Ref AZ2` in the subnet definitions.

**Important**: Verify which AZs support OpenSearch Serverless VPC endpoints in us-gov-east-1:
```bash
aws ec2 describe-vpc-endpoint-services \
  --service-names com.amazonaws.us-gov-east-1.aoss \
  --query 'ServiceDetails[0].AvailabilityZones' \
  --region us-gov-east-1
```

#### 2. ALB Security Group — Update Allowed IPs

Edit `infra/network-stack.yaml` to set the appropriate allowed CIDR for your GovCloud environment (corporate VPN range, etc.) instead of a personal IP.

#### 3. Inference Profile ARN

The inference profile ARN format changes in GovCloud:

```
# Commercial:
arn:aws:bedrock:us-east-1:{account}:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0

# GovCloud:
arn:aws-us-gov:bedrock:us-gov-east-1:{account}:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0
```

Verify available inference profiles in GovCloud:
```bash
aws bedrock list-inference-profiles --region us-gov-east-1 \
  --query 'inferenceProfileSummaries[?status==`ACTIVE`].{name:inferenceProfileName,arn:inferenceProfileArn}'
```

Use the appropriate ARN when deploying the services stack.

### GovCloud Deployment Commands

```bash
export AWS_DEFAULT_REGION=us-gov-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-gov-east-1

# Step 1: Network (after updating AZs in template)
aws cloudformation create-stack \
  --stack-name nl-search-network \
  --template-body file://infra/network-stack.yaml \
  --parameters ParameterKey=Environment,ParameterValue=govcloud

# Step 2: OpenSearch
VPCE_ID=$(aws cloudformation describe-stacks --stack-name nl-search-network \
  --query 'Stacks[0].Outputs[?OutputKey==`OpenSearchServerlessEndpointId`].OutputValue' --output text)

aws cloudformation create-stack \
  --stack-name nl-search-opensearch \
  --template-body file://infra/opensearch-stack.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=govcloud \
    ParameterKey=VpcEndpointId,ParameterValue=$VPCE_ID

# Step 2b: Create vector index (same as commercial, change region to us-gov-east-1)

# Step 3: Knowledge Base
COLLECTION_ARN=$(aws cloudformation describe-stacks --stack-name nl-search-opensearch \
  --query 'Stacks[0].Outputs[?OutputKey==`CollectionArn`].OutputValue' --output text)

aws cloudformation create-stack \
  --stack-name nl-search-knowledgebase \
  --template-body file://infra/knowledgebase-stack.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=govcloud \
    ParameterKey=CollectionArn,ParameterValue=$COLLECTION_ARN \
  --capabilities CAPABILITY_NAMED_IAM

# Step 4: Build and push images to GovCloud ECR
aws ecr create-repository --repository-name nl-search-govcloud-backend --region $REGION
aws ecr create-repository --repository-name nl-search-govcloud-frontend --region $REGION

aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

docker build --platform linux/amd64 -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-govcloud-backend:latest backend/
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-govcloud-backend:latest

docker build --platform linux/amd64 -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-govcloud-frontend:latest frontend/
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-govcloud-frontend:latest

# Step 5: Platform stack
VPC_ID=$(aws cloudformation describe-stacks --stack-name nl-search-network --query 'Stacks[0].Outputs[?OutputKey==`VpcId`].OutputValue' --output text)
PUB1=$(aws cloudformation describe-stacks --stack-name nl-search-network --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnet1Id`].OutputValue' --output text)
PUB2=$(aws cloudformation describe-stacks --stack-name nl-search-network --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnet2Id`].OutputValue' --output text)
ALB_SG=$(aws cloudformation describe-stacks --stack-name nl-search-network --query 'Stacks[0].Outputs[?OutputKey==`ALBSecurityGroupId`].OutputValue' --output text)
KB_ID=$(aws cloudformation describe-stacks --stack-name nl-search-knowledgebase --query 'Stacks[0].Outputs[?OutputKey==`KnowledgeBaseId`].OutputValue' --output text)

aws cloudformation create-stack \
  --stack-name nl-search-platform \
  --template-body file://infra/ecs-platform-stack.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=govcloud \
    ParameterKey=VpcId,ParameterValue=$VPC_ID \
    ParameterKey=PublicSubnet1Id,ParameterValue=$PUB1 \
    ParameterKey=PublicSubnet2Id,ParameterValue=$PUB2 \
    ParameterKey=ALBSecurityGroupId,ParameterValue=$ALB_SG \
    ParameterKey=BedrockKbId,ParameterValue=$KB_ID \
    ParameterKey=BedrockRegion,ParameterValue=us-gov-east-1 \
  --capabilities CAPABILITY_NAMED_IAM

# Step 6: Services stack
OPENSEARCH_ENDPOINT=$(aws cloudformation describe-stacks --stack-name nl-search-opensearch --query 'Stacks[0].Outputs[?OutputKey==`CollectionEndpoint`].OutputValue' --output text)

# Get GovCloud inference profile ARN
INFERENCE_ARN=$(aws bedrock list-inference-profiles \
  --query 'inferenceProfileSummaries[?contains(inferenceProfileName, `Sonnet`) && status==`ACTIVE`].inferenceProfileArn | [0]' --output text)

aws cloudformation create-stack \
  --stack-name nl-search-services \
  --template-body file://infra/ecs-services-stack.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=govcloud \
    ParameterKey=PlatformStackName,ParameterValue=nl-search-platform \
    ParameterKey=NetworkStackName,ParameterValue=nl-search-network \
    ParameterKey=BackendImageUri,ParameterValue=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-govcloud-backend:latest \
    ParameterKey=FrontendImageUri,ParameterValue=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/nl-search-govcloud-frontend:latest \
    ParameterKey=OpenSearchEndpoint,ParameterValue=$OPENSEARCH_ENDPOINT \
    ParameterKey=BedrockKbId,ParameterValue=$KB_ID \
    ParameterKey=BedrockRegion,ParameterValue=us-gov-east-1 \
    ParameterKey=InferenceProfileArn,ParameterValue=$INFERENCE_ARN
```

### GovCloud Checklist

- [ ] Verify Bedrock model access is enabled for Claude and Titan Embeddings V2
- [ ] Verify OpenSearch Serverless is available in your GovCloud AZs
- [ ] Update network-stack.yaml AZs to valid us-gov-east-1 AZs
- [ ] Update ALB security group CIDRs for your network
- [ ] Build and push Docker images to GovCloud ECR (images don't transfer between partitions)
- [ ] Update AOSS data access policy to include the KB role ARN
- [ ] Create the vector index on the AOSS collection before deploying the KB
- [ ] Verify the inference profile ARN is correct for GovCloud

## Teardown

Delete stacks in reverse order:

```bash
aws cloudformation delete-stack --stack-name nl-search-services
# Wait for DELETE_COMPLETE
aws cloudformation delete-stack --stack-name nl-search-platform
# Wait for DELETE_COMPLETE
aws cloudformation delete-stack --stack-name nl-search-knowledgebase
# Wait for DELETE_COMPLETE
aws cloudformation delete-stack --stack-name nl-search-opensearch
# Wait for DELETE_COMPLETE
aws cloudformation delete-stack --stack-name nl-search-network
```

Also clean up manually-created resources:
```bash
aws ecr delete-repository --repository-name nl-search-commercial-backend --force
aws ecr delete-repository --repository-name nl-search-commercial-frontend --force
```

## Running Tests

```bash
cd backend
pip install -r requirements.txt
python -m pytest tests/ -v
```

All 225 tests should pass, including 17 Hypothesis property-based tests validating the formal correctness properties defined in the design document.
# sample-govcloud-natural-language-search
# sample-govcloud-natural-language-search
# sample-govcloud-natural-language-search
