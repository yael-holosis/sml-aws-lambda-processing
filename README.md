# AWS Lambda Processing Project

This project sets up an AWS Lambda function using Docker containers, CDK, and Poetry for Python dependency management.

## Prerequisites

- AWS CLI installed
- Node.js and npm installed
- Python 3.8+ installed
- Poetry installed
- Docker installed and running

## Project Structure

```
.
├── README.md
├── lib/
│   └── aws_lambda_processing-stack.ts  # CDK Infrastructure
├── image/
│   ├── src/
│   │   └── main.py                    # Main Lambda Function
│   ├── Dockerfile                      # Docker configuration
│   └── requirements.txt                # Python dependencies
├── cdk.json
└── pyproject.toml
```

## Lambda Function Implementation

### Main Handler (`image/src/main.py`)

The Lambda function's entry point is the `main.py` file, with the main function named `main`. This is where your main processing logic lives. The main function receives events from various AWS services or HTTP requests through Function URL.

Example structure:
```python
import json
import boto3
import os

def get_secret():
    """Retrieve secret from AWS Secrets Manager"""
    secret_name = "system_db/credentials"
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager'
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        return json.loads(get_secret_value_response['SecretString'])
    except Exception as e:
        raise e

def main(event, context):
    """
    Main Lambda function
    :param event: Trigger event (HTTP request, S3 event, etc.)
    :param context: Lambda runtime information
    :return: API Gateway/Function URL response
    """
    try:
        # Get secrets if needed
        credentials = get_secret()
        
        # Process the event based on source
        if 'httpMethod' in event:  # Function URL/API Gateway request
            body = json.loads(event.get('body', '{}'))
            # Process HTTP request...
        elif 'Records' in event:  # S3, SQS, or other AWS service event
            # Process AWS service event...
            pass
            
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': 'Success',
                'data': result
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
```

### Function Configuration

The Lambda function is configured with the following specifications in `lib/aws_lambda_processing-stack.ts`:

```typescript
const dockerFunc = new lambda.DockerImageFunction(this, "DockerFunc", {
  code: lambda.DockerImageCode.fromImageAsset("./image"),
  memorySize: 4096,  // Memory in MB (can be from 128MB to 10,240MB)
  timeout: cdk.Duration.seconds(900),  // Timeout in seconds (max 900 seconds/15 minutes)
  architecture: lambda.Architecture.ARM_64,
});
```

Key configurations:
- `memorySize`: 4GB RAM allocated (also affects CPU power)
- `timeout`: 15 minutes maximum execution time
- `architecture`: ARM64 for better cost-performance ratio

## Initial Setup

### 1. Configure AWS Locally

If you haven't configured AWS CLI:

```bash
aws configure
```

You'll need to enter:
- AWS Access Key ID
- AWS Secret Access Key
- Default region
- Default output format (json recommended)

### 2. Poetry Initialization

```bash
# Initialize poetry project
poetry init

# Install project dependencies
poetry add aws-cdk-lib constructs
```

### 3. Project Structure Setup

Create the required directories:
```bash
mkdir -p image/src
```

### 4. Generate Requirements File

```bash
# Generate requirements.txt in the image folder
poetry export -f requirements.txt --output image/requirements.txt --without-hashes
```

## Secrets Manager Configuration

The stack uses AWS Secrets Manager to securely store and access credentials:

```typescript
const secret = secretsmanager.Secret.fromSecretNameV2(
  this, 
  "SystemDBCredentials", 
  "system_db/credentials"
);
```

Best practices:
- Use descriptive paths for secrets (e.g., "environment/service/credentials")
- Never hardcode secret values in your code
- Use environment variables to pass the secret name to your Lambda function
- Rotate secrets regularly using AWS Secrets Manager automatic rotation

## IAM Policies

Note: The following IAM policies are only needed if your IAM user/role doesn't already have these permissions through IAM Identity Center or existing role configurations. Check with your AWS administrator if you're unsure.

Required AWS managed policies:
- AmazonS3ReadOnlyAccess
- AWSLambdaBasicExecutionRole
- AWSLambdaVPCAccessExecutionRole
- SecretsManagerReadWrite

These provide permissions for:
- Reading from S3 buckets
- Basic Lambda execution (CloudWatch logs)
- VPC access if needed
- Reading/Writing to Secrets Manager

## Deployment

### 1. Bootstrap CDK Environment

```bash
cdk bootstrap
```

This command is necessary because:
- It creates an S3 bucket for storing CDK assets
- Sets up IAM roles for deployment
- Creates other necessary resources for CDK to work

### 2. Deploy the Stack

```bash
cdk deploy
```

After deployment, the Function URL will be displayed in the outputs.

## Development

To make changes to the Lambda function:
1. Update code in `image/src/`
2. Build and test locally if needed
3. Run `cdk deploy` to deploy changes

## Cleanup

To remove all deployed resources:
```bash
cdk destroy
```
