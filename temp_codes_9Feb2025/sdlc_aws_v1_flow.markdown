# Agentic SDLC Application Implementation

## Overview
This document outlines the end-to-end implementation of a generic **Agentic SDLC Application** that processes any requirement document through a 10-step Software Development Life Cycle (SDLC) using an AI-driven, AWS-centric workflow. The application leverages AWS services, external tools, and a Human-in-the-Loop (HITL) interface to validate, develop, test, deploy, and monitor software based on user-provided requirements. The default tech stack is Python with FastAPI (backend) and ReactJS (HITL UI), but users can override it via the UI.

### SDLC Steps
1. Input and Validation of Requirement Document
2. Creation of Technical Documentation
3. Task Breakdown and Jira Creation
4. Architecture and Environment Setup
5. Code Development
6. Test Case Generation and Testing
7. Code Check-in and Version Control
8. Build and Deployment Package Creation
9. Deployment to Amazon EKS
10. Post-Deployment Monitoring and Maintenance

### AWS Services
- **FastAPI**: Backend API for HITL UI and workflow interactions.
- **ReactJS**: HITL interface for requirement uploads, feedback, and approvals.
- **AWS Step Functions**: Orchestrates the 10-step workflow.
- **Amazon SQS**: Manages task queues for agent execution.
- **Amazon ElastiCache (Redis)**: In-memory session state.
- **Amazon DynamoDB**: Persistent storage for sessions, steps, transactions, and feedback.
- **AWS Bedrock (Claude 3.5 Sonnet)**: LLM for validation, code generation, decision-making and documentation.
- **AWS Lambda**: Executes agent logic for each step.
- **Amazon EventBridge**: Triggers workflows on S3 uploads.
- **Amazon S3**: Stores requirement documents and artifacts.
- **Amazon CloudWatch**: Monitors execution and application metrics.
- **Amazon EKS**: Deploys the application.
- **AWS Secrets Manager**: Stores sensitive credentials.
- **Amazon API Gateway** (Optional): Exposes FastAPI publicly.

### External Tools
- **ESTJIRA**: Task tracking.
- **GitLab**: Version control and CI/CD.
- **Confluence**: Documentation storage.
- **SonarQube**: Code quality checks.
- **Docker/NEXUS**: Containerization and artifact storage.

## Coding Standards
### Python (Backend and Lambda)
- **Style**: [PEP 8](https://www.python.org/dev/peps/pep-0008/) (88-character lines, snake_case variables, PascalCase classes).
- **Documentation**: Google-style docstrings for modules, classes, functions; inline comments for complex logic.
- **Linting**: Black (`black .`), Flake8 (`flake8 . --max-line-length=88`).
- **Error Handling**: Try-except for AWS calls, log errors to CloudWatch and DynamoDB.
- **Testing**: Pytest with >80% coverage, mock AWS services with moto.
- **Commits**: Semantic messages (e.g., `feat: add JWT`, `fix: SQS retry`).

### JavaScript (ReactJS)
- **Style**: [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript) (camelCase, functional components).
- **Documentation**: JSDoc for functions/components, comments for state logic.
- **Linting**: ESLint with Airbnb preset (`eslint .`), Prettier (`prettier --write .`).
- **Error Handling**: Try-catch for API calls, user-friendly UI messages.

### General
- **Modularity**: Reusable modules (e.g., `utils/db.py`).
- **Security**: No hardcoded credentials, use Secrets Manager, validate inputs.
- **Performance**: Optimize Lambda cold starts, use connection pooling for Redis/DynamoDB.

## Architecture
The workflow is orchestrated by Step Functions, invoking Lambda functions (agents) for each step. FastAPI, exposed via API Gateway (optional), serves the ReactJS HITL UI. SQS queues tasks, ElastiCache stores session state, and DynamoDB persists data. Bedrock Claude processes text and code. EventBridge triggers workflows, S3 stores artifacts, CloudWatch logs metrics, and Secrets Manager secures credentials.

**Diagram 1** (Mermaid):
```mermaid
graph TD
    classDef agent fill:#f9d5e5,stroke:#333,stroke-width:2px;
    classDef aws fill:#ffebcc,stroke:#333,stroke-width:2px;
    classDef external fill:#c4e7c4,stroke:#333,stroke-width:2px;
    classDef hitl fill:#d4e7f7,stroke:#333,stroke-width:2px;

    A[ReactJS Frontend<br>Tech: React, TailwindCSS, Axios<br>HITL: Upload, Feedback, Approvals]:::hitl -->|HTTP| N[API Gateway<br>Public Endpoint]:::aws --> B[FastAPI Backend<br>Tech: Python, FastAPI, Uvicorn]:::hitl
    B -->|Start Execution| C[AWS Step Functions<br>Orchestrates Workflow]:::aws
    J[Amazon S3<br>Stores Requirements]:::aws
    A -->|Upload| J
    B -->|Fetch| J
    I[Amazon EventBridge<br>Triggers Workflow]:::aws -->|Trigger| C
    O[AWS Secrets Manager<br>Stores Credentials]:::aws --> B
    O --> D1

    C -->|Invoke| D1[Requirement Validator<br>Agent: Lambda<br>Tech: Python, Bedrock Claude 3.5]:::agent
    D1 -->|Queue Task| E[Amazon SQS<br>Task Queue]:::aws
    D1 -->|Store State| F[ElastiCache Redis<br>Session State]:::aws
    D1 -->|Persist Data| G[Amazon DynamoDB<br>Sessions, Steps, Transactions, Feedback]:::aws
    D1 -->|Generate Report| L1[Confluence<br>Validation Report]:::external
    D1 -->|HITL Feedback| A

    C -->|Invoke| D2[Tech Doc Generator<br>Agent: Lambda<br>Tech: Python, Bedrock Claude 3.5]:::agent --> E --> F --> G --> L1 --> A
    C -->|Invoke| D3[Task Orchestrator<br>Agent: Lambda<br>Tech: Python, Bedrock Claude 3.5]:::agent --> E --> F --> G --> L2[ESTJIRA<br>Task Tracking]:::external --> A
    C -->|Invoke| D4[Environment Setup<br>Agent: Lambda<br>Tech: Python, GitLab API]:::agent --> E --> F --> G --> L3[GitLab<br>Version Control, CI/CD]:::external --> A
    C -->|Invoke| D5[Code Generator<br>Agent: Lambda<br>Tech: Python, Bedrock Claude 3.5]:::agent --> E --> F --> G --> L3 --> L4[SonarQube<br>Code Analysis]:::external --> A
    C -->|Invoke| D6[Test Automation<br>Agent: Lambda<br>Tech: Python, Bedrock Claude 3.5, Pytest]:::agent --> E --> F --> G --> L3 --> A
    C -->|Invoke| D7[Version Control<br>Agent: Lambda<br>Tech: Python, GitLab API]:::agent --> E --> F --> G --> L3 --> A
    C -->|Invoke| D8[Build<br>Agent: Lambda<br>Tech: Python, Docker]:::agent --> E --> F --> G --> L3 --> L5[NEXUS<br>Artifact Repository]:::external --> A
    C -->|Invoke| D9[Deployment<br>Agent: Lambda<br>Tech: Python, Kubernetes]:::agent --> E --> F --> G --> M[Amazon EKS<br>Hosts Application]:::aws --> L3 --> A
    C -->|Invoke| D10[Monitoring<br>Agent: Lambda<br>Tech: Python, CloudWatch]:::agent --> E --> F --> G --> K[Amazon CloudWatch<br>Monitoring & Logging]:::aws --> L2 --> A

    subgraph Legend
        LAgent[Agent (Lambda)]
        LAWS[AWS Service]
        LExternal[External Tool]
        LHITL[HITL Interface]
    end
    class LAgent agent; class LAWS aws; class LExternal external; class LHITL hitl;
```

**Diagram 2** (Mermaid):
```mermaid
flowchart TD
    %% Styling
    classDef requirements fill:#f9d5e5,stroke:#333,stroke-width:2px,color:#000;
    classDef design fill:#ffebcc,stroke:#333,stroke-width:2px,color:#000;
    classDef development fill:#c4e7c4,stroke:#333,stroke-width:2px,color:#000;
    classDef testing fill:#d4e7f7,stroke:#333,stroke-width:2px,color:#000;
    classDef deployment fill:#e0daf7,stroke:#333,stroke-width:2px,color:#000;
    classDef monitoring fill:#ffeedd,stroke:#333,stroke-width:2px,color:#000;
    classDef infrastructure fill:#dddddd,stroke:#333,stroke-width:2px,color:#000;
    
    %% SDLC Phases with Agentic AI
    subgraph Requirements [Requirements Phase]
        R1[User<br>Requirements Input]:::requirements
        R2[Requirement Validator Agent<br>Claude 3.5-powered]:::requirements
        R3[Requirements Validation<br>Report]:::requirements
        R1 --> R2 --> R3
    end
    
    subgraph Design [Design Phase]
        D1[Tech Doc Generator Agent<br>Claude 3.5-powered]:::design
        D2[Task Orchestrator Agent<br>Claude 3.5-powered]:::design
        D3[Technical Specifications<br>& Task Planning]:::design
        D1 --> D3
        D2 --> D3
    end
    
    subgraph Development [Development Phase]
        Dev1[Environment Setup Agent<br>GitLab API Integration]:::development
        Dev2[Code Generator Agent<br>Claude 3.5-powered]:::development
        Dev3[Version Control Agent<br>GitLab API Integration]:::development
        Dev4[Source Code Repository<br>& Version Management]:::development
        Dev1 --> Dev2 --> Dev3 --> Dev4
    end
    
    subgraph Testing [Testing Phase]
        T1[Test Automation Agent<br>Claude 3.5 & Pytest]:::testing
        T2[Automated Tests<br>& Quality Assurance]:::testing
        T3[Code Quality Analysis<br>SonarQube Integration]:::testing
        T1 --> T2
        Dev4 --> T3
    end
    
    subgraph Deployment [Deployment Phase]
        Dep1[Build Agent<br>Docker Integration]:::deployment
        Dep2[Deployment Agent<br>Kubernetes Integration]:::deployment
        Dep3[Containerized Application<br>Deployment]:::deployment
        Dep1 --> Dep2 --> Dep3
    end
    
    subgraph Monitoring [Monitoring & Maintenance]
        M1[Monitoring Agent<br>CloudWatch Integration]:::monitoring
        M2[Performance Metrics<br>& Issue Tracking]:::monitoring
        M1 --> M2
    end
    
    subgraph Infrastructure [Shared Infrastructure]
        I1[AWS Step Functions<br>Workflow Orchestration]:::infrastructure
        I2[Amazon SQS<br>Task Queue]:::infrastructure
        I3[ElastiCache Redis<br>In-Memory State]:::infrastructure
        I4[Amazon DynamoDB<br>Persistent Storage]:::infrastructure
        I5[Human-in-the-Loop<br>Feedback & Approvals]:::infrastructure
    end
    
    %% SDLC Phase Flow
    Requirements --> Design
    Design --> Development
    Development --> Testing
    Testing --> Deployment
    Deployment --> Monitoring
    Monitoring -.-> Requirements
    
    %% Infrastructure connections
    Infrastructure --- Requirements
    Infrastructure --- Design
    Infrastructure --- Development
    Infrastructure --- Testing
    Infrastructure --- Deployment
    Infrastructure --- Monitoring
    
    %% External integrations
    Design --- ExtInt1[Confluence<br>Documentation]
    Design --- ExtInt2[ESTJIRA<br>Task Tracking]
    Development --- ExtInt3[GitLab<br>Version Control/CI/CD]
    Testing --- ExtInt4[SonarQube<br>Code Analysis]
    Deployment --- ExtInt5[NEXUS<br>Artifact Repository]
    Deployment --- ExtInt6[Amazon EKS<br>Application Hosting]
    Monitoring --- ExtInt7[CloudWatch<br>Monitoring & Logging]
```

**Diagram 3** (Mermaid):
```mermaid
flowchart LR
    %% Styling
    classDef presentation fill:#d4e7f7,stroke:#333,stroke-width:1px,color:#000
    classDef agents fill:#f9d5e5,stroke:#333,stroke-width:1px,color:#000
    classDef infra fill:#c4e7c4,stroke:#333,stroke-width:1px,color:#000
    classDef external fill:#e0daf7,stroke:#333,stroke-width:1px,color:#000
    classDef header fill:#eeeeee,stroke:#333,stroke-width:2px,color:#000

    %% Columns for SDLC phases
    subgraph col1 [Requirements]
        direction TB
        R1[Requirements<br>Phase]:::header
        R2[ReactJS UI]:::presentation
        R3[Requirement<br>Validator Agent]:::agents
        R4[AWS Lambda +<br>Claude 3.5]:::infra
        R5[Confluence<br>Integration]:::external
        
        R2 --> R3 --> R4 --> R5
    end
    
    subgraph col2 [Design]
        direction TB
        D1[Design<br>Phase]:::header
        D2[Document<br>Review UI]:::presentation
        D3[Tech Doc Generator<br>Task Orchestrator]:::agents
        D4[AWS Lambda +<br>Claude 3.5]:::infra
        D5[Confluence<br>ESTJIRA]:::external
        
        D2 --> D3 --> D4 --> D5
    end
    
    subgraph col3 [Development]
        direction TB
        Dev1[Development<br>Phase]:::header
        Dev2[Code Review<br>UI]:::presentation
        Dev3[Environment Setup<br>Code Generator<br>Version Control]:::agents
        Dev4[AWS Lambda +<br>Claude 3.5]:::infra
        Dev5[GitLab<br>Integration]:::external
        
        Dev2 --> Dev3 --> Dev4 --> Dev5
    end
    
    subgraph col4 [Testing]
        direction TB
        T1[Testing<br>Phase]:::header
        T2[Test Results<br>UI]:::presentation
        T3[Test Automation<br>Agent]:::agents
        T4[AWS Lambda +<br>Pytest]:::infra
        T5[SonarQube<br>Integration]:::external
        
        T2 --> T3 --> T4 --> T5
    end
    
    subgraph col5 [Deployment]
        direction TB
        Dep1[Deployment<br>Phase]:::header
        Dep2[Deployment<br>Dashboard]:::presentation
        Dep3[Build Agent<br>Deployment Agent]:::agents
        Dep4[AWS Lambda<br>EKS]:::infra
        Dep5[NEXUS<br>GitLab]:::external
        
        Dep2 --> Dep3 --> Dep4 --> Dep5
    end
    
    subgraph col6 [Monitoring]
        direction TB
        M1[Monitoring<br>Phase]:::header
        M2[Monitoring<br>Dashboard]:::presentation
        M3[Monitoring<br>Agent]:::agents
        M4[AWS Lambda<br>CloudWatch]:::infra
        M5[ESTJIRA<br>Notifications]:::external
        
        M2 --> M3 --> M4 --> M5
    end
    
    %% Shared infrastructure layer at bottom
    subgraph shared [Shared Infrastructure]
        direction LR
        S1[API Gateway]
        S2[Step Functions]
        S3[S3 Storage]
        S4[DynamoDB]
        S5[ElastiCache]
        S6[SQS]
        S7[Secrets Manager]
        S8[EventBridge]
    end
    
    %% Connect phases
    col1 --> col2
    col2 --> col3
    col3 --> col4
    col4 --> col5
    col5 --> col6
    col6 -.-> col1
    
    %% Connect all columns to shared infrastructure
    shared --- col1
    shared --- col2
    shared --- col3
    shared --- col4
    shared --- col5
    shared --- col6
```

## Prerequisites
- **AWS Account**: IAM roles for Lambda, Step Functions, SQS, ElastiCache, DynamoDB, Bedrock, S3, EKS, CloudWatch, Secrets Manager, API Gateway.
- **Dependencies**:
  - Backend: `pip install fastapi==0.115.2 uvicorn==0.32.0 boto3==1.35.24 python-jose[cryptography]==3.3.0`
  - Frontend: Node.js v18, `npm install react@18.2.0 react-dom@18.2.0 axios@1.4.0 tailwindcss@3.4.1`
  - Lambda: `pip install boto3==1.35.24 gitlab==4.17.0 atlassian-python-api==3.41.5 pytest==8.3.3 moto==5.0.12`
- **Configuration**:
  - Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `BEDROCK_REGION=us-east-1`.
  - Secrets in Secrets Manager: `GITLAB_TOKEN`, `CONFLUENCE_CREDENTIALS`, `JIRA_CREDENTIALS`, `JWT_SECRET_KEY`.
  - Deploy ElastiCache (Redis), DynamoDB, SQS, Step Functions, EKS, API Gateway (optional).

## Security
- **FastAPI Authentication**: JWT with HS256, validated via `python-jose`.
  - JWT-based authentication secures API endpoints.
  - Example middleware in `backend/main.py` validates tokens.
- **IAM Roles**:
  - Lambda: Permissions for S3, SQS, DynamoDB, Bedrock, Step Functions, CloudWatch, Secrets Manager.
  - EKS: Permissions for EKS cluster management and DynamoDB access.
  - Example IAM policy:
    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": [
            "s3:GetObject",
            "s3:PutObject",
            "sqs:SendMessage",
            "sqs:ReceiveMessage",
            "dynamodb:PutItem",
            "dynamodb:UpdateItem",
            "dynamodb:GetItem",
            "bedrock:InvokeModel",
            "states:StartExecution",
            "logs:CreateLogStream",
            "logs:PutLogEvents",
            "secretsmanager:GetSecretValue"
          ],
          "Resource": "*"
        }
      ]
    }
    ```
- **DynamoDB Security**:
  - Fine-grained access control restricts Lambda access to specific tables.
  - Example policy:
    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": [
            "dynamodb:PutItem",
            "dynamodb:UpdateItem",
            "dynamodb:GetItem"
          ],
          "Resource": [
            "arn:aws:dynamodb:region:account-id:table/Sessions",
            "arn:aws:dynamodb:region:account-id:table/WorkflowSteps",
            "arn:aws:dynamodb:region:account-id:table/Transactions",
            "arn:aws:dynamodb:region:account-id:table/HITLFeedback"
          ]
        }
      ]
    }
    ```
- **Secrets Manager**:
  - Stores `GITLAB_TOKEN`, `CONFLUENCE_CREDENTIALS`, `JIRA_CREDENTIALS`.
  - Accessed via `boto3.client("secretsmanager")` in `utils/tools.py`.

## Error Handling
- **Lambda Retry Logic**:
  - Configured with `boto3` client retries for transient failures (e.g., Bedrock API timeouts).
  - Example:
    ```python
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1", config=Config(retries={"max_attempts": 3}))
    ```
- **Step Functions**:
  - Retry and catch states handle errors.
  - Updated `sdlc_workflow.json` includes:
    ```json
    "RequirementValidation": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account-id:function:requirement_validator",
      "Retry": [
        {
          "ErrorEquals": ["States.TaskFailed"],
          "IntervalSeconds": 2,
          "MaxAttempts": 3,
          "BackoffRate": 2.0
        }
      ],
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "Next": "ErrorHandler"
        }
      ],
      "Next": "WaitForValidationApproval"
    },
    "ErrorHandler": {
      "Type": "Fail",
      "Error": "TaskFailed",
      "Cause": "Task execution failed after retries"
    }
    ```
- **SQS**:
  - Visibility timeout: 30 seconds.
  - Dead-letter queue for failed messages.

## Scalability
- **Lambda**:
  - Concurrency limit: 1000 (configurable).
  - Reserved concurrency per function: 10 to prevent throttling.
- **SQS**:
  - Batch processing: Up to 10 messages per poll.
  - Throughput: Scaled automatically with AWS managed queues.
- **EKS**:
  - Horizontal Pod Autoscaling (HPA) for the deployed application.
  - Example HPA configuration:
    ```yaml
    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: sdlc-app-hpa
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: sdlc-app
      minReplicas: 2
      maxReplicas: 10
      metrics:
      - type: Resource
        resource:
          name: cpu
          target:
            type: Utilization
            averageUtilization: 70
    ```
## Monitoring
- **CloudWatch Alarms**: EKS CPU >80%, Lambda errors/throttles.
- **Dashboard**: Metrics for Lambda duration, SQS queue depth, EKS CPU/memory, DynamoDB capacity.

## Directory Structure
```
sdlc_aws_workflow/
├── backend/
│   ├── main.py
│   ├── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
├── lambda/
│   ├── requirement_validator/
│   │   ├── lambda_function.py
│   │   ├── test_lambda_function.py
│   ├── tech_doc_generator/lambda_function.py
│   ├── task_orchestrator/lambda_function.py
│   ├── env_setup/lambda_function.py
│   ├── code_generator/lambda_function.py
│   ├── test_automation/lambda_function.py
│   ├── version_control/lambda_function.py
│   ├── build/lambda_function.py
│   ├── deployment/lambda_function.py
│   ├── monitoring/lambda_function.py
├── utils/
│   ├── db.py
│   ├── tools.py
├── step_functions/
│   ├── sdlc_workflow.json
├── deploy.sh
```

## Database Schema (DynamoDB)
### Tables
1. **Sessions**:
   - **Partition Key**: `session_id` (String)
   - **Attributes**: `created_at`, `updated_at`, `current_step`, `status`, `user_id`, `tech_stack` (Map: `{"language": "Python", "backend": "FastAPI", "ui": "None"}`)
2. **WorkflowSteps**:
   - **Partition Key**: `session_id` (String)
   - **Sort Key**: `step_id` (String)
   - **Attributes**: `step_name`, `status`, `artifacts` (Map), `started_at`, `completed_at`
3. **Transactions**:
   - **Partition Key**: `session_id` (String)
   - **Sort Key**: `transaction_id` (String)
   - **Attributes**: `step_id`, `action`, `actor`, `details` (Map), `timestamp`
4. **HITLFeedback**:
   - **Partition Key**: `session_id` (String)
   - **Sort Key**: `feedback_id` (String)
   - **Attributes**: `step_id`, `user_id`, `feedback`, `submitted_at`, `applied` (Boolean)

**Setup Script**:
```python
"""DynamoDB table setup script."""
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

dynamodb = boto3.resource("dynamodb", config=Config(retries={"max_attempts": 3}))

def create_tables():
    """Create DynamoDB tables for SDLC application.

    Creates four tables: Sessions, WorkflowSteps, Transactions, HITLFeedback.
    Uses PAY_PER_REQUEST billing mode for scalability.
    """
    tables = [
        {
            "TableName": "Sessions",
            "KeySchema": [{"AttributeName": "session_id", "KeyType": "HASH"}],
            "AttributeDefinitions": [{"AttributeName": "session_id", "AttributeType": "S"}]
        },
        {
            "TableName": "WorkflowSteps",
            "KeySchema": [
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "step_id", "KeyType": "RANGE"}
            ],
            "AttributeDefinitions": [
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "step_id", "AttributeType": "S"}
            ]
        },
        {
            "TableName": "Transactions",
            "KeySchema": [
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "transaction_id", "KeyType": "RANGE"}
            ],
            "AttributeDefinitions": [
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "transaction_id", "AttributeType": "S"}
            ]
        },
        {
            "TableName": "HITLFeedback",
            "KeySchema": [
                {"AttributeName": "session_id", "KeyType": "HASH"},
                {"AttributeName": "feedback_id", "KeyType": "RANGE"}
            ],
            "AttributeDefinitions": [
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "feedback_id", "AttributeType": "S"}
            ]
        }
    ]
    for table in tables:
        try:
            dynamodb.create_table(
                TableName=table["TableName"],
                KeySchema=table["KeySchema"],
                AttributeDefinitions=table["AttributeDefinitions"],
                BillingMode="PAY_PER_REQUEST"
            )
            print(f"Created table {table['TableName']}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceInUseException":
                print(f"Table {table['TableName']} already exists")
            else:
                raise

if __name__ == "__main__":
    create_tables()
```

## Implementation Details

### Step 1: Input and Validation of Requirement Document
- **Objective**: Validate the uploaded requirement document for compatibility with the selected tech stack.
- **Components**:
  - **Lambda (Requirement Validator)**: Uses Bedrock Claude to validate the document.
  - **FastAPI**: Exposes `/upload-requirements`, `/start-session`, `/submit-feedback`, `/approve-step`.
  - **ReactJS**: UI for uploading documents, selecting tech stack, and providing feedback.
  - **SQS**: Queues validation tasks.
  - **DynamoDB/Redis**: Stores session and step data.
- **HITL**: Users review validation report in ReactJS, submit feedback, or approve via ReactJS.
- **Code**:
  - **utils/db.py**:
    ```python
    """Database utilities for SDLC application."""
    import boto3
    import redis
    import uuid
    from datetime import datetime
    import json
    from botocore.config import Config

    dynamodb = boto3.resource("dynamodb", config=Config(retries={"max_attempts": 3}))
    redis_client = redis.Redis(host="your-elasticache-endpoint", port=6379, decode_responses=True)

    def create_session(user_id: str, tech_stack: dict = None) -> str:
        """Create a new session and store in DynamoDB and Redis.

        Args:
            user_id: Unique identifier for the user.
            tech_stack: Dictionary specifying language, backend, and UI (optional).

        Returns:
            session_id: Unique identifier for the session.
        """
        session_id = str(uuid.uuid4())
        tech_stack = tech_stack or {"language": "Python", "backend": "FastAPI", "ui": "None"}
        session = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "current_step": "Requirement Validation",
            "status": "Active",
            "user_id": user_id,
            "tech_stack": tech_stack
        }
        dynamodb.Table("Sessions").put_item(Item=session)
        redis_client.set(f"session:{session_id}", json.dumps(session))
        return session_id

    def log_step(session_id: str, step_name: str, status: str, artifacts: dict) -> str:
        """Log a workflow step to DynamoDB.

        Args:
            session_id: Session identifier.
            step_name: Name of the SDLC step.
            status: Step status (e.g., In Review, Approved).
            artifacts: Dictionary of step artifacts (e.g., report URLs).

        Returns:
            step_id: Unique identifier for the step.
        """
        step_id = str(uuid.uuid4())
        step = {
            "session_id": session_id,
            "step_id": step_id,
            "step_name": step_name,
            "status": status,
            "artifacts": artifacts,
            "started_at": datetime.utcnow().isoformat()
        }
        dynamodb.Table("WorkflowSteps").put_item(Item=step)
        return step_id

    def log_transaction(session_id: str, step_id: str, action: str, actor: str, details: dict):
        """Log a transaction to DynamoDB.

        Args:
            session_id: Session identifier.
            step_id: Step identifier.
            action: Action performed (e.g., Generate Report).
            actor: Entity performing the action (e.g., Requirement Validator).
            details: Additional metadata.
        """
        transaction_id = str(uuid.uuid4())
        dynamodb.Table("Transactions").put_item(Item={
            "session_id": session_id,
            "transaction_id": transaction_id,
            "step_id": step_id,
            "action": action,
            "actor": actor,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })

    def log_feedback(session_id: str, step_id: str, user_id: str, feedback: str):
        """Log HITL feedback to DynamoDB and update step status.

        Args:
            session_id: Session identifier.
            step_id: Step identifier.
            user_id: User providing feedback.
            feedback: Feedback content.
        """
        feedback_id = str(uuid.uuid4())
        dynamodb.Table("HITLFeedback").put_item(Item={
            "session_id": session_id,
            "feedback_id": feedback_id,
            "step_id": step_id,
            "user_id": user_id,
            "feedback": feedback,
            "submitted_at": datetime.utcnow().isoformat(),
            "applied": False
        })
        dynamodb.Table("WorkflowSteps").update_item(
            Key={"session_id": session_id, "step_id": step_id},
            UpdateExpression="SET #status = :status",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":status": "Pending"}
        )
    ```
  - **utils/tools.py**:
    ```python
    """Utility functions for AWS and external tool interactions."""
    import boto3
    import json
    from atlassian import Confluence, Jira
    import gitlab
    from botocore.config import Config

    s3_client = boto3.client("s3", config=Config(retries={"max_attempts": 3}))
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1", config=Config(retries={"max_attempts": 3}))
    secrets_client = boto3.client("secretsmanager", config=Config(retries={"max_attempts": 3}))

    def get_secret(secret_name: str) -> dict:
        """Retrieve a secret from AWS Secrets Manager.

        Args:
            secret_name: Name of the secret.

        Returns:
            Secret value as a dictionary.
        """
        response = secrets_client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    confluence_creds = get_secret("CONFLUENCE_CREDENTIALS")
    jira_creds = get_secret("JIRA_CREDENTIALS")
    gitlab_token = get_secret("GITLAB_TOKEN")["token"]
    confluence = Confluence(url="your-confluence-url", username=confluence_creds["username"], password=confluence_creds["password"])
    jira = Jira(url="your-estjira-url", username=jira_creds["username"], password=jira_creds["password"])
    gl = gitlab.Gitlab("your-gitlab-url", private_token=gitlab_token)

    def read_s3_file(bucket: str, key: str) -> str:
        """Read a file from S3.

        Args:
            bucket: S3 bucket name.
            key: S3 object key.

        Returns:
            File content as a string.
        """
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8")

    def upload_s3_file(content: bytes, bucket: str, key: str):
        """Upload a file to S3.

        Args:
            content: File content.
            bucket: S3 bucket name.
            key: S3 object key.
        """
        s3_client.put_object(Body=content, Bucket=bucket, Key=key)

    def invoke_claude(prompt: str) -> str:
        """Invoke Bedrock Claude 3.5 Sonnet model.

        Args:
            prompt: Input prompt for the model.

        Returns:
            Model response as a string.
        """
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=json.dumps({
                "prompt": prompt,
                "max_tokens": 2000
            })
        )
        return json.loads(response["body"].read())["completion"]
    ```
  - **lambda/requirement_validator/lambda_function.py**:
    ```python
    """Lambda function for requirement validation."""
    import json
    import boto3
    from botocore.exceptions import ClientError
    from utils.db import log_step, log_transaction
    from utils.tools import read_s3_file, invoke_claude, confluence

    sqs = boto3.client("sqs", config=Config(retries={"max_attempts": 3}))
    QUEUE_URL = "your-sqs-queue-url"

    def lambda_handler(event: dict, context: object) -> dict:
        """Validate requirement document using Bedrock Claude.

        Args:
            event: Lambda event containing session_id, document_path, tech_stack.
            context: Lambda context object.

        Returns:
            Dictionary with status code and response body.

        Raises:
            ClientError: If AWS service calls fail after retries.
        """
        try:
            session_id = event["session_id"]
            document_path = event["document_path"]
            tech_stack = event["tech_stack"]
            s3_key = document_path.split("s3://your-bucket/")[1]
            document = read_s3_file("your-bucket", s3_key)
            prompt = (
                f"Validate this requirement document for compatibility with {tech_stack['language']} "
                f"(backend: {tech_stack['backend']}, UI: {tech_stack['ui']}):\n{document}\n"
                f"Flag any incompatibilities and suggest alternatives."
            )
            report = invoke_claude(prompt)
            step_id = log_step(session_id, "Requirement Validation", "In Review", {"report": "confluence://validation-report"})
            log_transaction(session_id, step_id, "Generate Report", "Requirement Validator", {"report": report})
            confluence.create_page(space="SDLC", title=f"Validation Report {session_id}", body=report)
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps({"session_id": session_id, "step_id": step_id, "report": report})
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"session_id": session_id, "step_id": step_id, "report": report})
            }
        except ClientError as e:
            log_transaction(session_id, step_id, "Error", "Requirement Validator", {"error": str(e)})
            raise
    ```
  - **lambda/requirement_validator/test_lambda_function.py**:
    ```python
    """Unit tests for requirement_validator Lambda."""
    import pytest
    import json
    from moto import mock_s3, mock_sqs, mock_dynamodb
    from lambda.requirement_validator.lambda_function import lambda_handler

    @mock_s3
    @mock_sqs
    @mock_dynamodb
    def test_lambda_handler(mocker):
        """Test requirement_validator Lambda function.

        Mocks S3, SQS, DynamoDB, and Claude to simulate validation.
        """
        mocker.patch("utils.tools.read_s3_file", return_value="Sample requirements")
        mocker.patch("utils.tools.invoke_claude", return_value="Validation report")
        mocker.patch("utils.db.log_step", return_value="step123")
        mocker.patch("utils.db.log_transaction")
        mocker.patch("utils.tools.confluence.create_page")
        sqs_client = boto3.client("sqs")
        sqs_client.create_queue(QueueName="test-queue")
        event = {
            "session_id": "session123",
            "document_path": "s3://your-bucket/requirements-session123.md",
            "tech_stack": {"language": "Python", "backend": "FastAPI", "ui": "None"}
        }
        response = lambda_handler(event, None)
        assert response["statusCode"] == 200
        assert json.loads(response["body"])["session_id"] == "session123"
    ```

### Step 2: Creation of Technical Documentation
- **Objective**: Generate technical specifications and architecture diagrams.
- **Components**:
  - **Lambda (Tech Doc Generator)**: Uses Claude to create specs and Mermaid diagrams.
  - **Confluence**: Stores documentation.
- **HITL**: Users review specs in ReactJS.
- **Code**:
  - **lambda/tech_doc_generator/lambda_function.py**:
    ```python
    """Lambda function for technical documentation generation."""
    import json
    import boto3
    from botocore.exceptions import ClientError
    from utils.db import log_step, log_transaction
    from utils.tools import read_s3_file, invoke_claude, confluence

    sqs = boto3.client("sqs", config=Config(retries={"max_attempts": 3}))
    QUEUE_URL = "your-sqs-queue-url"

    def lambda_handler(event: dict, context: object) -> dict:
        """Generate technical specs and Mermaid diagram using Bedrock Claude.

        Args:
            event: Lambda event containing session_id, document_path, tech_stack.
            context: Lambda context object.

        Returns:
            Dictionary with status code and response body.

        Raises:
            ClientError: If AWS service calls fail after retries.
        """
        try:
            session_id = event["session_id"]
            s3_key = event["document_path"].split("s3://your-bucket/")[1]
            requirements = read_s3_file("your-bucket", s3_key)
            tech_stack = event["tech_stack"]
            step_id = log_step(session_id, "Technical Documentation", "In Review", {})
            prompt = (
                f"Generate technical specifications for a {tech_stack['language']} application with "
                f"{tech_stack['backend']} backend and {tech_stack['ui']} UI, based on:\n{requirements}\n"
                f"Include architecture details and implementation steps."
            )
            specs = invoke_claude(prompt)
            mermaid_prompt = f"Generate a Mermaid diagram for the architecture described in:\n{specs}"
            mermaid_code = invoke_claude(mermaid_prompt)
            artifacts = {"specs": "confluence://tech-specs", "mermaid": mermaid_code}
            dynamodb = boto3.resource("dynamodb")
            dynamodb.Table("WorkflowSteps").update_item(
                Key={"session_id": session_id, "step_id": step_id},
                UpdateExpression="SET artifacts = :artifacts",
                ExpressionAttributeValues={":artifacts": artifacts}
            )
            log_transaction(session_id, step_id, "Generate Specs", "Tech Doc Generator", {"specs": specs})
            confluence.create_page(space="SDLC", title=f"Technical Specs {session_id}", body=f"{specs}\n```mermaid\n{mermaid_code}\n```")
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps({"session_id": session_id, "step_id": step_id, "specs": specs})
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"session_id": session_id, "step_id": step_id, "specs": specs})
            }
        except ClientError as e:
            log_transaction(session_id, step_id, "Error", "Tech Doc Generator", {"error": str(e)})
            raise
    ```

### Step 3: Task Breakdown and Jira Creation
- **Objective**: Decompose specs into tasks and create ESTJIRA tickets.
- **Components**:
  - **Lambda (Task Orchestrator)**: Generates tasks and tickets.
  - **ESTJIRA**: Stores tickets.
- **HITL**: Users prioritize tasks in ReactJS.
- **Code**:
  - **lambda/task_orchestrator/lambda_function.py**:
    ```python
    """Lambda function for task breakdown and Jira ticket creation."""
    import json
    import boto3
    from botocore.exceptions import ClientError
    from utils.db import log_step, log_transaction
    from utils.tools import invoke_claude, jira

    sqs = boto3.client("sqs", config=Config(retries={"max_attempts": 3}))
    QUEUE_URL = "your-sqs-queue-url"

    def lambda_handler(event: dict, context: object) -> dict:
        """Decompose specs into tasks and create Jira tickets.

        Args:
            event: Lambda event containing session_id, specs, tech_stack.
            context: Lambda context object.

        Returns:
            Dictionary with status code and response body.

        Raises:
            ClientError: If AWS service calls fail after retries.
        """
        try:
            session_id = event["session_id"]
            specs = event["specs"]
            tech_stack = event["tech_stack"]
            step_id = log_step(session_id, "Task Breakdown", "In Review", {})
            prompt = (
                f"Decompose the following specs into development tasks for a {tech_stack['language']} "
                f"application with {tech_stack['backend']} backend and {tech_stack['ui']} UI:\n{specs}"
            )
            tasks = invoke_claude(prompt)
            ticket_ids = []
            for task in tasks.split("\n"):
                if task.strip():
                    ticket = jira.issue_create(fields={
                        "project": {"key": "SDLC"},
                        "summary": task,
                        "issuetype": {"name": "Task"}
                    })
                    ticket_ids.append(ticket["key"])
            artifacts = {"tickets": ticket_ids}
            dynamodb = boto3.resource("dynamodb")
            dynamodb.Table("WorkflowSteps").update_item(
                Key={"session_id": session_id, "step_id": step_id},
                UpdateExpression="SET artifacts = :artifacts",
                ExpressionAttributeValues={":artifacts": artifacts}
            )
            log_transaction(session_id, step_id, "Create Tickets", "Task Orchestrator", {"tickets": ticket_ids})
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps({"session_id": session_id, "step_id": step_id, "tickets": ticket_ids})
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"session_id": session_id, "step_id": step_id, "tickets": ticket_ids})
            }
        except ClientError as e:
            log_transaction(session_id, step_id, "Error", "Task Orchestrator", {"error": str(e)})
            raise
    ```

### Step 4: Architecture and Environment Setup
- **Objective**: Set up GitLab repository, CI/CD pipeline, and EKS environment.
- **Components**:
  - **Lambda (Environment Setup)**: Configures GitLab and CI/CD.
  - **GitLab**: Hosts repository.
- **HITL**: Users confirm setup in ReactJS.
- **Code**:
  - **lambda/env_setup/lambda_function.py**:
    ```python
    """Lambda function for environment setup."""
    import json
    import boto3
    from botocore.exceptions import ClientError
    from utils.db import log_step, log_transaction
    from utils.tools import gl

    sqs = boto3.client("sqs", config=Config(retries={"max_attempts": 3}))
    QUEUE_URL = "your-sqs-queue-url"

    def lambda_handler(event: dict, context: object) -> dict:
        """Set up GitLab repository and CI/CD pipeline.

        Args:
            event: Lambda event containing session_id, tech_stack.
            context: Lambda context object.

        Returns:
            Dictionary with status code and response body.

        Raises:
            ClientError: If AWS service calls fail after retries.
        """
        try:
            session_id = event["session_id"]
            tech_stack = event["tech_stack"]
            step_id = log_step(session_id, "Environment Setup", "In Review", {})
            project = gl.projects.create({"name": f"sdlc-project-{session_id}"})
            ci_config = """
stages:
  - lint
  - test
  - build
lint:
  stage: lint
  script:
    - pip install black
    - black --check .
test:
  stage: test
  script:
    - pip install pytest
    - pytest
build:
  stage: build
  script:
    - echo "Building..."
"""
            project.files.create({
                "file_path": ".gitlab-ci.yml",
                "branch": "main",
                "content": ci_config,
                "commit_message": "Add CI config"
            })
            requirements = "fastapi==0.115.2\nboto3==1.35.24\npytest==8.3.3"
            project.files.create({
                "file_path": "requirements.txt",
                "branch": "main",
                "content": requirements,
                "commit_message": "Add requirements"
            })
            artifacts = {"project_id": project.id, "ci_config": ci_config}
            dynamodb = boto3.resource("dynamodb")
            dynamodb.Table("WorkflowSteps").update_item(
                Key={"session_id": session_id, "step_id": step_id},
                UpdateExpression="SET artifacts = :artifacts",
                ExpressionAttributeValues={":artifacts": artifacts}
            )
            log_transaction(session_id, step_id, "Setup Environment", "Environment Setup", artifacts)
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps({"session_id": session_id, "step_id": step_id, "project_id": project.id})
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"session_id": session_id, "step_id": step_id, "project_id": project.id})
            }
        except ClientError as e:
            log_transaction(session_id, step_id, "Error", "Environment Setup", {"error": str(e)})
            raise
    ```

### Step 5: Code Development
- **Objective**: Generate code based on tasks and tech stack.
- **Components**:
  - **Lambda (Code Generator)**: Uses Claude to generate code.
  - **GitLab**: Stores code.
  - **SonarQube**: Checks code quality.
- **HITL**: Users review code in ReactJS.
- **Code**:
  - **lambda/code_generator/lambda_function.py**:
    ```python
    """Lambda function for code generation."""
    import json
    import boto3
    from botocore.exceptions import ClientError
    from utils.db import log_step, log_transaction
    from utils.tools import gl, invoke_claude

    sqs = boto3.client("sqs", config=Config(retries={"max_attempts": 3}))
    QUEUE_URL = "your-sqs-queue-url"

    def lambda_handler(event: dict, context: object) -> dict:
        """Generate code for a task using Bedrock Claude.

        Args:
            event: Lambda event containing session_id, task, tech_stack.
            context: Lambda context object.

        Returns:
            Dictionary with status code and response body.

        Raises:
            ClientError: If AWS service calls fail after retries.
        """
        try:
            session_id = event["session_id"]
            task = event["task"]
            tech_stack = event["tech_stack"]
            step_id = log_step(session_id, "Code Development", "In Review", {})
            prompt = (
                f"Generate {tech_stack['language']} code for a {tech_stack['backend']} backend "
                f"for task: {task}. Ensure compatibility with {tech_stack['ui']} UI."
            )
            code = invoke_claude(prompt)
            project = gl.projects.get(f"sdlc-project-{session_id}")
            project.files.create({
                "file_path": "main.py",
                "branch": "feature/task",
                "content": code,
                "commit_message": f"Add code for {task}"
            })
            artifacts = {"code_path": "gitlab://main.py"}
            dynamodb = boto3.resource("dynamodb")
            dynamodb.Table("WorkflowSteps").update_item(
                Key={"session_id": session_id, "step_id": step_id},
                UpdateExpression="SET artifacts = :artifacts",
                ExpressionAttributeValues={":artifacts": artifacts}
            )
            log_transaction(session_id, step_id, "Generate Code", "Code Generator", {"task": task})
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps({"session_id": session_id, "step_id": step_id, "code": code})
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"session_id": session_id, "step_id": step_id, "code": code})
            }
        except ClientError as e:
            log_transaction(session_id, step_id, "Error", "Code Generator", {"error": str(e)})
            raise
    ```

### Step 6: Test Case Generation and Testing
- **Objective**: Generate and run test cases for the code.
- **Components**:
  - **Lambda (Test Automation)**: Generates Pytest cases.
  - **GitLab CI**: Executes tests.
- **HITL**: Users review test results in ReactJS.
- **Code**:
  - **lambda/test_automation/lambda_function.py**:
    ```python
    """Lambda function for test case generation."""
    import json
    import boto3
    from botocore.exceptions import ClientError
    from utils.db import log_step, log_transaction
    from utils.tools import gl, invoke_claude

    sqs = boto3.client("sqs", config=Config(retries={"max_attempts": 3}))
    QUEUE_URL = "your-sqs-queue-url"

    def lambda_handler(event: dict, context: object) -> dict:
        """Generate Pytest cases and commit to GitLab.

        Args:
            event: Lambda event containing session_id, code, tech_stack.
            context: Lambda context object.

        Returns:
            Dictionary with status code and response body.

        Raises:
            ClientError: If AWS service calls fail after retries.
        """
        try:
            session_id = event["session_id"]
            code = event["code"]
            tech_stack = event["tech_stack"]
            step_id = log_step(session_id, "Testing", "In Review", {})
            prompt = (
                f"Generate Pytest cases for {tech_stack['language']} code with {tech_stack['backend']} backend:\n{code}\n"
                f"Cover key functionalities specified in the requirements."
            )
            tests = invoke_claude(prompt)
            project = gl.projects.get(f"sdlc-project-{session_id}")
            project.files.create({
                "file_path": "test_main.py",
                "branch": "feature/task",
                "content": tests,
                "commit_message": "Add tests"
            })
            artifacts = {"tests": "gitlab://test_main.py"}
            dynamodb = boto3.resource("dynamodb")
            dynamodb.Table("WorkflowSteps").update_item(
                Key={"session_id": session_id, "step_id": step_id},
                UpdateExpression="SET artifacts = :artifacts",
                ExpressionAttributeValues={":artifacts": artifacts}
            )
            log_transaction(session_id, step_id, "Generate Tests", "Test Automation", {"tests": tests})
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps({"session_id": session_id, "step_id": step_id, "tests": tests})
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"session_id": session_id, "step_id": step_id, "tests": tests})
            }
        except ClientError as e:
            log_transaction(session_id, step_id, "Error", "Test Automation", {"error": str(e)})
            raise
    ```

### Step 7: Code Check-in and Version Control
- **Objective**: Commit code to GitLab.
- **Components**:
  - **Lambda (Version Control)**: Manages commits.
  - **GitLab**: Version control.
- **HITL**: Users confirm commits in ReactJS.
- **Code**:
  - **lambda/version_control/lambda_function.py**:
    ```python
    """Lambda function for version control."""
    import json
    import boto3
    from botocore.exceptions import ClientError
    from utils.db import log_step, log_transaction
    from utils.tools import gl, invoke_claude

    sqs = boto3.client("sqs", config=Config(retries={"max_attempts": 3}))
    QUEUE_URL = "your-sqs-queue-url"

    def lambda_handler(event: dict, context: object) -> dict:
        """Commit code to GitLab with semantic commit message.

        Args:
            event: Lambda event containing session_id.
            context: Lambda context object.

        Returns:
            Dictionary with status code and response body.

        Raises:
            ClientError: If AWS service calls fail after retries.
        """
        try:
            session_id = event["session_id"]
            step_id = log_step(session_id, "Version Control", "In Review", {})
            commit_message = invoke_claude("Generate semantic commit message for the project code")
            project = gl.projects.get(f"sdlc-project-{session_id}")
            project.commits.create({
                "branch": "main",
                "commit_message": commit_message,
                "actions": []
            })
            artifacts = {"commit_message": commit_message}
            dynamodb = boto3.resource("dynamodb")
            dynamodb.Table("WorkflowSteps").update_item(
                Key={"session_id": session_id, "step_id": step_id},
                UpdateExpression="SET artifacts = :artifacts",
                ExpressionAttributeValues={":artifacts": artifacts}
            )
            log_transaction(session_id, step_id, "Commit Code", "Version Control", {"commit_message": commit_message})
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps({"session_id": session_id, "step_id": step_id, "commit_message": commit_message})
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"session_id": session_id, "step_id": step_id, "commit_message": commit_message})
            }
        except ClientError as e:
            log_transaction(session_id, step_id, "Error", "Version Control", {"error": str(e)})
            raise
    ```

### Step 8: Build and Deployment Package Creation
- **Objective**: Create a Docker image for the application.
- **Components**:
  - **Lambda (Build)**: Generates Dockerfile.
  - **Docker/NEXUS**: Stores image.
- **HITL**: Users confirm build in ReactJS.
- **Code**:
  - **lambda/build/lambda_function.py**:
    ```python
    """Lambda function for build process."""
    import json
    import boto3
    from botocore.exceptions import ClientError
    from utils.db import log_step, log_transaction
    from utils.tools import gl

    sqs = boto3.client("sqs", config=Config(retries={"max_attempts": 3}))
    QUEUE_URL = "your-sqs-queue-url"

    def lambda_handler(event: dict, context: object) -> dict:
        """Create Dockerfile and commit to GitLab.

        Args:
            event: Lambda event containing session_id, tech_stack.
            context: Lambda context object.

        Returns:
            Dictionary with status code and response body.

        Raises:
            ClientError: If AWS service calls fail after retries.
        """
        try:
            session_id = event["session_id"]
            tech_stack = event["tech_stack"]
            step_id = log_step(session_id, "Build", "In Review", {})
            dockerfile = """
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
            project = gl.projects.get(f"sdlc-project-{session_id}")
            project.files.create({
                "file_path": "Dockerfile",
                "branch": "main",
                "content": dockerfile,
                "commit_message": "Add Dockerfile"
            })
            artifacts = {"dockerfile": "gitlab://Dockerfile"}
            dynamodb = boto3.resource("dynamodb")
            dynamodb.Table("WorkflowSteps").update_item(
                Key={"session_id": session_id, "step_id": step_id},
                UpdateExpression="SET artifacts = :artifacts",
                ExpressionAttributeValues={":artifacts": artifacts}
            )
            log_transaction(session_id, step_id, "Build Docker", "Build Agent", {"dockerfile": dockerfile})
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps({"session_id": session_id, "step_id": step_id, "dockerfile": dockerfile})
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"session_id": session_id, "step_id": step_id, "dockerfile": dockerfile})
            }
        except ClientError as e:
            log_transaction(session_id, step_id, "Error", "Build Agent", {"error": str(e)})
            raise
    ```

### Step 9: Deployment to Amazon EKS
- **Objective**: Deploy the application to EKS.
- **Components**:
  - **Lambda (Deployment)**: Creates Kubernetes manifests.
  - **EKS**: Hosts application.
- **HITL**: Users confirm deployment in ReactJS.
- **Code**:
  - **lambda/deployment/lambda_function.py**:
    ```python
    """Lambda function for EKS deployment."""
    import json
    import boto3
    from botocore.exceptions import ClientError
    from utils.db import log_step, log_transaction
    from utils.tools import gl

    sqs = boto3.client("sqs", config=Config(retries={"max_attempts": 3}))
    QUEUE_URL = "your-sqs-queue-url"

    def lambda_handler(event: dict, context: object) -> dict:
        """Create Kubernetes manifest and commit to GitLab.

        Args:
            event: Lambda event containing session_id, tech_stack.
            context: Lambda context object.

        Returns:
            Dictionary with status code and response body.

        Raises:
            ClientError: If AWS service calls fail after retries.
        """
        try:
            session_id = event["session_id"]
            tech_stack = event["tech_stack"]
            step_id = log_step(session_id, "Deployment", "In Review", {})
            manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sdlc-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sdlc-app
  template:
    metadata:
      labels:
        app: sdlc-app
    spec:
      containers:
      - name: sdlc-app
        image: nexus.your-org.com/sdlc-app:latest
        ports:
        - containerPort: 8000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sdlc-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sdlc-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
            project = gl.projects.get(f"sdlc-project-{session_id}")
            project.files.create({
                "file_path": "k8s/deployment.yaml",
                "branch": "main",
                "content": manifest,
                "commit_message": "Add K8s manifest with HPA"
            })
            artifacts = {"manifest": "gitlab://k8s/deployment.yaml"}
            dynamodb = boto3.resource("dynamodb")
            dynamodb.Table("WorkflowSteps").update_item(
                Key={"session_id": session_id, "step_id": step_id},
                UpdateExpression="SET artifacts = :artifacts",
                ExpressionAttributeValues={":artifacts": artifacts}
            )
            log_transaction(session_id, step_id, "Deploy to EKS", "Deployment Agent", {"manifest": manifest})
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps({"session_id": session_id, "step_id": step_id, "manifest": manifest})
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"session_id": session_id, "step_id": step_id, "manifest": manifest})
            }
        except ClientError as e:
            log_transaction(session_id, step_id, "Error", "Deployment Agent", {"error": str(e)})
            raise
    ```

### Step 10: Post-Deployment Monitoring and Maintenance
- **Objective**: Monitor the application and create maintenance tasks.
- **Components**:
  - **Lambda (Monitoring)**: Monitors metrics and creates tickets.
  - **CloudWatch**: Logs metrics.
  - **ESTJIRA**: Stores tasks.
- **HITL**: Users review tasks in ReactJS.
- **Code**:
  - **lambda/monitoring/lambda_function.py**:
    ```python
    """Lambda function for monitoring."""
    import json
    import boto3
    from botocore.exceptions import ClientError
    from datetime import datetime, timedelta
    from utils.db import log_step, log_transaction
    from utils.tools import jira

    sqs = boto3.client("sqs", config=Config(retries={"max_attempts": 3}))
    cloudwatch = boto3.client("cloudwatch", config=Config(retries={"max_attempts": 3}))
    QUEUE_URL = "your-sqs-queue-url"

    def lambda_handler(event: dict, context: object) -> dict:
        """Monitor EKS application and create maintenance tasks.

        Args:
            event: Lambda event containing session_id.
            context: Lambda context object.

        Returns:
            Dictionary with status code and response body.

        Raises:
            ClientError: If AWS service calls fail after retries.
        """
        try:
            session_id = event["session_id"]
            step_id = log_step(session_id, "Monitoring", "In Review", {})
            # Create CloudWatch alarm
            cloudwatch.put_metric_alarm(
                AlarmName=f"SDLCAppHighCPU-{session_id}",
                ComparisonOperator="GreaterThanThreshold",
                EvaluationPeriods=2,
                MetricName="pod_cpu_utilization",
                Namespace="AWS/EKS",
                Period=300,
                Threshold=80.0,
                AlarmActions=["your-sns-topic-arn"],
                Dimensions=[{"Name": "ClusterName", "Value": "sdlc-cluster"}]
            )
            # Retrieve metrics
            metrics = cloudwatch.get_metric_data(
                MetricDataQueries=[{
                    "Id": "m1",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": "AWS/EKS",
                            "MetricName": "pod_cpu_utilization",
                            "Dimensions": [{"Name": "ClusterName", "Value": "sdlc-cluster"}]
                        },
                        "Period": 300,
                        "Stat": "Average"
                    }
                }],
                StartTime=datetime.utcnow() - timedelta(minutes=60),
                EndTime=datetime.utcnow()
            )
            tasks = ["Optimize performance", "Review application logs"]
            ticket_ids = []
            for task in tasks:
                ticket = jira.issue_create(fields={
                    "project": {"key": "SDLC"},
                    "summary": task,
                    "issuetype": {"name": "Task"}
                })
                ticket_ids.append(ticket["key"])
            artifacts = {"tickets": ticket_ids}
            dynamodb = boto3.resource("dynamodb")
            dynamodb.Table("WorkflowSteps").update_item(
                Key={"session_id": session_id, "step_id": step_id},
                UpdateExpression="SET artifacts = :artifacts",
                ExpressionAttributeValues={":artifacts": artifacts}
            )
            log_transaction(session_id, step_id, "Monitor App", "Monitoring Agent", {"tickets": ticket_ids})
            sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps({"session_id": session_id, "step_id": step_id, "tickets": ticket_ids})
            )
            return {
                "statusCode": 200,
                "body": json.dumps({"session_id": session_id, "step_id": step_id, "tickets": ticket_ids})
            }
        except ClientError as e:
            log_transaction(session_id, step_id, "Error", "Monitoring Agent", {"error": str(e)})
            raise
    ```

### Backend (FastAPI)
- **Objective**: Serve APIs for the ReactJS frontend and workflow.
- **Code**:
  - **backend/main.py**:
    ```python
    """FastAPI backend for SDLC application."""
    from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    import boto3
    import json
    import jwt
    from utils.db import create_session, log_feedback
    from utils.tools import upload_s3_file, get_secret
    from datetime import datetime
    from botocore.config import Config

    app = FastAPI(title="SDLC Workflow API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    sfn = boto3.client("stepfunctions", config=Config(retries={"max_attempts": 3}))
    dynamodb = boto3.resource("dynamodb", config=Config(retries={"max_attempts": 3}))
    STATE_MACHINE_ARN = "your-step-functions-arn"
    security = HTTPBearer()
    SECRET_KEY = get_secret("JWT_SECRET_KEY")["key"]

    def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
        """Verify JWT token.

        Args:
            credentials: HTTP authorization credentials.

        Returns:
            Decoded token payload.

        Raises:
            HTTPException: If token is invalid.
        """
        try:
            payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    class SessionRequest(BaseModel):
        user_id: str
        tech_stack: dict
    class FeedbackRequest(BaseModel):
        session_id: str
        step_id: str
        user_id: str
        feedback: str

    @app.post("/start-session", summary="Start a new SDLC session")
    async def start_session(req: SessionRequest, payload: dict = Depends(verify_token)):
        """Start a new SDLC session and trigger Step Functions workflow.

        Args:
            req: Session request with user_id and tech_stack.
            payload: JWT token payload.

        Returns:
            Dictionary with session_id.
        """
        session_id = create_session(req.user_id, req.tech_stack)
        sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            input=json.dumps({
                "session_id": session_id,
                "document_path": f"s3://your-bucket/requirements-{session_id}.md",
                "tech_stack": req.tech_stack
            })
        )
        return {"session_id": session_id}

    @app.post("/upload-requirements", summary="Upload requirement document")
    async def upload_requirements(file: UploadFile = File(...), session_id: str = "", payload: dict = Depends(verify_token)):
        """Upload requirement document to S3.

        Args:
            file: Uploaded file.
            session_id: Session identifier.
            payload: JWT token payload.

        Returns:
            Dictionary with upload status and document path.
        """
        content = await file.read()
        key = f"requirements-{session_id}.md"
        upload_s3_file(content, "your-bucket", key)
        return {"message": "Document uploaded", "document_path": f"s3://your-bucket/{key}"}

    @app.post("/submit-feedback", summary="Submit HITL feedback")
    async def submit_feedback(req: FeedbackRequest, payload: dict = Depends(verify_token)):
        """Submit HITL feedback for a step.

        Args:
            req: Feedback request with session_id, step_id, user_id, feedback.
            payload: JWT token payload.

        Returns:
            Dictionary with submission status.
        """
        log_feedback(req.session_id, req.step_id, req.user_id, req.feedback)
        return {"message": "Feedback submitted"}

    @app.post("/approve-step/{session_id}/{step_id}", summary="Approve a workflow step")
    async def approve_step(session_id: str, step_id: str, payload: dict = Depends(verify_token)):
        """Approve a workflow step and advance to next step.

        Args:
            session_id: Session identifier.
            step_id: Step identifier.
            payload: JWT token payload.

        Returns:
            Dictionary with approval status.
        """
        table = dynamodb.Table("WorkflowSteps")
        table.update_item(
            Key={"session_id": session_id, "step_id": step_id},
            UpdateExpression="SET #status = :status, completed_at = :completed_at",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": "Approved",
                ":completed_at": datetime.utcnow().isoformat()
            }
        )
        table = dynamodb.Table("Sessions")
        current_step = table.get_item(Key={"session_id": session_id})["Item"]["current_step"]
        next_steps = {
            "Requirement Validation": "Technical Documentation",
            "Technical Documentation": "Task Breakdown",
            "Task Breakdown": "Environment Setup",
            "Environment Setup": "Code Development",
            "Code Development": "Testing",
            "Testing": "Version Control",
            "Version Control": "Build",
            "Build": "Deployment",
            "Deployment": "Monitoring",
            "Monitoring": "Completed"
        }
        table.update_item(
            Key={"session_id": session_id},
            UpdateExpression="SET current_step = :step, status = :status",
            ExpressionAttributeValues={
                ":step": next_steps.get(current_step, "Completed"),
                ":status": "Completed" if next_steps.get(current_step) == "Completed" else "Active"
            }
        )
        return {"message": "Step approved"}
    ```

### Frontend (ReactJS)
- **Objective**: Provide HITL interface with JWT authentication.
- **Code**:
  - **frontend/index.html**:
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>SDLC Workflow</title>
      <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.development.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.development.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/axios@1.4.0/dist/axios.min.js"></script>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body>
      <div id="root"></div>
      <script type="text/babel">
        /** @jsx React.createElement */
        const { useState } = React;

        /**
         * Main App component for SDLC Workflow UI.
         * Handles user login, session creation, file uploads, and feedback.
         */
        const App = () => {
          const [userId, setUserId] = useState("");
          const [token, setToken] = useState("");
          const [techStack, setTechStack] = useState({ language: "Python", backend: "FastAPI", ui: "None" });
          const [sessionId, setSessionId] = useState(null);
          const [file, setFile] = useState(null);
          const [report, setReport] = useState("");
          const [feedback, setFeedback] = useState("");
          const [stepId, setStepId] = useState(null);
          const [error, setError] = useState("");

          /**
           * Log in user and retrieve JWT token.
           */
          const login = async () => {
            try {
              // Placeholder: Replace with actual login API
              const response = await axios.post("http://localhost:8000/login", { userId });
              setToken(response.data.token);
              setError("");
            } catch (err) {
              setError("Login failed");
            }
          };

          /**
           * Start a new SDLC session.
           */
          const startSession = async () => {
            try {
              const response = await axios.post("http://localhost:8000/start-session", {
                user_id: userId,
                tech_stack: techStack
              }, {
                headers: { Authorization: `Bearer ${token}` }
              });
              setSessionId(response.data.session_id);
              setError("");
            } catch (err) {
              setError("Failed to start session");
            }
          };

          /**
           * Upload requirement document.
           */
          const uploadFile = async () => {
            try {
              const formData = new FormData();
              formData.append("file", file);
              await axios.post(`http://localhost:8000/upload-requirements?session_id=${sessionId}`, formData, {
                headers: { Authorization: `Bearer ${token}` }
              });
              // Poll for validation report (simplified)
              setTimeout(async () => {
                const response = await axios.get("http://localhost:8000/validate", {
                  headers: { Authorization: `Bearer ${token}` }
                });
                setReport(response.data.report);
                setStepId(response.data.step_id);
                setError("");
              }, 5000);
            } catch (err) {
              setError("Failed to upload file");
            }
          };

          /**
           * Submit HITL feedback.
           */
          const submitFeedback = async () => {
            try {
              await axios.post("http://localhost:8000/submit-feedback", {
                session_id: sessionId,
                step_id: stepId,
                user_id: userId,
                feedback
              }, {
                headers: { Authorization: `Bearer ${token}` }
              });
              setFeedback("");
              setError("");
            } catch (err) {
              setError("Failed to submit feedback");
            }
          };

          /**
           * Approve a workflow step.
           */
          const approveStep = async () => {
            try {
              await axios.post(`http://localhost:8000/approve-step/${sessionId}/${stepId}`, {}, {
                headers: { Authorization: `Bearer ${token}` }
              });
              setReport("");
              setStepId(null);
              setError("");
            } catch (err) {
              setError("Failed to approve step");
            }
          };

          return (
            <div className="container mx-auto p-4">
              <h1 className="text-2xl font-bold">SDLC Workflow</h1>
              {error && <p className="text-red-500">{error}</p>}
              {!token && (
                <div className="mt-4">
                  <input
                    type="text"
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                    placeholder="User ID"
                    className="border p-2 mr-2"
                  />
                  <button onClick={login} className="bg-blue-500 text-white p-2">Login</button>
                </div>
              )}
              {token && (
                <div className="mt-4">
                  <input
                    type="text"
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                    placeholder="User ID"
                    className="border p-2 mr-2"
                    disabled
                  />
                  <select
                    value={techStack.language}
                    onChange={(e) => setTechStack({ ...techStack, language: e.target.value })}
                    className="border p-2 mr-2"
                  >
                    <option value="Python">Python</option>
                    <option value="Other">Other</option>
                  </select>
                  <select
                    value={techStack.backend}
                    onChange={(e) => setTechStack({ ...techStack, backend: e.target.value })}
                    className="border p-2 mr-2"
                  >
                    <option value="FastAPI">FastAPI</option>
                    <option value="Flask">Flask</option>
                    <option value="Other">Other</option>
                  </select>
                  <select
                    value={techStack.ui}
                    onChange={(e) => setTechStack({ ...techStack, ui: e.target.value })}
                    className="border p-2 mr-2"
                  >
                    <option value="None">None</option>
                    <option value="Other">Other</option>
                  </select>
                  <button onClick={startSession} className="bg-blue-500 text-white p-2">Start Session</button>
                </div>
              )}
              {sessionId && (
                <div className="mt-4">
                  <input
                    type="file"
                    onChange={(e) => setFile(e.target.files[0])}
                    className="mb-2"
                  />
                  <button onClick={uploadFile} className="bg-green-500 text-white p-2">Upload Requirements</button>
                </div>
              )}
              {report && (
                <div className="mt-4">
                  <h2 className="text-xl">Validation Report</h2>
                  <p>{report}</p>
                  <textarea
                    value={feedback}
                    onChange={(e) => setFeedback(e.target.value)}
                    placeholder="Provide Feedback"
                    className="border p-2 w-full mt-2"
                  ></textarea>
                  <button onClick={submitFeedback} className="bg-yellow-500 text-white p-2 mt-2">Submit Feedback</button>
                  <button onClick={approveStep} className="bg-green-500 text-white p-2 mt-2 ml-2">Approve</button>
                </div>
              )}
            </div>
          );
        };

        ReactDOM.render(<App />, document.getElementById("root"));
      </script>
    </body>
    </html>
    ```

## Step Functions State Machine

### Objective
Orchestrate the 10-step SDLC workflow, invoking Lambda functions for each step, handling HITL approvals, retries, and errors, and integrating with SQS, DynamoDB, and external tools.

### SDLC Steps
1. **Requirement Validation**: Validate requirement document (Lambda: `requirement_validator`).
2. **Technical Documentation**: Generate specs and diagrams (Lambda: `tech_doc_generator`).
3. **Task Breakdown**: Create Jira tickets (Lambda: `task_orchestrator`).
4. **Environment Setup**: Set up GitLab repository (Lambda: `env_setup`).
5. **Code Development**: Generate code (Lambda: `code_generator`).
6. **Testing**: Generate and run tests (Lambda: `test_automation`).
7. **Version Control**: Commit to GitLab (Lambda: `version_control`).
8. **Build**: Create Docker image (Lambda: `build`).
9. **Deployment**: Deploy to EKS (Lambda: `deployment`).
10. **Monitoring**: Set up CloudWatch alarms (Lambda: `monitoring`).

### State Machine Definition
The state machine uses AWS Step Functions to orchestrate Lambda invocations, with retry logic (3 attempts, exponential backoff), error handling (dead-letter queue via SQS), and HITL approval waits (1-hour polling). Each step invokes a Lambda function, logs to DynamoDB, and queues tasks via SQS. Approval is checked via DynamoDB `WorkflowSteps` table status.

- **File**: `step_functions/sdlc_workflow.json`
- **Code**:
- **step_functions/sdlc_workflow.json**:
  ```json
  {
    "Comment": "Agentic SDLC Workflow orchestrating 10-step development process",
    "StartAt": "RequirementValidation",
    "States": {
      "RequirementValidation": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:requirement_validator",
        "Parameters": {
          "session_id.$": "$.session_id",
          "document_path.$": "$.document_path",
          "tech_stack.$": "$.tech_stack"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "ErrorHandler"
          }
        ],
        "Next": "WaitForValidationApproval"
      },
      "WaitForValidationApproval": {
        "Type": "Wait",
        "Seconds": 3600,
        "Next": "CheckValidationApproval"
      },
      "CheckValidationApproval": {
        "Type": "Choice",
        "Choices": [
          {
            "Variable": "$.status",
            "StringEquals": "Approved",
            "Next": "TechnicalDocumentation"
          }
        ],
        "Default": "WaitForValidationApproval"
      },
      "TechnicalDocumentation": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:tech_doc_generator",
        "Parameters": {
          "session_id.$": "$.session_id",
          "document_path.$": "$.document_path",
          "tech_stack.$": "$.tech_stack"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "ErrorHandler"
          }
        ],
        "Next": "WaitForTechDocApproval"
      },
      "WaitForTechDocApproval": {
        "Type": "Wait",
        "Seconds": 3600,
        "Next": "CheckTechDocApproval"
      },
      "CheckTechDocApproval": {
        "Type": "Choice",
        "Choices": [
          {
            "Variable": "$.status",
            "StringEquals": "Approved",
            "Next": "TaskBreakdown"
          }
        ],
        "Default": "WaitForTechDocApproval"
      },
      "TaskBreakdown": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:task_orchestrator",
        "Parameters": {
          "session_id.$": "$.session_id",
          "specs.$": "$.specs",
          "tech_stack.$": "$.tech_stack"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "ErrorHandler"
          }
        ],
        "Next": "WaitForTaskApproval"
      },
      "WaitForTaskApproval": {
        "Type": "Wait",
        "Seconds": 3600,
        "Next": "CheckTaskApproval"
      },
      "CheckTaskApproval": {
        "Type": "Choice",
        "Choices": [
          {
            "Variable": "$.status",
            "StringEquals": "Approved",
            "Next": "EnvironmentSetup"
          }
        ],
        "Default": "WaitForTaskApproval"
      },
      "EnvironmentSetup": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:env_setup",
        "Parameters": {
          "session_id.$": "$.session_id",
          "tech_stack.$": "$.tech_stack"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "ErrorHandler"
          }
        ],
        "Next": "WaitForEnvSetupApproval"
      },
      "WaitForEnvSetupApproval": {
        "Type": "Wait",
        "Seconds": 3600,
        "Next": "CheckEnvSetupApproval"
      },
      "CheckEnvSetupApproval": {
        "Type": "Choice",
        "Choices": [
          {
            "Variable": "$.status",
            "StringEquals": "Approved",
            "Next": "CodeDevelopment"
          }
        ],
        "Default": "WaitForEnvSetupApproval"
      },
      "CodeDevelopment": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:code_generator",
        "Parameters": {
          "session_id.$": "$.session_id",
          "task.$": "$.task",
          "tech_stack.$": "$.tech_stack"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalsSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "ErrorHandler"
          }
        ],
        "Next": "WaitForCodeApproval"
      },
      "WaitForCodeApproval": {
        "Type": "Wait",
        "Seconds": 3600,
        "Next": "CheckCodeApproval"
      },
      "CheckCodeApproval": {
        "Type": "Choice",
        "Choices": [
          {
            "Variable": "$.status",
            "StringEquals": "Approved",
            "Next": "Testing"
          }
        ],
        "Default": "WaitForCodeApproval"
      },
      "Testing": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:test_automation",
        "Parameters": {
          "session_id.$": "$.session_id",
          "code.$": "$.code",
          "tech_stack.$": "$.tech_stack"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "ErrorHandler"
          }
        ],
        "Next": "WaitForTestApproval"
      },
      "WaitForTestApproval": {
        "Type": "Wait",
        "Seconds": 3600,
        "Next": "CheckTestApproval"
      },
      "CheckTestApproval": {
        "Type": "Choice",
        "Choices": [
          {
            "Variable": "$.status",
            "StringEquals": "Approved",
            "Next": "VersionControl"
          }
        ],
        "Default": "WaitForTestApproval"
      },
      "VersionControl": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:version_control",
        "Parameters": {
          "session_id.$": "$.session_id"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "ErrorHandler"
          }
        ],
        "Next": "WaitForVersionControlApproval"
      },
      "WaitForVersionControlApproval": {
        "Type": "Wait",
        "Seconds": 3600,
        "Next": "CheckVersionControlApproval"
      },
      "CheckVersionControlApproval": {
        "Type": "Choice",
        "Choices": [
          {
            "Variable": "$.status",
            "StringEquals": "Approved",
            "Next": "Build"
          }
        ],
        "Default": "WaitForVersionControlApproval"
      },
      "Build": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:build",
        "Parameters": {
          "session_id.$": "$.session_id",
          "tech_stack.$": "$.tech_stack"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "ErrorHandler"
          }
        ],
        "Next": "WaitForBuildApproval"
      },
      "WaitForBuildApproval": {
        "Type": "Wait",
        "Seconds": 3600,
        "Next": "CheckBuildApproval"
      },
      "CheckBuildApproval": {
        "Type": "Choice",
        "Choices": [
          {
            "Variable": "$.status",
            "StringEquals": "Approved",
            "Next": "Deployment"
          }
        ],
        "Default": "WaitForBuildApproval"
      },
      "Deployment": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:deployment",
        "Parameters": {
          "session_id.$": "$.session_id",
          "tech_stack.$": "$.tech_stack"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "ErrorHandler"
          }
        ],
        "Next": "WaitForDeploymentApproval"
      },
      "WaitForDeploymentApproval": {
        "Type": "Wait",
        "Seconds": 3600,
        "Next": "CheckDeploymentApproval"
      },
      "CheckDeploymentApproval": {
        "Type": "Choice",
        "Choices": [
          {
            "Variable": "$.status",
            "StringEquals": "Approved",
            "Next": "Monitoring"
          }
        ],
        "Default": "WaitForDeploymentApproval"
      },
      "Monitoring": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:monitoring",
        "Parameters": {
          "session_id.$": "$.session_id"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "ResultPath": "$.error",
            "Next": "ErrorHandler"
          }
        ],
        "End": true
      },
      "ErrorHandler": {
        "Type": "Task",
        "Resource": "arn:aws:lambda:region:account-id:function:error_handler",
        "Parameters": {
          "session_id.$": "$.session_id",
          "error.$": "$.error"
        },
        "Retry": [
          {
            "ErrorEquals": ["States.TaskFailed"],
            "IntervalSeconds": 2,
            "MaxAttempts": 3,
            "BackoffRate": 2.0
          }
        ],
        "Catch": [
          {
            "ErrorEquals": ["States.ALL"],
            "Next": "DeadLetterQueue"
          }
        ],
        "End": true
      },
      "DeadLetterQueue": {
        "Type": "Task",
        "Resource": "arn:aws:states:::sqs:sendMessage",
        "Parameters": {
          "QueueUrl": "your-dlq-url",
          "MessageBody.$": "$"
        },
        "End": true
      }
    }
  }
  ```

### Error Handling
- **Retries**: Each task state retries on `States.TaskFailed` errors (3 attempts, 2-second initial delay, 2x backoff).
- **Catch**: Unhandled errors route to `ErrorHandler` Lambda, which logs to DynamoDB (`Transactions` table).
- **Dead-Letter Queue**: Persistent failures are sent to an SQS dead-letter queue (`DeadLetterQueue` state).

### HITL Approval
- **Wait States**: 1-hour wait (`WaitForXApproval`) before checking approval status.
- **Choice States**: Check `$.status` in DynamoDB `WorkflowSteps` table (`Approved` advances to next step).
- **FastAPI Endpoint**: `/approve-step/{session_id}/{step_id}` updates `WorkflowSteps` status to `Approved`.

## Deployment

### Deployment Script
Deploys the Step Functions state machine, SQS queues, and DynamoDB tables using AWS CLI.

- **File**: `deploy.sh`
- **Code**:
  ```bash
  #!/bin/bash

  # Deploy Agentic SDLC Application
  # Prerequisites: AWS CLI configured, jq installed

  set -e

  # Configuration
  REGION="us-east-1"
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  BUCKET="your-bucket"
  QUEUE_NAME="sdlc-task-queue"
  DLQ_NAME="sdlc-dlq"
  STATE_MACHINE_NAME="SDLCWorkflow"

  # Create S3 bucket
  aws s3 mb s3://$BUCKET --region $REGION || echo "Bucket already exists"

  # Create SQS queues
  QUEUE_URL=$(aws sqs create-queue --queue-name $QUEUE_NAME --region $REGION --query QueueUrl --output text)
  DLQ_URL=$(aws sqs create-queue --queue-name $DLQ_NAME --region $REGION --query QueueUrl --output text)
  echo "Task Queue URL: $QUEUE_URL"
  echo "Dead-Letter Queue URL: $DLQ_URL"

  # Create DynamoDB tables
  aws dynamodb create-table \
    --table-name Sessions \
    --attribute-definitions AttributeName=session_id,AttributeType=S \
    --key-schema AttributeName=session_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region $REGION || echo "Sessions table already exists"

  aws dynamodb create-table \
    --table-name WorkflowSteps \
    --attribute-definitions \
        AttributeName=session_id,AttributeType=S \
        AttributeName=step_id,AttributeType=S \
    --key-schema \
        AttributeName=session_id,KeyType=HASH \
        AttributeName=step_id,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region $REGION || echo "WorkflowSteps table already exists"

  aws dynamodb create-table \
    --table-name Transactions \
    --attribute-definitions \
        AttributeName=session_id,AttributeType=S \
        AttributeName=transaction_id,AttributeType=S \
    --key-schema \
        AttributeName=session_id,KeyType=HASH \
        AttributeName=transaction_id,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region $REGION || echo "Transactions table already exists"

  aws dynamodb create-table \
    --table-name HITLFeedback \
    --attribute-definitions \
        AttributeName=session_id,AttributeType=S \
        AttributeName=feedback_id,AttributeType=S \
    --key-schema \
        AttributeName=session_id,KeyType=HASH \
        AttributeName=feedback_id,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region $REGION || echo "HITLFeedback table already exists"

  # Update state machine definition with ARNs
  sed -i "s|region:account-id|$REGION:$ACCOUNT_ID|g" step_functions/sdlc_workflow.json
  sed -i "s|your-dlq-url|$DLQ_URL|g" step_functions/sdlc_workflow.json

  # Create IAM role for Step Functions
  ROLE_ARN=$(aws iam create-role \
    --role-name SDLCStepFunctionsRole \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"states.'$REGION'.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --query Role.Arn --output text || aws iam get-role --role-name SDLCStepFunctionsRole --query Role.Arn --output text)

  aws iam put-role-policy \
    --role-name SDLCStepFunctionsRole \
    --policy-name SDLCStepFunctionsPolicy \
    --policy-document '{
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": [
            "lambda:InvokeFunction",
            "sqs:SendMessage",
            "dynamodb:PutItem",
            "dynamodb:UpdateItem",
            "dynamodb:GetItem",
            "logs:CreateLogStream",
            "logs:PutLogEvents"
          ],
          "Resource": "*"
        }
      ]
    }'

  # Deploy state machine
  aws stepfunctions create-state-machine \
    --name $STATE_MACHINE_NAME \
    --definition file://step_functions/sdlc_workflow.json \
    --role-arn $ROLE_ARN \
    --region $REGION

  echo "Deployment completed. State Machine ARN: arn:aws:states:$REGION:$ACCOUNT_ID:stateMachine:$STATE_MACHINE_NAME"
  ```

### Prerequisites
- **AWS CLI**: Configured with credentials (`aws configure`).
- **IAM Permissions**: Admin access for creating roles, policies, and resources.
- **Dependencies**:
  - `jq` for JSON parsing (`sudo apt-get install jq` or equivalent).
  - Python: `pip install boto3==1.35.24`.
- **Secrets Manager**: Store `GITLAB_TOKEN`, `CONFLUENCE_CREDENTIALS`, `JIRA_CREDENTIALS`, `JWT_SECRET_KEY`.
- **Environment Variables**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `BEDROCK_REGION=us-east-1`.

### Steps
1. **Clone Repository**:
   ```bash
   git clone your-gitlab-repo
   cd sdlc_aws_workflow
   ```
2. **Set Up Secrets**:
   - Create secrets in AWS Secrets Manager:
     ```bash
     aws secretsmanager create-secret --name GITLAB_TOKEN --secret-string '{"token":"your-gitlab-token"}'
     aws secretsmanager create-secret --name CONFLUENCE_CREDENTIALS --secret-string '{"username":"your-user","password":"your-pass"}'
     aws secretsmanager create-secret --name JIRA_CREDENTIALS --secret-string '{"username":"your-user","password":"your-pass"}'
     aws secretsmanager create-secret --name JWT_SECRET_KEY --secret-string '{"key":"your-jwt-secret"}'
     ```
3. **Run Deployment Script**:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```
4. **Verify Deployment**:
   - Check Step Functions console for `SDLCWorkflow`.
   - Verify SQS queues (`sdlc-task-queue`, `sdlc-dlq`) and DynamoDB tables (`Sessions`, `WorkflowSteps`, `Transactions`, `HITLFeedback`).

## Monitoring and Maintenance

### CloudWatch Setup
- **Alarms**:
  - Step Functions execution failures (`ExecutionFailed` > 0).
  - Lambda errors (`Errors` > 0).
  - SQS queue depth (`ApproximateNumberOfMessagesVisible` > 100).
- **Dashboard**:
  - Metrics: Step Functions execution time, Lambda duration, SQS queue depth, DynamoDB read/write capacity.
  - **Setup Script**:
    ```python
    """Set up CloudWatch alarms and dashboard."""
    import boto3
    from botocore.config import Config

    cloudwatch = boto3.client("cloudwatch", config=Config(retries={"max_attempts": 3}))

    def create_alarms():
        """Create CloudWatch alarms for SDLC application."""
        alarms = [
            {
                "AlarmName": "SDLCExecutionFailed",
                "MetricName": "ExecutionFailed",
                "Namespace": "AWS/States",
                "Threshold": 0.0,
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 1,
                "Period": 300,
                "Statistic": "Sum",
                "Actions": ["your-sns-topic-arn"]
            },
            {
                "AlarmName": "SDLCLambdaErrors",
                "MetricName": "Errors",
                "Namespace": "AWS/Lambda",
                "Threshold": 0.0,
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 1,
                "Period": 300,
                "Statistic": "Sum",
                "Actions": ["your-sns-topic-arn"]
            },
            {
                "AlarmName": "SDLCQueueDepth",
                "MetricName": "ApproximateNumberOfMessagesVisible",
                "Namespace": "AWS/SQS",
                "Threshold": 100.0,
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 1,
                "Period": 300,
                "Statistic": "Average",
                "Dimensions": [{"Name": "QueueName", "Value": "sdlc-task-queue"}],
                "Actions": ["your-sns-topic-arn"]
            }
        ]
        for alarm in alarms:
            cloudwatch.put_metric_alarm(**alarm)
            print(f"Created alarm {alarm['AlarmName']}")

    def create_dashboard():
        """Create CloudWatch dashboard for SDLC application."""
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/States", "ExecutionTime", "StateMachineArn", "arn:aws:states:us-east-1:account-id:stateMachine:SDLCWorkflow"]
                        ],
                        "title": "Step Functions Execution Time",
                        "region": "us-east-1"
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/Lambda", "Duration"]
                        ],
                        "title": "Lambda Duration",
                        "region": "us-east-1"
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/SQS", "ApproximateNumberOfMessagesVisible", "QueueName", "sdlc-task-queue"]
                        ],
                        "title": "SQS Queue Depth",
                        "region": "us-east-1"
                    }
                }
            ]
        }
        cloudwatch.put_dashboard(
            DashboardName="SDLCDashboard",
            DashboardBody=json.dumps(dashboard_body)
        )
        print("Created dashboard SDLCDashboard")

    if __name__ == "__main__":
        create_alarms()
        create_dashboard()
    ```

### Maintenance Tasks
- **Jira Tickets**: Created by `monitoring` Lambda for performance optimization and log reviews.
- **Scaling**:
  - Step Functions: No scaling limits.
  - SQS: Auto-scales with queue depth.
  - DynamoDB: On-demand capacity mode.
- **Security**:
  - IAM roles restrict Lambda/Step Functions access to specific resources.
  - Secrets Manager secures external tool credentials.

## Setup Instructions

### Local Development
1. **Install Dependencies**:
   ```bash
   pip install boto3==1.35.24 fastapi==0.115.2 uvicorn==0.32.0 python-jose[cryptography]==3.3.0 pytest==8.3.3 moto==5.0.12
   cd frontend && npm install react@18.2.0 react-dom@18.2.0 axios@1.4.0 tailwindcss@3.4.1
   ```
2. **Run FastAPI Backend**:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
3. **Run React Frontend**:
   - Serve `frontend/index.html` via a local server (e.g., `python -m http.server 3000`).
4. **Test Lambda Functions**:
   ```bash
   pytest lambda/ --cov=. --cov-report=html
   ```

### AWS Deployment
1. **Configure AWS Resources**:
   - Deploy ElastiCache (Redis), S3 bucket, EKS cluster, API Gateway (optional).
   - Set up EventBridge rule to trigger Step Functions on S3 uploads:
     ```json
     {
       "source": ["aws.s3"],
       "detail-type": ["Object Created"],
       "detail": {
         "bucket": {"name": ["your-bucket"]},
         "object": {"key": [{"prefix": "requirements-"}]}
       }
     }
     ```
2. **Deploy Lambda Functions**:
   - Package each Lambda with dependencies:
     ```bash
     cd lambda/requirement_validator
     zip -r function.zip . -x "*.pyc" -x "__pycache__/*"
     aws lambda create-function \
       --function-name requirement_validator \
       --zip-file fileb://function.zip \
       --handler lambda_function.lambda_handler \
       --runtime python3.11 \
       --role arn:aws:iam::account-id:role/SDLCStepFunctionsRole
     ```
   - Repeat for other Lambdas.
3. **Run Deployment Script** (see Deployment section).
4. **Test Workflow**:
   - Upload a requirement document to `s3://your-bucket/requirements-session123.md`.
   - Monitor Step Functions execution in AWS Console.

## Security Considerations
- **IAM Roles**: Fine-grained permissions for Step Functions and Lambda (e.g., `lambda:InvokeFunction`, `sqs:SendMessage`).
- **Secrets Manager**: Encrypts credentials with AWS KMS.
- **DynamoDB**: Server-side encryption, fine-grained access control.
- **SQS**: Server-side encryption, access policies limiting Lambda access.
- **JWT**: HS256 algorithm, secret stored in Secrets Manager.

## Scalability
- **Step Functions**: Handles thousands of concurrent executions.
- **SQS**: Scales with message volume, batch processing (10 messages).
- **DynamoDB**: On-demand mode auto-scales read/write capacity.
- **Lambda**: 1000 concurrent executions, reserved concurrency per function.

## Testing
- **Unit Tests**: Pytest with `moto` for AWS mocks (e.g., `lambda/requirement_validator/test_lambda_function.py`).
- **Integration Tests**: Test Step Functions workflow with sample inputs.
- **Coverage**: Target >80% (run `pytest --cov=.`).
- **Linting**:
  - Python: `black . && flake8 . --max-line-length=88`
  - JavaScript: `eslint frontend/ && prettier --check frontend/`

## Troubleshooting
- **Execution Fails**: Check CloudWatch Logs for Lambda errors, verify IAM permissions.
- **Approval Delays**: Ensure FastAPI `/approve-step` endpoint is accessible, check DynamoDB `WorkflowSteps` status.
- **SQS Backlog**: Increase Lambda concurrency or adjust visibility timeout (30 seconds).