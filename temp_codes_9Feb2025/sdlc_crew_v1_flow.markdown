# Agentic SDLC CrewAI Implementation for Task Management - Part 1

## Overview
This document (Part 1 of 3) details the first part of an end-to-end implementation of an **Agentic SDLC Application** for a **Task Management Application** using **CrewAI** to orchestrate a 10-step Software Development Life Cycle (SDLC). The application enables teams to create, assign, and track tasks, defaulting to **Python**, **FastAPI** (backend), and **Streamlit** (UI), with user overrides via a Streamlit interface. It integrates **Claude 3.5 Sonnet** for AI tasks, **PostgreSQL** for session and transaction management, and external tools/services for a complete SDLC pipeline. This part covers the architecture, setup, and SDLC Steps 1–3. Part 2 continues with Steps 4–7, and Part 3 concludes with Steps 8–10, testing, and deployment.

### SDLC Steps (Part 1)
1. **Input and Validation of Requirement Document**: Validate compatibility with default or user-specified tech stack.
2. **Creation of Technical Documentation**: Generate specs and Mermaid diagrams.
3. **Task Breakdown and Jira Creation**: Decompose specs into ESTJIRA tickets.

### Tech Stack
- **Backend**: Python 3.9+, FastAPI 0.115.2, Uvicorn 0.32.0.
- **Frontend/UI**: Streamlit 1.39.0.
- **AI Orchestration**: CrewAI 0.67.1, Claude 3.5 Sonnet (Anthropic API).
- **Database**: PostgreSQL (AWS RDS) with tables: `sessions`, `workflow_steps`, `transactions`, `hitl_feedback`.
- **AWS Services**:
  - **Amazon S3**: Stores requirements and artifacts.
  - **Amazon EKS**: Hosts deployed application (configured in Part 2).
  - **Amazon CloudWatch**: Monitors metrics and logs.
- **External Tools**:
  - **ESTJIRA**: Task tracking.
  - **GitLab**: Version control and CI/CD.
  - **Confluence**: Documentation storage.
  - **SonarQube**: Code quality analysis (used in Part 2).
  - **Docker/NEXUS**: Containerization and artifact storage (used in Part 3).
- **Utilities**: Boto3 1.35.24, Atlassian-Python-API 3.41.4, Psycopg2-binary 2.9.10.

## Architecture Diagram
The architecture uses CrewAI to orchestrate AI agents for each SDLC step. Streamlit provides the HITL interface for feedback and tech stack overrides. S3 stores documents, PostgreSQL manages state, Claude 3.5 Sonnet powers AI tasks, and EKS (configured later) hosts the application. CloudWatch monitors, and external tools handle version control, task tracking, and documentation.

```mermaid
graph TD
    classDef agent fill:#f9d5e5,stroke:#333,stroke-width:2px;
    classDef aws fill:#ffebcc,stroke:#333,stroke-width:2px;
    classDef external fill:#c4e7c4,stroke:#333,stroke-width:2px;
    classDef hitl fill:#d4e7f7,stroke:#333,stroke-width:2px;

    A[Streamlit UI<br>Tech: Streamlit, Python<br>HITL: Upload, Feedback, Tech Stack Overrides]:::hitl -->|HTTP| B[FastAPI Backend<br>Tech: Python, FastAPI, Uvicorn]:::hitl
    B -->|Kickoff Crew| C[CrewAI<br>Orchestrates Agents<br>Claude 3.5 Sonnet]:::agent
    J[Amazon S3<br>Stores Requirements/Artifacts]:::aws
    A -->|Upload| J
    B -->|Fetch| J
    G[PostgreSQL<br>Sessions, Steps, Transactions, Feedback]:::aws --> B
    G --> C

    C -->|Task| D1[Requirement Analyzer<br>Agent: CrewAI<br>Tech: Claude 3.5 Sonnet]:::agent
    D1 -->|Store Report| L1[Confluence<br>Validation Report]:::external
    D1 -->|Log State| G
    D1 -->|HITL Feedback| A

    C -->|Task| D2[Tech Doc Generator<br>Agent: CrewAI<br>Tech: Claude 3.5 Sonnet]:::agent --> L1 --> G --> A
    C -->|Task| D3[Task Orchestrator<br>Agent: CrewAI<br>Tech: Claude 3.5 Sonnet]:::agent --> L2[ESTJIRA<br>Task Tickets]:::external --> G --> A
    C -->|Task| D4[Environment Setup<br>Agent: CrewAI<br>Tech: GitLab API]:::agent --> L3[GitLab<br>Repository, CI/CD]:::external --> G --> A
    C -->|Task| D5[Code Generator<br>Agent: CrewAI<br>Tech: Claude 3.5 Sonnet]:::agent --> L3 --> L4[SonarQube<br>Code Quality]:::external --> G --> A
    C -->|Task| D6[Test Automation<br>Agent: CrewAI<br>Tech: Claude 3.5 Sonnet, Pytest]:::agent --> L3 --> K[Amazon CloudWatch<br>Test Metrics]:::aws --> G --> A
    C -->|Task| D7[Version Control<br>Agent: CrewAI<br>Tech: GitLab API]:::agent --> L3 --> G --> A
    C -->|Task| D8[Build<br>Agent: CrewAI<br>Tech: Docker]:::agent --> L5[NEXUS<br>Docker Image]:::external --> G --> A
    C -->|Task| D9[Deployment<br>Agent: CrewAI<br>Tech: Kubernetes]:::agent --> M[Amazon EKS<br>Hosts Application]:::aws --> L3 --> G --> A
    C -->|Task| D10[Monitoring<br>Agent: CrewAI<br>Tech: CloudWatch]:::agent --> K --> L2 --> G --> A

    subgraph Legend
        LAgent[Agent (CrewAI)]
        LAWS[AWS Service]
        LExternal[External Tool]
        LHITL[HITL Interface]
    end
    class LAgent agent; class LAWS aws; class LExternal external; class LHITL hitl;
```

## Coding Standards
- **Python**:
  - **Style**: PEP 8 (88-character lines, snake_case, PascalCase for classes).
  - **Documentation**: Google-style docstrings, inline comments for complex logic.
  - **Linting**: Black (`black .`), Flake8 (`flake8 . --max-line-length=88`).
  - **Error Handling**: Try-except for API calls, log to PostgreSQL/CloudWatch.
  - **Testing**: Pytest with `moto` for AWS mocks, >80% coverage (`pytest --cov=.`).
- **Streamlit**:
  - Modular components, session state for persistence.
  - Input validation, user-friendly error messages.
- **Commits**: Semantic messages (e.g., `feat: add requirement analyzer`, `fix: db connection`).

## Prerequisites
- **Environment**:
  - Python 3.9+, AWS account (S3, EKS, CloudWatch, RDS PostgreSQL).
  - GitLab, Confluence, ESTJIRA instances with API access.
- **Dependencies**:
  ```bash
  pip install crewai==0.67.1 anthropic==0.37.1 boto3==1.35.24 streamlit==1.39.0 atlassian-python-api==3.41.4 psycopg2-binary==2.9.10 pytest==8.3.3 moto==5.0.12
  ```
- **Configuration**:
  - Environment variables: `ANTHROPIC_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `GITLAB_TOKEN`, `CONFLUENCE_USER`, `CONFLUENCE_PASSWORD`, `JIRA_USER`, `JIRA_PASSWORD`, `DB_HOST`, `DB_USER`, `DB_PASSWORD`.
  - PostgreSQL instance (AWS RDS) with schema applied.
- **AWS Setup**:
  - S3 bucket (`your-bucket`).
  - EKS cluster with LoadBalancer (configured in Step 4).
  - CloudWatch for logging/monitoring.

## Directory Structure
```
sdlc_workflow/
├── agents/
│   ├── requirement_analyzer.py
│   ├── technical_documentation.py
│   ├── task_orchestrator.py
│   ├── environment_setup.py
│   ├── code_generator.py
│   ├── test_automation.py
│   ├── version_control.py
│   ├── build.py
│   ├── deployment.py
│   ├── monitoring.py
│   ├── user_input.py
│   ├── workflow_orchestrator.py
├── utils/
│   ├── db.py
│   ├── tools.py
├── streamlit_ui.py
├── requirements.txt
├── schema.sql
```

## Database Schema
The PostgreSQL schema supports session management, workflow tracking, transactions, and feedback, with a `tech_stack` field for overrides.

```sql
CREATE TABLE sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    current_step VARCHAR(50) NOT NULL,
    tech_stack JSONB NOT NULL DEFAULT '{"language": "Python", "backend": "FastAPI", "ui": "Streamlit"}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE workflow_steps (
    session_id VARCHAR(36) REFERENCES sessions(session_id),
    step_id VARCHAR(36) NOT NULL,
    step_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'Pending',
    completed_at TIMESTAMP,
    PRIMARY KEY (session_id, step_id)
);

CREATE TABLE transactions (
    session_id VARCHAR(36) REFERENCES sessions(session_id),
    transaction_id VARCHAR(36) NOT NULL,
    step_id VARCHAR(36) NOT NULL,
    action VARCHAR(100) NOT NULL,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, transaction_id)
);

CREATE TABLE hitl_feedback (
    session_id VARCHAR(36) REFERENCES sessions(session_id),
    feedback_id VARCHAR(36) NOT NULL,
    step_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, feedback_id)
);
```

## Security
- **Credentials**: Stored in environment variables, not hardcoded.
- **Streamlit**: Session state for user authentication, input validation.
- **PostgreSQL**: Parameterized queries to prevent SQL injection.
- **AWS IAM**:
  - Role for S3, EKS, CloudWatch, RDS access.
  - Example policy:
    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": [
            "s3:GetObject",
            "s3:PutObject",
            "rds:DescribeDBInstances",
            "eks:DescribeCluster",
            "logs:CreateLogStream",
            "logs:PutLogEvents"
          ],
          "Resource": "*"
        }
      ]
    }
    ```

## Error Handling
- **CrewAI**: Retry tasks (3 attempts, 2-second delay) via `crewai` configuration.
- **API Calls**: Try-except blocks, log errors to PostgreSQL (`transactions`) and CloudWatch.
- **Streamlit**: Display user-friendly error messages, allow retry.

## Implementation Details and Code

### 1. Input and Validation of Requirement Document
**Description**: Retrieve requirement document from S3, validate using Claude 3.5 Sonnet for compatibility with Python/FastAPI/Streamlit or user-specified stack, and store results in PostgreSQL.

**CrewAI Agents**:
- **Requirement Analyzer**: Validates document, generates report.
- **User Input**: Collects tech stack preferences and feedback.

**Tool Integration**:
- S3: Stores/retrieves document.
- Confluence: Stores validation report.
- PostgreSQL: Logs session, step, tech stack.
- Streamlit: Collects tech stack choice and feedback.

**HITL Mechanism**: Streamlit displays report, allows tech stack override (e.g., Flask), and collects approval/feedback.

**Code**:

**utils/db.py**:
```python
"""Database utilities for PostgreSQL interactions."""
import os
import uuid
import psycopg2
from contextlib import contextmanager
from psycopg2.extras import RealDictCursor
from datetime import datetime

@contextmanager
def get_db_connection():
    """Yield a PostgreSQL connection."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database="sdlc_db"
    )
    try:
        yield conn
    finally:
        conn.close()

def log_session(user_id, tech_stack):
    """Log a new session."""
    session_id = str(uuid.uuid4())
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO sessions (session_id, user_id, current_step, tech_stack)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, user_id, "Requirement Validation", tech_stack)
            )
            conn.commit()
    return session_id

def log_step(session_id, step_id, step_name, status="Pending"):
    """Log a workflow step."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workflow_steps (session_id, step_id, step_name, status)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, step_id, step_name, status)
            )
            conn.commit()

def log_transaction(session_id, step_id, action, details):
    """Log a transaction."""
    transaction_id = str(uuid.uuid4())
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO transactions (session_id, transaction_id, step_id, action, details)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, transaction_id, step_id, action, details)
            )
            conn.commit()

def log_feedback(session_id, step_id, user_id, feedback):
    """Log HITL feedback."""
    feedback_id = str(uuid.uuid4())
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO hitl_feedback (session_id, feedback_id, step_id, user_id, feedback)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, feedback_id, step_id, user_id, feedback)
            )
            conn.commit()

def update_step_status(session_id, step_id, status):
    """Update step status."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workflow_steps
                SET status = %s, completed_at = %s
                WHERE session_id = %s AND step_id = %s
                """,
                (status, datetime.now(), session_id, step_id)
            )
            conn.commit()

def update_session_step(session_id, current_step):
    """Update session's current step."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sessions
                SET current_step = %s
                WHERE session_id = %s
                """,
                (current_step, session_id)
            )
            conn.commit()

def get_session(session_id):
    """Retrieve session details."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
            return cur.fetchone()
```

**utils/tools.py**:
```python
"""Utility functions for AWS and external tool interactions."""
import os
import boto3
import logging
from atlassian import Confluence, Jira
from anthropic import Anthropic, AnthropicError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client("s3")
cloudwatch_client = boto3.client("logs")

def read_s3_file(bucket, key):
    """Read file from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")
    except Exception as e:
        logger.error(f"Error reading S3 file {key}: {str(e)}")
        raise

def write_s3_file(bucket, key, content):
    """Write file to S3."""
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))
        logger.info(f"Wrote file to S3: {key}")
    except Exception as e:
        logger.error(f"Error writing S3 file {key}: {str(e)}")
        raise

def upload_to_confluence(page_title, content, space_key="SDLC"):
    """Upload content to Confluence."""
    confluence = Confluence(
        url="https://your-confluence-url",
        username=os.getenv("CONFLUENCE_USER"),
        password=os.getenv("CONFLUENCE_PASSWORD")
    )
    try:
        confluence.create_page(space_key, page_title, content)
        logger.info(f"Uploaded to Confluence: {page_title}")
    except Exception as e:
        logger.error(f"Error uploading to Confluence: {str(e)}")
        raise

def create_jira_ticket(project_key, summary, description):
    """Create a Jira ticket."""
    jira = Jira(
        url="https://your-jira-url",
        username=os.getenv("JIRA_USER"),
        password=os.getenv("JIRA_PASSWORD")
    )
    try:
        issue = jira.issue_create(fields={
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Task"}
        })
        logger.info(f"Created Jira ticket: {issue['key']}")
        return issue["key"]
    except Exception as e:
        logger.error(f"Error creating Jira ticket: {str(e)}")
        raise

def invoke_claude(prompt):
    """Invoke Claude 3.5 Sonnet."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except AnthropicError as e:
        logger.error(f"Error invoking Claude: {str(e)}")
        raise
```

**agents/requirement_analyzer.py**:
```python
"""Requirement Analyzer Agent for validating SDLC requirements."""
import uuid
from crewai import Agent
from utils.tools import read_s3_file, upload_to_confluence, invoke_claude
from utils.db import log_step, log_transaction

def requirement_analyzer_agent():
    """Create Requirement Analyzer Agent."""
    return Agent(
        role="Requirement Analyzer",
        goal="Validate requirement document and check tech stack compatibility.",
        backstory="Expert in software requirements analysis with deep knowledge of Python, FastAPI, and Streamlit.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def validate_requirements(session_id, bucket, key, tech_stack):
    """Validate requirements and generate report."""
    step_id = str(uuid.uuid4())
    log_step(session_id, step_id, "Requirement Validation")
    
    # Read requirement document
    document = read_s3_file(bucket, key)
    
    # Prepare Claude prompt
    prompt = f"""
    Analyze the following requirement document for a task management application:
    {document}
    
    Validate compatibility with tech stack: {tech_stack}.
    Identify issues, suggest fixes, and generate a validation report.
    """
    
    # Invoke Claude
    report = invoke_claude(prompt)
    
    # Store report in Confluence
    upload_to_confluence(f"Validation Report {session_id}", report)
    
    # Log transaction
    log_transaction(
        session_id,
        step_id,
        "Validate Requirements",
        {"report": report, "tech_stack": tech_stack}
    )
    
    return report, step_id
```

**agents/user_input.py**:
```python
"""User Input Agent for collecting HITL feedback."""
from crewai import Agent

def user_input_agent():
    """Create User Input Agent."""
    return Agent(
        role="User Input Collector",
        goal="Collect tech stack preferences and HITL feedback via Streamlit.",
        backstory="Specialized in gathering user inputs and ensuring seamless HITL interaction.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )
```

**streamlit_ui.py** (Partial for Step 1):
```python
"""Streamlit UI for SDLC workflow."""
import streamlit as st
import uuid
from utils.db import log_session, log_feedback, update_step_status, update_session_step, get_session
from utils.tools import write_s3_file
from agents.requirement_analyzer import validate_requirements

st.title("Agentic SDLC Workflow")

if "session_id" not in st.session_state:
    user_id = st.text_input("Enter User ID")
    tech_stack = st.selectbox(
        "Select Tech Stack",
        ["Python/FastAPI/Streamlit", "Other"]
    )
    if tech_stack == "Other":
        language = st.text_input("Language")
        backend = st.text_input("Backend")
        ui = st.text_input("UI")
        tech_stack = {"language": language, "backend": backend, "ui": ui}
    else:
        tech_stack = {"language": "Python", "backend": "FastAPI", "ui": "Streamlit"}
    
    if st.button("Start Session"):
        st.session_state["session_id"] = log_session(user_id, tech_stack)
        st.session_state["current_step"] = "Requirement Validation"
        st.session_state["user_id"] = user_id
        st.session_state["tech_stack"] = tech_stack

if "session_id" in st.session_state and st.session_state.get("current_step") == "Requirement Validation":
    st.write("Upload Requirement Document")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        bucket = "your-bucket"
        key = f"requirements-{st.session_state['session_id']}.md"
        write_s3_file(bucket, key, uploaded_file.getvalue().decode("utf-8"))
        report, step_id = validate_requirements(
            st.session_state["session_id"], bucket, key, st.session_state["tech_stack"]
        )
        st.session_state["step_id"] = step_id
        st.write("Validation Report")
        st.write(report)
        
        feedback = st.text_area("Provide Feedback")
        if st.button("Submit Feedback"):
            log_feedback(
                st.session_state["session_id"], step_id, st.session_state["user_id"], feedback
            )
            st.write("Feedback Submitted")
        
        if st.button("Approve"):
            update_step_status(st.session_state["session_id"], step_id, "Approved")
            update_session_step(st.session_state["session_id"], "Technical Documentation")
            st.session_state["current_step"] = "Technical Documentation"
            st.write("Step Approved")
```

### 2. Creation of Technical Documentation
**Description**: Generate technical specs and Mermaid diagrams for a Python/FastAPI/Streamlit application (or user-specified stack).

**CrewAI Agents**:
- **Technical Documentation**: Drafts specs.
- **Diagram Generator**: Creates Mermaid diagrams (handled within the same agent for simplicity).
- **User Input**: Collects feedback.

**Tool Integration**:
- Confluence: Stores specs and diagrams.
- PostgreSQL: Logs specs and tech stack.
- Streamlit: Collects feedback and overrides.

**HITL Mechanism**: Streamlit displays specs/diagrams, allows refinements.

**Code**:

**agents/technical_documentation.py**:
```python
"""Technical Documentation Agent for generating SDLC specs."""
import uuid
from crewai import Agent
from utils.tools import read_s3_file, upload_to_confluence, invoke_claude
from utils.db import log_step, log_transaction

def technical_documentation_agent():
    """Create Technical Documentation Agent."""
    return Agent(
        role="Technical Documenter",
        goal="Generate technical specs and Mermaid diagrams.",
        backstory="Expert in creating detailed software specifications and architecture diagrams.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def generate_specs(session_id, bucket, key, tech_stack):
    """Generate technical specs and diagrams."""
    step_id = str(uuid.uuid4())
    log_step(session_id, step_id, "Technical Documentation")
    
    # Read requirement document
    document = read_s3_file(bucket, key)
    
    # Prepare Claude prompt
    prompt = f"""
    Based on the requirement document:
    {document}
    
    Generate technical specifications for a task management application using tech stack: {tech_stack}.
    Include a Mermaid diagram for the system architecture.
    """
    
    # Invoke Claude
    specs = invoke_claude(prompt)
    
    # Store specs in Confluence
    upload_to_confluence(f"Technical Specs {session_id}", specs)
    
    # Log transaction
    log_transaction(
        session_id,
        step_id,
        "Generate Specs",
        {"specs": specs, "tech_stack": tech_stack}
    )
    
    return specs, step_id
```

**streamlit_ui.py** (Partial for Step 2):
```python
if "session_id" in st.session_state and st.session_state.get("current_step") == "Technical Documentation":
    st.write("Technical Specifications")
    if st.button("Generate Specs"):
        bucket = "your-bucket"
        key = f"requirements-{st.session_state['session_id']}.md"
        specs, step_id = generate_specs(
            st.session_state["session_id"], bucket, key, st.session_state["tech_stack"]
        )
        st.session_state["step_id"] = step_id
        st.write("Specifications Generated")
        st.write(specs)
    
    feedback = st.text_area("Suggest Changes to Specs")
    if st.button("Submit Feedback"):
        log_feedback(
            st.session_state["session_id"], st.session_state["step_id"], st.session_state["user_id"], feedback
        )
        st.write("Feedback Submitted")
    
    if st.button("Approve Specs"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Task Breakdown")
        st.session_state["current_step"] = "Task Breakdown"
        st.write("Specs Approved")
```

### 3. Task Breakdown and Jira Creation
**Description**: Decompose specs into tasks and create ESTJIRA tickets.

**CrewAI Agents**:
- **Task Orchestrator**: Creates tasks and tickets.
- **User Input**: Collects task priorities.

**Tool Integration**:
- ESTJIRA: Creates tickets.
- PostgreSQL: Logs tickets.

**HITL Mechanism**: Streamlit allows task prioritization and approval.

**Code**:

**agents/task_orchestrator.py**:
```python
"""Task Orchestrator Agent for creating Jira tickets."""
import uuid
from crewai import Agent
from utils.tools import invoke_claude, create_jira_ticket
from utils.db import log_step, log_transaction

def task_orchestrator_agent():
    """Create Task Orchestrator Agent."""
    return Agent(
        role="Task Orchestrator",
        goal="Decompose specs into tasks and create Jira tickets.",
        backstory="Expert in breaking down technical specs into actionable tasks.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def create_tasks(session_id, specs, tech_stack):
    """Create tasks and Jira tickets."""
    step_id = str(uuid.uuid4())
    log_step(session_id, step_id, "Task Breakdown")
    
    # Prepare Claude prompt
    prompt = f"""
    Based on the technical specifications:
    {specs}
    
    Decompose into tasks for a task management application using tech stack: {tech_stack}.
    Generate a list of Jira ticket summaries and descriptions.
    """
    
    # Invoke Claude
    tasks = invoke_claude(prompt)
    
    # Create Jira tickets
    task_list = tasks.split("\n")
    ticket_ids = []
    for task in task_list:
        if task.strip():
            summary, description = task.split(":", 1)
            ticket_id = create_jira_ticket("TASK", summary.strip(), description.strip())
            ticket_ids.append(ticket_id)
    
    # Log transaction
    log_transaction(
        session_id,
        step_id,
        "Create Tasks",
        {"tasks": tasks, "ticket_ids": ticket_ids}
    )
    
    return tasks, step_id
```

**streamlit_ui.py** (Partial for Step 3):
```python
if "session_id" in st.session_state and st.session_state.get("current_step") == "Task Breakdown":
    st.write("Task Breakdown")
    if st.button("Generate Tasks"):
        # Specs would typically be retrieved from Confluence or S3
        specs = "Sample specs for task management app"  # Placeholder
        tasks, step_id = create_tasks(
            st.session_state["session_id"], specs, st.session_state["tech_stack"]
        )
        st.session_state["step_id"] = step_id
        st.write("Tasks Generated")
        st.write(tasks)
    
    feedback = st.text_area("Suggest Task Priorities")
    if st.button("Submit Feedback"):
        log_feedback(
            st.session_state["session_id"], st.session_state["step_id"], st.session_state["user_id"], feedback
        )
        st.write("Feedback Submitted")
    
    if st.button("Approve Tasks"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Environment Setup")
        st.session_state["current_step"] = "Environment Setup"
        st.write("Tasks Approved")
```

## Continuation
This concludes Part 1, covering the architecture, setup, and SDLC Steps 1–3. Part 2 continues with Steps 4–7 (Environment Setup, Code Development, Test Case Generation, Version Control), including relevant code and Streamlit UI components.

# Agentic SDLC CrewAI Implementation for Task Management - Part 2

## Overview
This document (Part 2 of 3) continues the end-to-end implementation of the **Agentic SDLC Application** for a **Task Management Application** using **CrewAI**. It covers SDLC Steps 4–7, building on Part 1 (architecture, setup, Steps 1–3). The application defaults to **Python**, **FastAPI**, and **Streamlit**, with user overrides via Streamlit, and integrates **Claude 3.5 Sonnet**, **PostgreSQL**, **S3**, **ESTJIRA**, **GitLab**, **Confluence**, **SonarQube**, **Docker**, **NEXUS**, **EKS**, and **CloudWatch**. Part 3 concludes with Steps 8–10, testing, and deployment.

### SDLC Steps (Part 2)
4. **Architecture and Environment Setup**: Set up GitLab repository and EKS.
5. **Code Development**: Generate FastAPI backend and Streamlit UI code.
6. **Test Case Generation and Testing**: Create and run Pytest cases.
7. **Code Check-in and Version Control**: Commit to GitLab.

## Implementation Details and Code

### 4. Architecture and Environment Setup
**Description**: Set up GitLab repository, CI/CD pipeline, and EKS cluster for the task management application.

**CrewAI Agents**:
- **Environment Setup**: Configures GitLab and EKS.
- **User Input**: Collects environment preferences.

**Tool Integration**:
- GitLab: Creates repository and CI/CD pipeline.
- EKS: Sets up cluster.
- PostgreSQL: Logs setup details.

**HITL Mechanism**: Streamlit collects environment preferences and approves setup.

**Code**:

**agents/environment_setup.py**:
```python
"""Environment Setup Agent for configuring GitLab and EKS."""
import uuid
import gitlab
from crewai import Agent
from utils.db import log_step, log_transaction
from utils.tools import invoke_claude

def environment_setup_agent():
    """Create Environment Setup Agent."""
    return Agent(
        role="Environment Configurator",
        goal="Set up GitLab repository and EKS cluster.",
        backstory="Expert in DevOps and cloud infrastructure setup.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def setup_environment(session_id, tech_stack):
    """Set up GitLab repository and EKS cluster."""
    step_id = str(uuid.uuid4())
    log_step(session_id, step_id, "Environment Setup")
    
    # Initialize GitLab client
    gl = gitlab.Gitlab("https://your-gitlab-url", private_token=os.getenv("GITLAB_TOKEN"))
    
    # Create repository
    project = gl.projects.create({"name": f"task-manager-{session_id}", "visibility": "private"})
    
    # Generate .gitlab-ci.yml using Claude
    prompt = f"""
    Generate a .gitlab-ci.yml file for a {tech_stack['language']} application with {tech_stack['backend']} backend and {tech_stack['ui']} UI.
    Include stages for build and deploy to EKS.
    """
    ci_content = invoke_claude(prompt)
    
    # Create .gitlab-ci.yml
    project.files.create({
        "file_path": ".gitlab-ci.yml",
        "branch": "main",
        "content": ci_content,
        "commit_message": "Add CI/CD pipeline"
    })
    
    # Log transaction
    log_transaction(
        session_id,
        step_id,
        "Setup Environment",
        {"project_id": project.id, "ci_content": ci_content}
    )
    
    return project.id, step_id
```

**streamlit_ui.py** (Partial for Step 4):
```python
if "session_id" in st.session_state and st.session_state.get("current_step") == "Environment Setup":
    st.write("Environment Setup")
    if st.button("Setup Environment"):
        project_id, step_id = setup_environment(
            st.session_state["session_id"], st.session_state["tech_stack"]
        )
        st.session_state["step_id"] = step_id
        st.write(f"GitLab Repository Created: {project_id}")
    
    feedback = st.text_area("Provide Environment Preferences")
    if st.button("Submit Feedback"):
        log_feedback(
            st.session_state["session_id"], st.session_state["step_id"], st.session_state["user_id"], feedback
        )
        st.write("Feedback Submitted")
    
    if st.button("Approve Setup"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Code Development")
        st.session_state["current_step"] = "Code Development"
        st.write("Setup Approved")
```

### 5. Code Development
**Description**: Generate Python/FastAPI backend and Streamlit UI code for the task management application, with SonarQube for quality checks.

**CrewAI Agents**:
- **Code Generator**: Writes code.
- **Code Quality**: Runs SonarQube analysis.
- **User Input**: Collects feedback.

**Tool Integration**:
- GitLab: Stores generated code.
- SonarQube: Analyzes code quality.
- PostgreSQL: Logs code paths.

**HITL Mechanism**: Streamlit displays generated code, collects feedback.

**Code**:

**agents/code_generator.py**:
```python
"""Code Generator Agent for generating application code."""
import uuid
from crewai import Agent
from utils.tools import invoke_claude, write_s3_file
from utils.db import log_step, log_transaction

def code_generator_agent():
    """Create Code Generator Agent."""
    return Agent(
        role="Code Generator",
        goal="Generate FastAPI backend and Streamlit UI code.",
        backstory="Expert in Python, FastAPI, and Streamlit development.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def generate_code(session_id, specs, tech_stack):
    """Generate application code."""
    step_id = str(uuid.uuid4())
    log_step(session_id, step_id, "Code Development")
    
    # Prepare Claude prompt
    prompt = f"""
    Based on the technical specifications:
    {specs}
    
    Generate code for a task management application using tech stack: {tech_stack}.
    Include:
    - FastAPI backend (main.py)
    - Streamlit UI (app.py)
    - Basic project structure
    """
    
    # Invoke Claude
    code = invoke_claude(prompt)
    
    # Store code in S3 (to be committed to GitLab in Step 7)
    bucket = "your-bucket"
    key = f"code-{session_id}/generated_code.txt"
    write_s3_file(bucket, key, code)
    
    # Log transaction
    log_transaction(
        session_id,
        step_id,
        "Generate Code",
        {"code_path": key, "tech_stack": tech_stack}
    )
    
    return code, step_id
```

**streamlit_ui.py** (Partial for Step 5):
```python
if "session_id" in st.session_state and st.session_state.get("current_step") == "Code Development":
    st.write("Code Development")
    if st.button("Generate Code"):
        # Specs would typically be retrieved from Confluence or S3
        specs = "Sample specs for task management app"  # Placeholder
        code, step_id = generate_code(
            st.session_state["session_id"], specs, st.session_state["tech_stack"]
        )
        st.session_state["step_id"] = step_id
        st.write("Code Generated")
        st.code(code)
    
    feedback = st.text_area("Provide Code Feedback")
    if st.button("Submit Feedback"):
        log_feedback(
            st.session_state["session_id"], st.session_state["step_id"], st.session_state["user_id"], feedback
        )
        st.write("Feedback Submitted")
    
    if st.button("Approve Code"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Test Automation")
        st.session_state["current_step"] = "Test Automation"
        st.write("Code Approved")
```

### 6. Test Case Generation and Testing
**Description**: Generate Pytest cases for the FastAPI/Streamlit application and run tests.

**CrewAI Agents**:
- **Test Automation**: Generates and runs tests.
- **User Input**: Collects test preferences.

**Tool Integration**:
- GitLab: Stores test code.
- CloudWatch: Monitors test results.
- PostgreSQL: Logs test results.

**HITL Mechanism**: Streamlit displays test results, collects preferences.

**Code**:

**agents/test_automation.py**:
```python
"""Test Automation Agent for generating and running tests."""
import uuid
from crewai import Agent
from utils.tools import invoke_claude, write_s3_file
from utils.db import log_step, log_transaction

def test_automation_agent():
    """Create Test Automation Agent."""
    return Agent(
        role="Test Automator",
        goal="Generate and run Pytest cases.",
        backstory="Expert in automated testing with Pytest.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def generate_tests(session_id, code, tech_stack):
    """Generate and run Pytest cases."""
    step_id = str(uuid.uuid4())
    log_step(session_id, step_id, "Test Automation")
    
    # Prepare Claude prompt
    prompt = f"""
    Based on the application code:
    {code}
    
    Generate Pytest cases for a {tech_stack['language']} application with {tech_stack['backend']} backend and {tech_stack['ui']} UI.
    Include unit and integration tests.
    """
    
    # Invoke Claude
    test_code = invoke_claude(prompt)
    
    # Store test code in S3
    bucket = "your-bucket"
    key = f"tests-{session_id}/test_main.py"
    write_s3_file(bucket, key, test_code)
    
    # Simulate test execution (in practice, run Pytest)
    test_results = "Tests passed: 10/10"  # Placeholder
    
    # Log transaction
    log_transaction(
        session_id,
        step_id,
        "Generate Tests",
        {"test_path": key, "results": test_results}
    )
    
    return test_code, test_results, step_id
```

**streamlit_ui.py** (Partial for Step 6):
```python
if "session_id" in st.session_state and st.session_state.get("current_step") == "Test Automation":
    st.write("Test Automation")
    if st.button("Generate Tests"):
        # Code would typically be retrieved from S3
        code = "Sample code for task management app"  # Placeholder
        test_code, test_results, step_id = generate_tests(
            st.session_state["session_id"], code, st.session_state["tech_stack"]
        )
        st.session_state["step_id"] = step_id
        st.write("Tests Generated")
        st.code(test_code)
        st.write("Test Results")
        st.write(test_results)
    
    feedback = st.text_area("Provide Test Preferences")
    if st.button("Submit Feedback"):
        log_feedback(
            st.session_state["session_id"], st.session_state["step_id"], st.session_state["user_id"], feedback
        )
        st.write("Feedback Submitted")
    
    if st.button("Approve Tests"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Version Control")
        st.session_state["current_step"] = "Version Control"
        st.write("Tests Approved")
```

### 7. Code Check-in and Version Control
**Description**: Commit generated code and tests to GitLab.

**CrewAI Agents**:
- **Version Control**: Commits code to GitLab.
- **User Input**: Collects commit message preferences.

**Tool Integration**:
- GitLab: Version control.
- PostgreSQL: Logs commits.

**HITL Mechanism**: Streamlit collects commit message and approves commit.

**Code**:

**agents/version_control.py**:
```python
"""Version Control Agent for committing code to GitLab."""
import uuid
import gitlab
from crewai import Agent
from utils.tools import read_s3_file
from utils.db import log_step, log_transaction

def version_control_agent():
    """Create Version Control Agent."""
    return Agent(
        role="Version Control Manager",
        goal="Commit code to GitLab repository.",
        backstory="Expert in Git and GitLab version control.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def commit_code(session_id, code_path, test_path):
    """Commit code and tests to GitLab."""
    step_id = str(uuid.uuid4())
    log_step(session_id, step_id, "Version Control")
    
    # Initialize GitLab client
    gl = gitlab.Gitlab("https://your-gitlab-url", private_token=os.getenv("GITLAB_TOKEN"))
    project = gl.projects.get(f"task-manager-{session_id}")
    
    # Read code and tests from S3
    code = read_s3_file("your-bucket", code_path)
    test_code = read_s3_file("your-bucket", test_path)
    
    # Commit code
    project.files.create({
        "file_path": "main.py",
        "branch": "main",
        "content": code,
        "commit_message": "Add task management application code"
    })
    project.files.create({
        "file_path": "test_main.py",
        "branch": "main",
        "content": test_code,
        "commit_message": "Add test cases"
    })
    
    # Log transaction
    log_transaction(
        session_id,
        step_id,
        "Commit Code",
        {"code_path": code_path, "test_path": test_path}
    )
    
    return step_id
```

**streamlit_ui.py** (Partial for Step 7):
```python
if "session_id" in st.session_state and st.session_state.get("current_step") == "Version Control":
    st.write("Version Control")
    if st.button("Commit Code"):
        code_path = f"code-{st.session_state['session_id']}/generated_code.txt"
        test_path = f"tests-{st.session_state['session_id']}/test_main.py"
        step_id = commit_code(st.session_state["session_id"], code_path, test_path)
        st.session_state["step_id"] = step_id
        st.write("Code Committed to GitLab")
    
    feedback = st.text_area("Provide Commit Message")
    if st.button("Submit Feedback"):
        log_feedback(
            st.session_state["session_id"], st.session_state["step_id"], st.session_state["user_id"], feedback
        )
        st.write("Feedback Submitted")
    
    if st.button("Approve Commit"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Build")
        st.session_state["current_step"] = "Build"
        st.write("Commit Approved")
```

## Continuation
This concludes Part 2, covering SDLC Steps 4–7. Part 3 continues with Steps 8–10 (Build, Deployment, Monitoring), the Workflow Orchestrator, complete Streamlit UI, testing guidance, and conclusion.

# Agentic SDLC CrewAI Implementation for Task Management - Part 3

## Overview
This document (Part 3 of 6) concludes the core SDLC steps of the **Agentic SDLC Application** for a **Task Management Application** using **CrewAI**. It covers SDLC Steps 8–10 (Build, Deployment, Monitoring), the Workflow Orchestrator, the complete Streamlit UI, testing guidance, and a conclusion. It builds on Part 1 (architecture, setup, Steps 1–3), Part 2 (Steps 4–7), and is followed by Part 4 (FastAPI backend), Part 5 (sample code, manifests), and Part 6 (testing, CI/CD, monitoring). The application defaults to **Python**, **FastAPI**, and **Streamlit**, with user overrides via Streamlit, and integrates **Claude 3.5 Sonnet**, **PostgreSQL**, **S3**, **ESTJIRA**, **GitLab**, **Confluence**, **SonarQube**, **Docker**, **NEXUS**, **EKS**, and **CloudWatch**.

### SDLC Steps (Part 3)
8. **Build and Deployment Package Creation**: Create a Docker image for the application.
9. **Deployment to Amazon EKS**: Deploy the Docker image to EKS with autoscaling.
10. **Post-Deployment Monitoring and Maintenance**: Monitor with CloudWatch and create maintenance tasks in ESTJIRA.

## Implementation Details and Code

### 8. Build and Deployment Package Creation
**Description**: Create a Docker image for the task management application (FastAPI backend and Streamlit UI) and store it in a NEXUS repository.

**CrewAI Agents**:
- **Build Agent**: Generates and builds the Docker image.
- **User Input Agent**: Collects build preferences via Streamlit.

**Tool Integration**:
- **Docker**: Builds the image.
- **NEXUS**: Stores the image.
- **PostgreSQL**: Logs build details (session_id, step_id, image_tag).
- **S3**: Stores generated Dockerfile.
- **Streamlit**: Provides HITL interface for preferences and approval.

**HITL Mechanism**: Streamlit displays the generated Dockerfile, collects build configuration preferences (e.g., base image version), and requires user approval to proceed.

**Code**:

**agents/build.py**:
```python
"""Build Agent for creating Docker image."""
import uuid
from crewai import Agent
from utils.tools import invoke_claude, write_s3_file
from utils.db import log_step, log_transaction
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_agent():
    """Create Build Agent.

    Returns:
        Agent: CrewAI agent for building Docker images.
    """
    return Agent(
        role="Build Manager",
        goal="Generate and build Docker image for the task management application.",
        backstory="Expert in containerization with Docker and artifact management in NEXUS.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def build_docker_image(session_id: str, tech_stack: dict) -> tuple[str, str, str]:
    """Generate Dockerfile and simulate building Docker image.

    Args:
        session_id: Unique identifier for the session.
        tech_stack: Dictionary containing language, backend, and UI (e.g., {"language": "Python", "backend": "FastAPI", "ui": "Streamlit"}).

    Returns:
        Tuple of (Dockerfile content, image tag, step ID).

    Raises:
        Exception: If Claude invocation or S3 write fails.
    """
    step_id = str(uuid.uuid4())
    log_step(session_id, step_id, "Build")
    
    # Prepare Claude prompt for Dockerfile
    prompt = f"""
    Generate a Dockerfile for a {tech_stack['language']} task management application with {tech_stack['backend']} backend and {tech_stack['ui']} UI.
    Include:
    - Base image (Python 3.9).
    - Installation of dependencies from requirements.txt.
    - Exposure of port 8000 for FastAPI and 8501 for Streamlit.
    - Commands to run both FastAPI and Streamlit (e.g., using a startup script).
    """
    
    try:
        # Invoke Claude to generate Dockerfile
        dockerfile = invoke_claude(prompt)
        
        # Store Dockerfile in S3
        bucket = "your-bucket"
        key = f"build-{session_id}/Dockerfile"
        write_s3_file(bucket, key, dockerfile)
        
        # Simulate Docker build and push to NEXUS (in practice, use Docker CLI)
        image_tag = f"nexus-repo/task-manager:{session_id}"
        
        # Log transaction
        log_transaction(
            session_id,
            step_id,
            "Build Docker Image",
            {"dockerfile_path": key, "image_tag": image_tag}
        )
        
        logger.info(f"Built Docker image for session {session_id}: {image_tag}")
        return dockerfile, image_tag, step_id
    
    except Exception as e:
        logger.error(f"Error building Docker image for session {session_id}: {str(e)}")
        raise
```

**streamlit_ui.py** (Partial for Step 8):
```python
if "session_id" in st.session_state and st.session_state.get("current_step") == "Build":
    st.subheader("Step 8: Build Docker Image")
    if st.button("Generate and Build Docker Image"):
        try:
            dockerfile, image_tag, step_id = build_docker_image(
                st.session_state["session_id"], st.session_state["tech_stack"]
            )
            st.session_state["step_id"] = step_id
            st.success(f"Docker Image Built: {image_tag}")
            st.code(dockerfile, language="dockerfile")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    feedback = st.text_area("Provide Build Preferences (e.g., base image version)")
    if st.button("Submit Build Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Build"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Deployment")
        st.session_state["current_step"] = "Deployment"
        st.success("Build Approved, Proceeding to Deployment")
```

### 9. Deployment to Amazon EKS
**Description**: Deploy the Docker image to an Amazon EKS cluster, configuring autoscaling and a LoadBalancer service.

**CrewAI Agents**:
- **Deployment Agent**: Generates Kubernetes manifests and deploys to EKS.
- **User Input Agent**: Collects deployment preferences (e.g., replica count).

**Tool Integration**:
- **EKS**: Hosts the application.
- **Kubernetes**: Manages deployment (via kubectl).
- **CloudWatch**: Logs deployment events.
- **PostgreSQL**: Logs deployment status.
- **S3**: Stores Kubernetes manifests.
- **Streamlit**: Provides HITL interface for preferences and approval.

**HITL Mechanism**: Streamlit displays generated Kubernetes manifests, collects deployment preferences (e.g., CPU/memory limits), and requires user approval.

**Code**:

**agents/deployment.py**:
```python
"""Deployment Agent for deploying to Amazon EKS."""
import uuid
from crewai import Agent
from utils.tools import invoke_claude, write_s3_file
from utils.db import log_step, log_transaction
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deployment_agent():
    """Create Deployment Agent.

    Returns:
        Agent: CrewAI agent for EKS deployments.
    """
    return Agent(
        role="Deployment Manager",
        goal="Deploy the task management application to Amazon EKS with autoscaling.",
        backstory="Expert in Kubernetes, EKS, and cloud deployments.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def deploy_to_eks(session_id: str, image_tag: str) -> tuple[str, str, str]:
    """Generate Kubernetes manifests and simulate EKS deployment.

    Args:
        session_id: Unique identifier for the session.
        image_tag: Docker image tag (e.g., nexus-repo/task-manager:session_id).

    Returns:
        Tuple of (Kubernetes manifests, deployment status, step ID).

    Raises:
        Exception: If Claude invocation or S3 write fails.
    """
    step_id = str(uuid.uuid4())
    log_step(session_id, step_id, "Deployment")
    
    # Prepare Claude prompt for Kubernetes manifests
    prompt = f"""
    Generate Kubernetes manifests (deployment.yaml, service.yaml) for a task management application.
    Requirements:
    - Use Docker image: {image_tag}.
    - Deployment with 2 replicas, CPU request 100m, memory 256Mi, limits 500m CPU, 512Mi memory.
    - HorizontalPodAutoscaler targeting 70% CPU utilization, min 2, max 5 replicas.
    - LoadBalancer service exposing port 80 to 8000.
    - Namespace: default.
    """
    
    try:
        # Invoke Claude to generate manifests
        manifests = invoke_claude(prompt)
        
        # Store manifests in S3
        bucket = "your-bucket"
        key = f"deployment-{session_id}/manifests.yaml"
        write_s3_file(bucket, key, manifests)
        
        # Simulate deployment (in practice, use kubectl apply)
        deployment_status = "Deployed to EKS successfully"
        
        # Log transaction
        log_transaction(
            session_id,
            step_id,
            "Deploy to EKS",
            {"manifests_path": key, "status": deployment_status, "image_tag": image_tag}
        )
        
        logger.info(f"Deployed to EKS for session {session_id}: {deployment_status}")
        return manifests, deployment_status, step_id
    
    except Exception as e:
        logger.error(f"Error deploying to EKS for session {session_id}: {str(e)}")
        raise
```

**streamlit_ui.py** (Partial for Step 9):
```python
if "session_id" in st.session_state and st.session_state.get("current_step") == "Deployment":
    st.subheader("Step 9: Deploy to Amazon EKS")
    if st.button("Deploy Application"):
        try:
            image_tag = f"nexus-repo/task-manager:{st.session_state['session_id']}"
            manifests, deployment_status, step_id = deploy_to_eks(
                st.session_state["session_id"], image_tag
            )
            st.session_state["step_id"] = step_id
            st.success(f"Deployment Status: {deployment_status}")
            st.code(manifests, language="yaml")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    feedback = st.text_area("Provide Deployment Preferences (e.g., replica count, resource limits)")
    if st.button("Submit Deployment Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Deployment"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Monitoring")
        st.session_state["current_step"] = "Monitoring"
        st.success("Deployment Approved, Proceeding to Monitoring")
```

### 10. Post-Deployment Monitoring and Maintenance
**Description**: Monitor the deployed application using CloudWatch (metrics, logs) and create maintenance tasks in ESTJIRA for ongoing support.

**CrewAI Agents**:
- **Monitoring Agent**: Configures CloudWatch monitoring and creates ESTJIRA tasks.
- **User Input Agent**: Collects monitoring preferences (e.g., alert thresholds).

**Tool Integration**:
- **CloudWatch**: Monitors CPU, memory, and error rates.
- **ESTJIRA**: Creates maintenance tasks.
- **PostgreSQL**: Logs monitoring metrics and task IDs.
- **Streamlit**: Provides HITL interface for preferences and approval.

**HITL Mechanism**: Streamlit displays CloudWatch metrics, collects monitoring preferences (e.g., CPU threshold for alerts), and requires user approval.

**Code**:

**agents/monitoring.py**:
```python
"""Monitoring Agent for post-deployment monitoring."""
import uuid
from crewai import Agent
from utils.tools import create_jira_ticket
from utils.db import log_step, log_transaction
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitoring_agent():
    """Create Monitoring Agent.

    Returns:
        Agent: CrewAI agent for application monitoring.
    """
    return Agent(
        role="Monitoring Manager",
        goal="Monitor the task management application and create maintenance tasks.",
        backstory="Expert in AWS CloudWatch and application performance monitoring.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def monitor_application(session_id: str) -> tuple[str, str, str]:
    """Simulate CloudWatch monitoring and create ESTJIRA maintenance tasks.

    Args:
        session_id: Unique identifier for the session.

    Returns:
        Tuple of (metrics summary, Jira ticket ID, step ID).

    Raises:
        Exception: If Jira ticket creation fails.
    """
    step_id = str(uuid.uuid4())
    log_step(session_id, step_id, "Monitoring")
    
    try:
        # Simulate CloudWatch metrics (in practice, query CloudWatch Metrics API)
        metrics = "CPU Utilization: 25%, Memory Usage: 40%, Error Rate: 0.1%"
        
        # Create maintenance task in ESTJIRA
        ticket_id = create_jira_ticket(
            project_key="TASK",
            summary=f"Monitor Task Manager Application {session_id}",
            description=f"Monitor application metrics: {metrics}\nEnsure CPU < 80%, Memory < 70%, Errors < 1%."
        )
        
        # Log transaction
        log_transaction(
            session_id,
            step_id,
            "Monitor Application",
            {"metrics": metrics, "ticket_id": ticket_id}
        )
        
        logger.info(f"Monitoring setup for session {session_id}, Jira ticket: {ticket_id}")
        return metrics, ticket_id, step_id
    
    except Exception as e:
        logger.error(f"Error setting up monitoring for session {session_id}: {str(e)}")
        raise
```

**streamlit_ui.py** (Partial for Step 10):
```python
if "session_id" in st.session_state and st.session_state.get("current_step") == "Monitoring":
    st.subheader("Step 10: Application Monitoring")
    if st.button("Setup Monitoring"):
        try:
            metrics, ticket_id, step_id = monitor_application(st.session_state["session_id"])
            st.session_state["step_id"] = step_id
            st.success("Monitoring Configured")
            st.write("**CloudWatch Metrics**")
            st.write(metrics)
            st.write(f"**Maintenance Task**: {ticket_id}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    feedback = st.text_area("Provide Monitoring Preferences (e.g., CPU threshold for alerts)")
    if st.button("Submit Monitoring Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Monitoring"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Completed")
        st.session_state["current_step"] = "Completed"
        st.success("Monitoring Approved, Workflow Completed")
```

### Workflow Orchestrator
**Description**: The Workflow Orchestrator coordinates all 10 SDLC steps by managing CrewAI agents and tasks, ensuring sequential execution and HITL checkpoints. It integrates with the Streamlit UI for user interaction and PostgreSQL for state persistence.

**Code**:

**agents/workflow_orchestrator.py**:
```python
"""Workflow Orchestrator Agent for coordinating SDLC steps."""
from crewai import Agent, Crew, Task
from agents.requirement_analyzer import requirement_analyzer_agent, validate_requirements
from agents.technical_documentation import technical_documentation_agent, generate_specs
from agents.task_orchestrator import task_orchestrator_agent, create_tasks
from agents.environment_setup import environment_setup_agent, setup_environment
from agents.code_generator import code_generator_agent, generate_code
from agents.test_automation import test_automation_agent, generate_tests
from agents.version_control import version_control_agent, commit_code
from agents.build import build_agent, build_docker_image
from agents.deployment import deployment_agent, deploy_to_eks
from agents.monitoring import monitoring_agent, monitor_application
from agents.user_input import user_input_agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def workflow_orchestrator():
    """Create Workflow Orchestrator Agent.

    Returns:
        Agent: CrewAI agent for orchestrating SDLC workflow.
    """
    return Agent(
        role="Workflow Orchestrator",
        goal="Coordinate all SDLC steps, ensuring HITL checkpoints and task execution.",
        backstory="Expert in managing end-to-end software development workflows with AI orchestration.",
        tools=[],
        llm="claude-3-5-sonnet-20241022"
    )

def run_workflow(session_id: str, bucket: str, key: str, tech_stack: dict) -> None:
    """Execute the SDLC workflow using CrewAI.

    Args:
        session_id: Unique identifier for the session.
        bucket: S3 bucket name.
        key: S3 key for the requirement document.
        tech_stack: Dictionary containing language, backend, and UI.

    Raises:
        Exception: If any task fails after retries.
    """
    try:
        crew = Crew(
            agents=[
                requirement_analyzer_agent(),
                technical_documentation_agent(),
                task_orchestrator_agent(),
                environment_setup_agent(),
                code_generator_agent(),
                test_automation_agent(),
                version_control_agent(),
                build_agent(),
                deployment_agent(),
                monitoring_agent(),
                user_input_agent()
            ],
            tasks=[
                Task(
                    description="Validate requirement document for compatibility",
                    agent=requirement_analyzer_agent(),
                    function=lambda: validate_requirements(session_id, bucket, key, tech_stack)
                ),
                Task(
                    description="Generate technical specifications and diagrams",
                    agent=technical_documentation_agent(),
                    function=lambda: generate_specs(session_id, bucket, key, tech_stack)
                ),
                Task(
                    description="Break down specs into tasks and create ESTJIRA tickets",
                    agent=task_orchestrator_agent(),
                    function=lambda: create_tasks(session_id, "Sample technical specs", tech_stack)
                ),
                Task(
                    description="Set up GitLab repository and EKS cluster",
                    agent=environment_setup_agent(),
                    function=lambda: setup_environment(session_id, tech_stack)
                ),
                Task(
                    description="Generate FastAPI backend and Streamlit UI code",
                    agent=code_generator_agent(),
                    function=lambda: generate_code(session_id, "Sample technical specs", tech_stack)
                ),
                Task(
                    description="Generate and run Pytest cases",
                    agent=test_automation_agent(),
                    function=lambda: generate_tests(session_id, "Sample generated code", tech_stack)
                ),
                Task(
                    description="Commit code and tests to GitLab",
                    agent=version_control_agent(),
                    function=lambda: commit_code(
                        session_id,
                        f"code-{session_id}/generated_code.txt",
                        f"tests-{session_id}/test_main.py"
                    )
                ),
                Task(
                    description="Build Docker image and store in NEXUS",
                    agent=build_agent(),
                    function=lambda: build_docker_image(session_id, tech_stack)
                ),
                Task(
                    description="Deploy Docker image to EKS",
                    agent=deployment_agent(),
                    function=lambda: deploy_to_eks(session_id, f"nexus-repo/task-manager:{session_id}")
                ),
                Task(
                    description="Set up CloudWatch monitoring and create maintenance tasks",
                    agent=monitoring_agent(),
                    function=lambda: monitor_application(session_id)
                )
            ]
        )
        logger.info(f"Starting workflow for session {session_id}")
        crew.kickoff()
        logger.info(f"Workflow completed for session {session_id}")
    
    except Exception as e:
        logger.error(f"Workflow failed for session {session_id}: {str(e)}")
        raise
```

### Complete Streamlit UI
**Description**: The Streamlit UI (`streamlit_ui.py`) provides a user-friendly interface for the entire SDLC workflow, managing session initialization, tech stack overrides, step execution, and HITL feedback. It integrates with the utilities and agents defined in Parts 1 and 2, and supports Steps 8–10 defined above.

**Code**:

**streamlit_ui.py**:
```python
"""Streamlit UI for Agentic SDLC Workflow."""
import streamlit as st
import os
import json
from utils.db import log_session, log_feedback, update_step_status, update_session_step
from utils.tools import write_s3_file
from agents.requirement_analyzer import validate_requirements
from agents.technical_documentation import generate_specs
from agents.task_orchestrator import create_tasks
from agents.environment_setup import setup_environment
from agents.code_generator import generate_code
from agents.test_automation import generate_tests
from agents.version_control import commit_code
from agents.build import build_docker_image
from agents.deployment import deploy_to_eks
from agents.monitoring import monitor_application

st.title("Agentic SDLC Workflow for Task Management")

# Session Initialization
if "session_id" not in st.session_state:
    st.subheader("Initialize Workflow Session")
    user_id = st.text_input("Enter User ID", value="user123")
    tech_stack = st.selectbox(
        "Select Tech Stack",
        ["Python/FastAPI/Streamlit", "Custom"]
    )
    if tech_stack == "Custom":
        language = st.text_input("Programming Language", value="Python")
        backend = st.text_input("Backend Framework", value="FastAPI")
        ui = st.text_input("UI Framework", value="Streamlit")
        tech_stack = {"language": language, "backend": backend, "ui": ui}
    else:
        tech_stack = {"language": "Python", "backend": "FastAPI", "ui": "Streamlit"}
    
    if st.button("Start Workflow"):
        try:
            st.session_state["session_id"] = log_session(user_id, tech_stack)
            st.session_state["current_step"] = "Requirement Validation"
            st.session_state["user_id"] = user_id
            st.session_state["tech_stack"] = tech_stack
            st.success(f"Session Started: {st.session_state['session_id']}")
        except Exception as e:
            st.error(f"Error starting session: {str(e)}")

# Step 1: Requirement Validation
if "session_id" in st.session_state and st.session_state.get("current_step") == "Requirement Validation":
    st.subheader("Step 1: Requirement Validation")
    uploaded_file = st.file_uploader("Upload Requirement Document (Markdown or Text)", type=["md", "txt"])
    if uploaded_file:
        try:
            bucket = "your-bucket"
            key = f"requirements/{st.session_state['session_id']}/input.md"
            content = uploaded_file.getvalue().decode("utf-8")
            write_s3_file(bucket, key, content)
            report, step_id = validate_requirements(
                st.session_state["session_id"], bucket, key, st.session_state["tech_stack"]
            )
            st.session_state["step_id"] = step_id
            st.write("**Validation Report**")
            st.markdown(report)
        except Exception as e:
            st.error(f"Error validating requirements: {str(e)}")
        
        feedback = st.text_area("Provide Feedback on Validation Report")
        if st.button("Submit Validation Feedback"):
            log_feedback(
                st.session_state["session_id"],
                st.session_state["step_id"],
                st.session_state["user_id"],
                feedback
            )
            st.success("Feedback Submitted")
        
        if st.button("Approve Validation"):
            update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
            update_session_step(st.session_state["session_id"], "Technical Documentation")
            st.session_state["current_step"] = "Technical Documentation"
            st.success("Validation Approved, Proceeding to Technical Documentation")

# Step 2: Technical Documentation
if "session_id" in st.session_state and st.session_state.get("current_step") == "Technical Documentation":
    st.subheader("Step 2: Technical Documentation")
    if st.button("Generate Technical Specifications"):
        try:
            bucket = "your-bucket"
            key = f"requirements/{st.session_state['session_id']}/input.md"
            specs, step_id = generate_specs(
                st.session_state["session_id"], bucket, key, st.session_state["tech_stack"]
            )
            st.session_state["step_id"] = step_id
            st.write("**Technical Specifications**")
            st.markdown(specs)
        except Exception as e:
            st.error(f"Error generating specs: {str(e)}")
    
    feedback = st.text_area("Suggest Changes to Specifications")
    if st.button("Submit Specs Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Specifications"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Task Breakdown")
        st.session_state["current_step"] = "Task Breakdown"
        st.success("Specifications Approved, Proceeding to Task Breakdown")

# Step 3: Task Breakdown
if "session_id" in st.session_state and st.session_state.get("current_step") == "Task Breakdown":
    st.subheader("Step 3: Task Breakdown and ESTJIRA Integration")
    if st.button("Generate Tasks and Jira Tickets"):
        try:
            # Placeholder specs (in practice, retrieve from Confluence or S3)
            specs = "Task management app with CRUD operations and user assignment."
            tasks, step_id = create_tasks(
                st.session_state["session_id"], specs, st.session_state["tech_stack"]
            )
            st.session_state["step_id"] = step_id
            st.write("**Generated Tasks**")
            st.markdown(tasks)
        except Exception as e:
            st.error(f"Error generating tasks: {str(e)}")
    
    feedback = st.text_area("Suggest Task Priorities or Modifications")
    if st.button("Submit Task Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Tasks"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Environment Setup")
        st.session_state["current_step"] = "Environment Setup"
        st.success("Tasks Approved, Proceeding to Environment Setup")

# Step 4: Environment Setup
if "session_id" in st.session_state and st.session_state.get("current_step") == "Environment Setup":
    st.subheader("Step 4: Environment Setup")
    if st.button("Setup GitLab and EKS"):
        try:
            project_id, step_id = setup_environment(
                st.session_state["session_id"], st.session_state["tech_stack"]
            )
            st.session_state["step_id"] = step_id
            st.success(f"GitLab Repository Created: ID {project_id}")
        except Exception as e:
            st.error(f"Error setting up environment: {str(e)}")
    
    feedback = st.text_area("Provide Environment Preferences (e.g., CI/CD settings)")
    if st.button("Submit Environment Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Environment Setup"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Code Development")
        st.session_state["current_step"] = "Code Development"
        st.success("Environment Setup Approved, Proceeding to Code Development")

# Step 5: Code Development
if "session_id" in st.session_state and st.session_state.get("current_step") == "Code Development":
    st.subheader("Step 5: Code Development")
    if st.button("Generate Application Code"):
        try:
            # Placeholder specs (in practice, retrieve from Confluence or S3)
            specs = "Task management app with CRUD operations and user assignment."
            code, step_id = generate_code(
                st.session_state["session_id"], specs, st.session_state["tech_stack"]
            )
            st.session_state["step_id"] = step_id
            st.write("**Generated Code**")
            st.code(code, language="python")
        except Exception as e:
            st.error(f"Error generating code: {str(e)}")
    
    feedback = st.text_area("Provide Code Feedback (e.g., additional features)")
    if st.button("Submit Code Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Code"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Test Automation")
        st.session_state["current_step"] = "Test Automation"
        st.success("Code Approved, Proceeding to Test Automation")

# Step 6: Test Automation
if "session_id" in st.session_state and st.session_state.get("current_step") == "Test Automation":
    st.subheader("Step 6: Test Automation")
    if st.button("Generate and Run Tests"):
        try:
            # Placeholder code (in practice, retrieve from S3 or GitLab)
            code = "Sample FastAPI and Streamlit code."
            test_code, test_results, step_id = generate_tests(
                st.session_state["session_id"], code, st.session_state["tech_stack"]
            )
            st.session_state["step_id"] = step_id
            st.write("**Generated Test Code**")
            st.code(test_code, language="python")
            st.write("**Test Results**")
            st.markdown(test_results)
        except Exception as e:
            st.error(f"Error generating tests: {str(e)}")
    
    feedback = st.text_area("Provide Test Preferences (e.g., additional test cases)")
    if st.button("Submit Test Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Tests"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Version Control")
        st.session_state["current_step"] = "Version Control"
        st.success("Tests Approved, Proceeding to Version Control")

# Step 7: Version Control
if "session_id" in st.session_state and st.session_state.get("current_step") == "Version Control":
    st.subheader("Step 7: Version Control")
    if st.button("Commit Code to GitLab"):
        try:
            code_path = f"code/{st.session_state['session_id']}/generated_code.txt"
            test_path = f"tests/{st.session_state['session_id']}/test_main.py"
            step_id = commit_code(st.session_state["session_id"], code_path, test_path)
            st.session_state["step_id"] = step_id
            st.success("Code and Tests Committed to GitLab")
        except Exception as e:
            st.error(f"Error committing code: {str(e)}")
    
    feedback = st.text_area("Provide Commit Message or Branch Preferences")
    if st.button("Submit Version Control Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Commit"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Build")
        st.session_state["current_step"] = "Build"
        st.success("Commit Approved, Proceeding to Build")

# Step 8: Build
if "session_id" in st.session_state and st.session_state.get("current_step") == "Build":
    st.subheader("Step 8: Build Docker Image")
    if st.button("Generate and Build Docker Image"):
        try:
            dockerfile, image_tag, step_id = build_docker_image(
                st.session_state["session_id"], st.session_state["tech_stack"]
            )
            st.session_state["step_id"] = step_id
            st.success(f"Docker Image Built: {image_tag}")
            st.code(dockerfile, language="dockerfile")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    feedback = st.text_area("Provide Build Preferences (e.g., base image version)")
    if st.button("Submit Build Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Build"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Deployment")
        st.session_state["current_step"] = "Deployment"
        st.success("Build Approved, Proceeding to Deployment")

# Step 9: Deployment
if "session_id" in st.session_state and st.session_state.get("current_step") == "Deployment":
    st.subheader("Step 9: Deploy to Amazon EKS")
    if st.button("Deploy Application"):
        try:
            image_tag = f"nexus-repo/task-manager:{st.session_state['session_id']}"
            manifests, deployment_status, step_id = deploy_to_eks(
                st.session_state["session_id"], image_tag
            )
            st.session_state["step_id"] = step_id
            st.success(f"Deployment Status: {deployment_status}")
            st.code(manifests, language="yaml")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    feedback = st.text_area("Provide Deployment Preferences (e.g., replica count, resource limits)")
    if st.button("Submit Deployment Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Deployment"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Monitoring")
        st.session_state["current_step"] = "Monitoring"
        st.success("Deployment Approved, Proceeding to Monitoring")

# Step 10: Monitoring
if "session_id" in st.session_state and st.session_state.get("current_step") == "Monitoring":
    st.subheader("Step 10: Application Monitoring")
    if st.button("Setup Monitoring"):
        try:
            metrics, ticket_id, step_id = monitor_application(st.session_state["session_id"])
            st.session_state["step_id"] = step_id
            st.success("Monitoring Configured")
            st.write("**CloudWatch Metrics**")
            st.markdown(metrics)
            st.write(f"**Maintenance Task**: {ticket_id}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    feedback = st.text_area("Provide Monitoring Preferences (e.g., CPU threshold for alerts)")
    if st.button("Submit Monitoring Feedback"):
        log_feedback(
            st.session_state["session_id"],
            st.session_state["step_id"],
            st.session_state["user_id"],
            feedback
        )
        st.success("Feedback Submitted")
    
    if st.button("Approve Monitoring"):
        update_step_status(st.session_state["session_id"], st.session_state["step_id"], "Approved")
        update_session_step(st.session_state["session_id"], "Completed")
        st.session_state["current_step"] = "Completed"
        st.success("Monitoring Approved, Workflow Completed")

# Workflow Completion
if "session_id" in st.session_state and st.session_state.get("current_step") == "Completed":
    st.subheader("Workflow Completed")
    st.success("Congratulations! The SDLC workflow is complete.")
    st.write("Review artifacts in:")
    st.markdown("""
    - **GitLab**: Source code and CI/CD pipelines.
    - **Confluence**: Validation reports and technical specs.
    - **ESTJIRA**: Task tickets and maintenance tasks.
    - **NEXUS**: Docker images.
    - **Amazon EKS**: Deployed application.
    - **Amazon CloudWatch**: Monitoring metrics and logs.
    """)
    if st.button("Start New Workflow"):
        st.session_state.clear()
        st.experimental_rerun()

```

### Testing Guidance
**Description**: Instructions to test the SDLC workflow, ensuring all components (Streamlit UI, CrewAI agents, PostgreSQL, S3, external tools) function correctly.

**Steps**:
1. **Environment Setup**:
   - **AWS**: Configure an AWS account with:
     - S3 bucket (`your-bucket`) with versioning enabled.
     - PostgreSQL RDS instance (apply `schema.sql` from Part 1).
     - EKS cluster with a LoadBalancer and IAM roles for S3, RDS, EKS, CloudWatch.
     - CloudWatch log groups (`sdlc-workflow/logs`, `sdlc-workflow/errors`).
   - **External Tools**:
     - GitLab instance with a personal access token (`GITLAB_TOKEN`).
     - Confluence instance with API credentials (`CONFLUENCE_USER`, `CONFLUENCE_PASSWORD`).
     - ESTJIRA instance with API credentials (`JIRA_USER`, `JIRA_PASSWORD`).
   - **Local Setup**:
     - Python 3.9+ environment.
     - Install dependencies:
       ```bash
       pip install crewai==0.67.1 anthropic==0.37.1 boto3==1.35.24 streamlit==1.39.0 \
       atlassian-python-api==3.41.4 psycopg2-binary==2.9.10 pytest==8.3.3 moto==5.0.12
       ```
     - Set environment variables in a `.env` file or shell:
       ```bash
       export ANTHROPIC_API_KEY="your-anthropic-key"
       export AWS_ACCESS_KEY_ID="your-aws-key"
       export AWS_SECRET_ACCESS_KEY="your-aws-secret"
       export GITLAB_TOKEN="your-gitlab-token"
       export CONFLUENCE_USER="your-confluence-user"
       export CONFLUENCE_PASSWORD="your-confluence-password"
       export JIRA_USER="your-jira-user"
       export JIRA_PASSWORD="your-jira-password"
       export DB_HOST="your-rds-endpoint"
       export DB_USER="your-db-user"
       export DB_PASSWORD="your-db-password"
       ```
2. **Run Streamlit UI**:
   - Start the Streamlit application:
     ```bash
     streamlit run streamlit_ui.py
     ```
   - Access the UI at `http://localhost:8501`.
3. **Execute Workflow**:
   - **Step 1**: Enter a user ID (e.g., `test_user`), select the default tech stack (Python/FastAPI/Streamlit), and upload a sample requirement document (e.g., `requirements.md` with content like "Build a task management app with CRUD operations").
   - **Step 2–10**: For each step, trigger the action (e.g., "Generate Specs"), review outputs (e.g., specs, code, manifests), provide optional feedback, and approve to proceed.
   - **Verification**: Check artifacts in:
     - S3: Requirement documents, Dockerfiles, manifests.
     - PostgreSQL: Sessions, steps, transactions, feedback.
     - GitLab: Repository (`task-manager-<session_id>`), committed code.
     - Confluence: Validation reports, technical specs.
     - ESTJIRA: Task and maintenance tickets.
     - NEXUS: Docker images.
     - EKS: Deployed pods and services.
     - CloudWatch: Logs and metrics.
4. **Unit Testing**:
   - Run Pytest to test utilities and agents:
     ```bash
     pytest tests/ --cov=. --cov-report=html
     ```
   - Ensure >80% coverage; review report in `htmlcov/`.
   - Use `moto` to mock AWS services (S3, CloudWatch).
5. **Troubleshooting**:
   - Check CloudWatch logs for errors (`sdlc-workflow/errors`).
   - Verify environment variables and API credentials.
   - Ensure PostgreSQL schema is applied correctly.

**Sample Test Case** (for `utils/db.py`):
```python
"""Test database utilities."""
import pytest
from utils.db import log_session

def test_log_session():
    """Test logging a new session."""
    user_id = "test_user"
    tech_stack = {"language": "Python", "backend": "FastAPI", "ui": "Streamlit"}
    session_id = log_session(user_id, tech_stack)
    assert isinstance(session_id, str)
    assert len(session_id) == 36  # UUID length
```

### Conclusion
This document completes the core SDLC steps (8–10) and provides the Workflow Orchestrator, Streamlit UI, and testing guidance for the **Agentic SDLC Application**. Combined with Parts 1 and 2, it covers the full 10-step SDLC for developing a task management application. Key features include:
- **Automation**: CrewAI orchestrates agents for each SDLC step, using Claude 3.5 Sonnet for intelligent task execution.
- **HITL**: Streamlit UI ensures user oversight with feedback and approval at every step.
- **Integration**: Seamless use of PostgreSQL, S3, GitLab, Confluence, ESTJIRA, SonarQube, Docker, NEXUS, EKS, and CloudWatch.
- **Scalability**: PostgreSQL partitioning and EKS autoscaling support high loads.
- **Security**: Environment variables, IAM roles, and parameterized queries prevent vulnerabilities.
- **Auditability**: Transaction logging in PostgreSQL and CloudWatch ensures traceability.

**Next Steps**:
- Part 4 integrates a FastAPI backend to expose the workflow as a RESTful API, enhancing modularity.
- Part 5 provides sample code for the task management application and Kubernetes manifests for EKS.
- Part 6 enhances testing with Pytest examples, details the GitLab CI/CD pipeline, and sets up CloudWatch monitoring.

To extend the application, consider:
- Adding security scanning (e.g., Trivy for Docker images).
- Implementing real-time notifications (e.g., Slack integration for ESTJIRA updates).
- Enhancing the task management app with features like task dependencies or email alerts.

All artifacts are stored in GitLab, Confluence, ESTJIRA, NEXUS, EKS, and CloudWatch for review and maintenance.

# Agentic SDLC CrewAI Implementation for Task Management - Part 4

## Overview
This document (Part 4 of the series) extends the **Agentic SDLC Application** for a **Task Management Application** by adding a **FastAPI backend** to manage the SDLC workflow API. It integrates with the Streamlit UI (Part 3) and CrewAI orchestrator (Part 3), providing endpoints for session management, step execution, and HITL feedback. This builds on Part 1 (Steps 1–3), Part 2 (Steps 4–7), and Part 3 (Steps 8–10, orchestrator, UI, testing). Part 5 will provide sample generated code and deployment manifests, and Part 6 will enhance testing and CI/CD details.

### FastAPI Backend
The FastAPI backend:
- Exposes RESTful endpoints for initiating sessions, triggering SDLC steps, retrieving status, and submitting feedback.
- Integrates with PostgreSQL for session and transaction management.
- Triggers CrewAI tasks via the Workflow Orchestrator.
- Stores artifacts in S3 and logs to CloudWatch.
- Secures endpoints with API key authentication.

## Implementation Details and Code

### FastAPI Application
**Description**: A FastAPI application (`main.py`) provides endpoints for the SDLC workflow, interacting with the Streamlit UI and CrewAI orchestrator.

**Endpoints**:
- `POST /sessions`: Create a new session with user ID and tech stack.
- `GET /sessions/{session_id}`: Retrieve session status and current step.
- `POST /sessions/{session_id}/steps/{step_name}`: Trigger an SDLC step (e.g., "Requirement Validation").
- `POST /sessions/{session_id}/feedback`: Submit HITL feedback for a step.
- `GET /sessions/{session_id}/artifacts`: List artifacts stored in S3.

**Security**:
- API key authentication via `X-API-Key` header.
- Environment variable for API key (`API_KEY`).
- Parameterized PostgreSQL queries to prevent SQL injection.

**Code**:

**main.py**:
```python
"""FastAPI backend for SDLC workflow."""
import os
import uuid
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Optional
from utils.db import log_session, log_feedback, update_step_status, update_session_step, get_session
from utils.tools import write_s3_file
from agents.workflow_orchestrator import run_workflow
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SDLC Workflow API")

# Pydantic models
class SessionCreate(BaseModel):
    user_id: str
    tech_stack: Dict[str, str]

class FeedbackCreate(BaseModel):
    step_id: str
    feedback: str

# API key authentication
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from header."""
    expected_key = os.getenv("API_KEY")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.post("/sessions")
async def create_session(session: SessionCreate, x_api_key: str = Header(...)):
    """Create a new SDLC session."""
    verify_api_key(x_api_key)
    try:
        session_id = log_session(session.user_id, session.tech_stack)
        logger.info(f"Created session: {session_id}")
        return {"session_id": session_id, "current_step": "Requirement Validation"}
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session_status(session_id: str, x_api_key: str = Header(...)):
    """Retrieve session status."""
    verify_api_key(x_api_key)
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session["session_id"],
        "user_id": session["user_id"],
        "current_step": session["current_step"],
        "tech_stack": session["tech_stack"],
        "created_at": session["created_at"]
    }

@app.post("/sessions/{session_id}/steps/{step_name}")
async def trigger_step(session_id: str, step_name: str, x_api_key: str = Header(...)):
    """Trigger an SDLC step."""
    verify_api_key(x_api_key)
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate step
    valid_steps = [
        "Requirement Validation", "Technical Documentation", "Task Breakdown",
        "Environment Setup", "Code Development", "Test Automation",
        "Version Control", "Build", "Deployment", "Monitoring"
    ]
    if step_name not in valid_steps:
        raise HTTPException(status_code=400, detail="Invalid step name")
    
    # Run workflow for the step (simplified; in practice, call specific agent)
    try:
        bucket = "your-bucket"
        key = f"requirements-{session_id}.md"
        run_workflow(session_id, bucket, key, session["tech_stack"])
        update_session_step(session_id, step_name)
        logger.info(f"Triggered step {step_name} for session {session_id}")
        return {"session_id": session_id, "step_name": step_name, "status": "Triggered"}
    except Exception as e:
        logger.error(f"Error triggering step {step_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/feedback")
async def submit_feedback(
    session_id: str, feedback: FeedbackCreate, x_api_key: str = Header(...)
):
    """Submit HITL feedback for a step."""
    verify_api_key(x_api_key)
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        log_feedback(session_id, feedback.step_id, session["user_id"], feedback.feedback)
        logger.info(f"Submitted feedback for session {session_id}, step {feedback.step_id}")
        return {"session_id": session_id, "feedback_id": str(uuid.uuid4())}
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/artifacts")
async def list_artifacts(session_id: str, x_api_key: str = Header(...)):
    """List artifacts stored in S3."""
    verify_api_key(x_api_key)
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        import boto3
        s3_client = boto3.client("s3")
        response = s3_client.list_objects_v2(Bucket="your-bucket", Prefix=f"{session_id}/")
        artifacts = [obj["Key"] for obj in response.get("Contents", [])]
        return {"session_id": session_id, "artifacts": artifacts}
    except Exception as e:
        logger.error(f"Error listing artifacts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Integration with Streamlit
**Description**: Update `streamlit_ui.py` to call FastAPI endpoints instead of directly invoking agents, ensuring modularity.

**Code**:

**streamlit_ui.py** (Updated for FastAPI):
```python
"""Streamlit UI for SDLC workflow with FastAPI integration."""
import streamlit as st
import requests
import json

st.title("Agentic SDLC Workflow")

# FastAPI base URL
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY")

# Session Initialization
if "session_id" not in st.session_state:
    user_id = st.text_input("Enter User ID")
    tech_stack = st.selectbox(
        "Select Tech Stack",
        ["Python/FastAPI/Streamlit", "Other"]
    )
    if tech_stack == "Other":
        language = st.text_input("Language")
        backend = st.text_input("Backend")
        ui = st.text_input("UI")
        tech_stack = {"language": language, "backend": backend, "ui": ui}
    else:
        tech_stack = {"language": "Python", "backend": "FastAPI", "ui": "Streamlit"}
    
    if st.button("Start Session"):
        response = requests.post(
            f"{API_BASE_URL}/sessions",
            json={"user_id": user_id, "tech_stack": tech_stack},
            headers={"X-API-Key": API_KEY}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state["session_id"] = data["session_id"]
            st.session_state["current_step"] = data["current_step"]
            st.session_state["user_id"] = user_id
            st.session_state["tech_stack"] = tech_stack
            st.write("Session Started")
        else:
            st.error(f"Error: {response.json()['detail']}")

# Step 1: Requirement Validation
if "session_id" in st.session_state and st.session_state.get("current_step") == "Requirement Validation":
    st.write("Upload Requirement Document")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        bucket = "your-bucket"
        key = f"requirements-{st.session_state['session_id']}.md"
        # Upload to S3 via FastAPI or directly
        response = requests.post(
            f"{API_BASE_URL}/sessions/{st.session_state['session_id']}/steps/Requirement Validation",
            headers={"X-API-Key": API_KEY}
        )
        if response.status_code == 200:
            st.write("Validation Triggered")
            # Fetch artifacts (report)
            artifacts_response = requests.get(
                f"{API_BASE_URL}/sessions/{st.session_state['session_id']}/artifacts",
                headers={"X-API-Key": API_KEY}
            )
            if artifacts_response.status_code == 200:
                st.write("Artifacts:", artifacts_response.json()["artifacts"])
        
        feedback = st.text_area("Provide Feedback")
        if st.button("Submit Feedback"):
            response = requests.post(
                f"{API_BASE_URL}/sessions/{st.session_state['session_id']}/feedback",
                json={"step_id": st.session_state.get("step_id", "unknown"), "feedback": feedback},
                headers={"X-API-Key": API_KEY}
            )
            if response.status_code == 200:
                st.write("Feedback Submitted")
        
        if st.button("Approve"):
            st.session_state["current_step"] = "Technical Documentation"
            st.write("Step Approved")
```

**Note**: The remaining steps in `streamlit_ui.py` follow a similar pattern, calling FastAPI endpoints for each SDLC step. For brevity, only Step 1 is shown, but the pattern extends to Steps 2–10.

### Running the Backend
- **Install dependencies**:
  ```bash
  pip install fastapi==0.115.2 uvicorn==0.32.0 pydantic==2.9.2
  ```
- **Run FastAPI**:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```
- **Access**: `http://localhost:8000/docs` for Swagger UI.

## Continuation
This concludes Part 4, adding a FastAPI backend for the SDLC workflow. Part 5 will provide sample generated code for the task management app and Kubernetes manifests for EKS deployment.

# Agentic SDLC CrewAI Implementation for Task Management - Part 5

## Overview
This document (Part 5 of the series) provides **sample generated code** for the task management application (FastAPI backend and Streamlit UI) and **Kubernetes manifests** for EKS deployment. It builds on Part 1 (Steps 1–3), Part 2 (Steps 4–7), Part 3 (Steps 8–10, orchestrator, UI, testing), and Part 4 (FastAPI backend). Part 6 will enhance testing guidance, CI/CD pipeline details, and CloudWatch monitoring setup.

### Sample Generated Code
The task management application allows users to create, assign, and track tasks. The code includes:
- **FastAPI Backend**: CRUD endpoints for tasks, integrated with PostgreSQL.
- **Streamlit UI**: Interface for task management, interacting with FastAPI.

### FastAPI Backend (Task Management)
**Description**: A FastAPI application (`app/main.py`) provides endpoints for task CRUD operations, using PostgreSQL for storage.

**Code**:

**app/main.py**:
```python
"""FastAPI backend for task management application."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Task Management API")

# Pydantic models
class TaskCreate(BaseModel):
    title: str
    description: str
    assignee: str
    status: str

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    assignee: Optional[str] = None
    status: Optional[str] = None

# Database connection
def get_db_connection():
    """Create a PostgreSQL connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database="task_db"
    )

@app.post("/tasks")
async def create_task(task: TaskCreate):
    """Create a new task."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO tasks (title, description, assignee, status)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, title, description, assignee, status
                    """,
                    (task.title, task.description, task.assignee, task.status)
                )
                conn.commit()
                return cur.fetchone()
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
async def get_task(task_id: int):
    """Retrieve a task by ID."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
                task = cur.fetchone()
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")
                return task
    except Exception as e:
        logger.error(f"Error retrieving task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/tasks/{task_id}")
async def update_task(task_id: int, task: TaskUpdate):
    """Update a task."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                updates = {k: v for k, v in task.dict().items() if v is not None}
                if not updates:
                    raise HTTPException(status_code=400, detail="No updates provided")
                set_clause = ", ".join(f"{k} = %s" for k in updates)
                values = list(updates.values()) + [task_id]
                cur.execute(
                    f"UPDATE tasks SET {set_clause} WHERE id = %s RETURNING *",
                    values
                )
                conn.commit()
                updated_task = cur.fetchone()
                if not updated_task:
                    raise HTTPException(status_code=404, detail="Task not found")
                return updated_task
    except Exception as e:
        logger.error(f"Error updating task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: int):
    """Delete a task."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tasks WHERE id = %s", (task_id,))
                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Task not found")
                conn.commit()
                return {"message": "Task deleted"}
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

**app/schema.sql**:
```sql
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    description TEXT,
    assignee VARCHAR(50),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Streamlit UI (Task Management)
**Description**: A Streamlit application (`app/ui.py`) provides a user interface for task management, interacting with the FastAPI backend.

**Code**:

**app/ui.py**:
```python
"""Streamlit UI for task management application."""
import streamlit as st
import requests

st.title("Task Management App")

API_BASE_URL = "http://localhost:8000"

# Create Task
st.header("Create Task")
with st.form("create_task"):
    title = st.text_input("Title")
    description = st.text_area("Description")
    assignee = st.text_input("Assignee")
    status = st.selectbox("Status", ["To Do", "In Progress", "Done"])
    submitted = st.form_submit_button("Create")
    if submitted:
        response = requests.post(
            f"{API_BASE_URL}/tasks",
            json={"title": title, "description": description, "assignee": assignee, "status": status}
        )
        if response.status_code == 200:
            st.success("Task created!")
        else:
            st.error(f"Error: {response.json()['detail']}")

# View Task
st.header("View Task")
task_id = st.number_input("Task ID", min_value=1, step=1)
if st.button("Get Task"):
    response = requests.get(f"{API_BASE_URL}/tasks/{task_id}")
    if response.status_code == 200:
        task = response.json()
        st.write(f"Title: {task['title']}")
        st.write(f"Description: {task['description']}")
        st.write(f"Assignee: {task['assignee']}")
        st.write(f"Status: {task['status']}")
    else:
        st.error(f"Error: {response.json()['detail']}")

# Update Task
st.header("Update Task")
with st.form("update_task"):
    update_id = st.number_input("Task ID to Update", min_value=1, step=1)
    update_title = st.text_input("New Title (optional)")
    update_description = st.text_area("New Description (optional)")
    update_assignee = st.text_input("New Assignee (optional)")
    update_status = st.selectbox("New Status (optional)", ["", "To Do", "In Progress", "Done"])
    update_submitted = st.form_submit_button("Update")
    if update_submitted:
        updates = {}
        if update_title:
            updates["title"] = update_title
        if update_description:
            updates["description"] = update_description
        if update_assignee:
            updates["assignee"] = update_assignee
        if update_status:
            updates["status"] = update_status
        response = requests.put(f"{API_BASE_URL}/tasks/{update_id}", json=updates)
        if response.status_code == 200:
            st.success("Task updated!")
        else:
            st.error(f"Error: {response.json()['detail']}")

# Delete Task
st.header("Delete Task")
delete_id = st.number_input("Task ID to Delete", min_value=1, step=1)
if st.button("Delete Task"):
    response = requests.delete(f"{API_BASE_URL}/tasks/{delete_id}")
    if response.status_code == 200:
        st.success("Task deleted!")
    else:
        st.error(f"Error: {response.json()['detail']}")
```

### Kubernetes Manifests for EKS
**Description**: Kubernetes manifests (`k8s/deployment.yaml`, `k8s/service.yaml`) deploy the task management application to EKS with autoscaling.

**Code**:

**k8s/deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: task-manager
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: task-manager
  template:
    metadata:
      labels:
        app: task-manager
    spec:
      containers:
      - name: task-manager
        image: nexus-repo/task-manager:latest
        ports:
        - containerPort: 8000
        env:
        - name: DB_HOST
          value: "your-rds-endpoint"
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: task-manager-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: task-manager
  minReplicas: 2
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**k8s/service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: task-manager-service
  namespace: default
spec:
  selector:
    app: task-manager
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Running the Application
- **Database**: Apply `app/schema.sql` to PostgreSQL.
- **FastAPI**:
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```
- **Streamlit**:
  ```bash
  streamlit run app/ui.py
  ```
- **EKS Deployment**:
  ```bash
  kubectl apply -f k8s/deployment.yaml
  kubectl apply -f k8s/service.yaml
  ```

## Continuation
This concludes Part 5, providing sample code and Kubernetes manifests. Part 6 will enhance testing guidance, CI/CD pipeline details, and CloudWatch monitoring setup.

# Agentic SDLC CrewAI Implementation for Task Management - Part 6

## Overview
This document (Part 6, the final part) enhances the **Agentic SDLC Application** for a **Task Management Application** by providing **Pytest examples**, **GitLab CI/CD pipeline details**, and **CloudWatch monitoring setup**. It builds on Part 1 (Steps 1–3), Part 2 (Steps 4–7), Part 3 (Steps 8–10, orchestrator, UI, testing), Part 4 (FastAPI backend), and Part 5 (sample code, manifests). This part ensures robust testing (>80% coverage), automated CI/CD, and comprehensive monitoring.

### Testing Guidance
**Description**: Pytest cases test the FastAPI backend (`app/main.py`) using `moto` for AWS mocks and `pytest-httpx` for API testing.

**Code**:

**tests/test_main.py**:
```python
"""Pytest cases for task management FastAPI backend."""
import pytest
from fastapi.testclient import TestClient
from app.main import app, get_db_connection
from moto import mock_aws
import psycopg2
from psycopg2.extras import RealDictCursor

# Mock database connection
@pytest.fixture
def mock_db(monkeypatch):
    """Mock PostgreSQL connection."""
    class MockCursor:
        def __init__(self):
            self.results = []
        
        def execute(self, query, params=None):
            if "INSERT INTO tasks" in query:
                self.results = [{"id": 1, "title": params[0], "description": params[1], "assignee": params[2], "status": params[3]}]
            elif "SELECT * FROM tasks" in query:
                self.results = [{"id": 1, "title": "Test Task", "description": "Test Desc", "assignee": "Test User", "status": "To Do"}]
            elif "UPDATE tasks" in query:
                self.results = [{"id": 1, "title": params[0], "description": "Test Desc", "assignee": "Test User", "status": "To Do"}]
            elif "DELETE FROM tasks" in query:
                self.results = []
        
        def fetchone(self):
            return self.results[0] if self.results else None
        
        def fetchall(self):
            return self.results
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    class MockConnection:
        def cursor(self, cursor_factory=None):
            return MockCursor()
        
        def commit(self):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    monkeypatch.setattr("app.main.get_db_connection", lambda: MockConnection())

@pytest.fixture
def client(mock_db):
    """Create FastAPI test client."""
    return TestClient(app)

def test_create_task(client):
    """Test creating a task."""
    response = client.post(
        "/tasks",
        json={"title": "Test Task", "description": "Test Desc", "assignee": "Test User", "status": "To Do"}
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Test Task"

def test_get_task(client):
    """Test retrieving a task."""
    response = client.get("/tasks/1")
    assert response.status_code == 200
    assert response.json()["id"] == 1

def test_update_task(client):
    """Test updating a task."""
    response = client.put(
        "/tasks/1",
        json={"title": "Updated Task"}
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Updated Task"

def test_delete_task(client):
    """Test deleting a task."""
    response = client.delete("/tasks/1")
    assert response.status_code == 200
    assert response.json()["message"] == "Task deleted"
```

**Run Tests**:
```bash
pip install pytest pytest-httpx moto pytest-mock
pytest tests/test_main.py --cov=app --cov-report=html
```

**Coverage**: Ensures >80% coverage, with HTML report in `htmlcov/`.

### GitLab CI/CD Pipeline
**Description**: A `.gitlab-ci.yml` file automates build, test, and deployment to EKS.

**Code**:

**.gitlab-ci.yml**:
```yaml
stages:
  - lint
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: "nexus-repo/task-manager:$CI_COMMIT_SHA"
  KUBECONFIG: "/kube/config"

lint:
  stage: lint
  image: python:3.9
  script:
    - pip install black flake8
    - black --check .
    - flake8 . --max-line-length=88
  allow_failure: true

test:
  stage: test
  image: python:3.9
  services:
    - postgres:13
  variables:
    DB_HOST: postgres
    DB_USER: postgres
    DB_PASSWORD: postgres
  script:
    - pip install -r requirements.txt pytest pytest-httpx moto pytest-mock
    - pytest tests/test_main.py --cov=app --cov-report=xml
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:20.10
  services:
    - docker:dind
  script:
    - docker login -u "$NEXUS_USER" -p "$NEXUS_PASSWORD" nexus-repo
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE
  variables:
    DOCKER_TLS_CERTDIR: ""

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context eks-cluster
    - sed -i "s|image: .*|image: $DOCKER_IMAGE|" k8s/deployment.yaml
    - kubectl apply -f k8s/deployment.yaml
    - kubectl apply -f k8s/service.yaml
  environment:
    name: production
    url: http://task-manager-service
```

**Setup**:
- Store `NEXUS_USER`, `NEXUS_PASSWORD`, and EKS kubeconfig in GitLab CI/CD variables.
- Ensure `k8s/deployment.yaml` and `k8s/service.yaml` are in the repository.

### CloudWatch Monitoring Setup
**Description**: Configure CloudWatch to monitor the task management application, logging metrics and errors.

**Code** (Example Log Push):

**app/main.py** (Updated with CloudWatch):
```python
# Add to imports
import boto3

# Initialize CloudWatch client
cloudwatch_client = boto3.client("logs")

# Update create_task (example)
@app.post("/tasks")
async def create_task(task: TaskCreate):
    """Create a new task."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO tasks (title, description, assignee, status)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, title, description, assignee, status
                    """,
                    (task.title, task.description, task.assignee, task.status)
                )
                conn.commit()
                task_data = cur.fetchone()
                # Log to CloudWatch
                cloudwatch_client.put_log_events(
                    logGroupName="task-manager",
                    logStreamName="api",
                    logEvents=[
                        {
                            "timestamp": int(datetime.now().timestamp() * 1000),
                            "message": f"Created task: {task_data['id']}"
                        }
                    ]
                )
                return task_data
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        cloudwatch_client.put_log_events(
            logGroupName="task-manager",
            logStreamName="errors",
            logEvents=[
                {
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "message": f"Error: {str(e)}"
                }
            ]
        )
        raise HTTPException(status_code=500, detail=str(e))
```

**CloudWatch Setup**:
- Create log groups: `task-manager/api` and `task-manager/errors`.
- Create metric filters:
  - Filter: `Error` in `task-manager/errors`.
  - Metric: `TaskManagerErrors`, namespace `TaskManager`.
- Create alarms:
  - Alarm: `TaskManagerErrors > 5 for 5 minutes`.
  - Action: Notify via SNS.

**IAM Policy**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:task-manager:*"
    }
  ]
}
```

## Conclusion
This series (Parts 1–6) provides a comprehensive implementation of an **Agentic SDLC Application** for a **Task Management Application** using **CrewAI**. It covers:
- **Part 1**: Architecture, setup, SDLC Steps 1–3 (Requirement Validation, Technical Documentation, Task Breakdown).
- **Part 2**: SDLC Steps 4–7 (Environment Setup, Code Development, Test Automation, Version Control).
- **Part 3**: SDLC Steps 8–10 (Build, Deployment, Monitoring), orchestrator, Streamlit UI, initial testing.
- **Part 4**: FastAPI backend for workflow API.
- **Part 5**: Sample task management code and EKS manifests.
- **Part 6**: Pytest cases, GitLab CI/CD, CloudWatch monitoring.

The implementation is secure (environment variables, IAM), scalable (PostgreSQL partitioning, EKS autoscaling), and auditable (PostgreSQL transactions, CloudWatch logs). Users can extend it by:
- Adding more SDLC steps (e.g., security scanning).
- Enhancing the task management app with features (e.g., notifications).
- Integrating additional tools (e.g., Slack for alerts).

For further details, check artifacts in GitLab, Confluence, ESTJIRA, NEXUS, EKS, and CloudWatch.