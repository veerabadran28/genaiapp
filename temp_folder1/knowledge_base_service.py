import boto3
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
from botocore.exceptions import ClientError
import time
from botocore.config import Config
import random
from contextlib import contextmanager
import re
from services.bedrock_service import BedrockService
from services.s3_service import S3Service
import csv
from io import StringIO
import traceback
import tempfile
import polars as pl
import s3fs
import uuid

class KnowledgeBaseService:
    def __init__(self, config: Dict):
        """Initialize Knowledge Base service with configuration."""
        self.config = config
        self.logger = self._setup_logger()
        
        # Configure retry settings
        boto3_config = Config(
            retries=dict(
                max_attempts=3,
                mode='adaptive'
            ),
            connect_timeout=5,
            read_timeout=60
        )
        
        self.bedrock_kb = boto3.client(
            'bedrock-agent-runtime',
            region_name='us-east-1',
            config=boto3_config
        )
        
        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            region_name='us-east-1',
            config=boto3_config
        )
        
        # General KB config for backward compatibility
        self.knowledge_base_id = config.get('knowledge_base_id')
        self.data_source_id = config.get('data_source_id')
        
        # Configure available programs
        self.programs = config.get('programs', {})
        
        # General parameters
        self.max_results = config.get('max_results', 10)
        
        # Default model configurations
        model_config = config.get('model_config', {})
        self.default_model_id = model_config.get('model_id', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
        self.default_nova_model_id = model_config.get('nova_model_id', 'amazon.nova-pro-v1:0')
        
        # Rate limiting parameters
        self.max_history = 10
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum time between requests in seconds
        
        # S3 setup
        self.s3_client = boto3.client('s3')
        self.s3_fs = s3fs.S3FileSystem()
        
        # Default metadata configuration
        self.default_metadata_bucket = config.get('s3_config', {}).get('bucket', 'myawstests3buckets1')
        self.default_metadata_prefix = "metadata"
        self.default_metadata_filename = "program_data_metadata.json"
        self.default_max_files = 5
        self.default_sample_limit = 10
        
        # Execution environment
        self.temp_dir = tempfile.gettempdir()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the Knowledge Base service."""
        logger = logging.getLogger('KnowledgeBaseService')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _wait_for_rate_limit(self):
        """Implement rate limiting with jitter and increased delays."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Increased base delay to 2 seconds
        min_delay = 2.0
        
        if time_since_last_request < min_delay:
            # Add larger jitter (0.5 to 1.5 seconds)
            jitter = random.uniform(0.5, 1.5)
            sleep_time = min_delay - time_since_last_request + jitter
            self.logger.info(f"Rate limiting: Waiting {sleep_time:.2f} seconds before next request")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _make_request_with_retry(self, func, *args, **kwargs):
        """Make request with retry logic for throttling."""
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    if attempt == max_retries - 1:  # Last attempt
                        raise  # Re-raise if we're out of retries
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = (base_delay ** attempt) + random.uniform(0.1, 1.0)
                    self.logger.warning(f"Request throttled. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise  # Re-raise non-throttling exceptions

    def search_documents_latest(self, query: str, bedrock_service, s3_service, prompt_template_file: str, 
                               program_config: dict, chat_history: Optional[List[Dict]] = None, 
                               use_nova: bool = False, program_id: str = None) -> str:
        """
        Enhanced document search with metadata-driven file selection and dynamic code generation.
        
        Args:
            query (str): User query about documents
            bedrock_service: Service for calling Bedrock models
            s3_service: Service for S3 operations
            prompt_template_file (str): Path to the prompt template file
            program_config (dict): Program configuration
            chat_history (Optional[List[Dict]]): Previous conversation history
            use_nova (bool): Whether to use Nova model instead of Claude
            program_id (str): Optional program identifier
            
        Returns:
            str: JSON-formatted response with answer and chart data
        """
        try:
            self.logger.info(f"Original User Query: {query}")
            
            # Step 1: Enrich query if mapping config exists
            if program_config and program_config.get('mapping_config'):
                enriched_query = self._enrich_user_query(query, bedrock_service, s3_service, program_config)
                self.logger.info(f"Enriched User Query: {enriched_query}")
            else:
                enriched_query = query
                self.logger.info("No mapping configuration found, using original query")
            
            # Step 2: Get metadata file from knowledge base to find relevant files
            max_files = program_config.get('metadata_config', {}).get('max_files', self.default_max_files)
            metadata_filename = program_config.get('metadata_config', {}).get('filename', 
                                  f"{program_id}_data_metadata.json" if program_id else self.default_metadata_filename)
            metadata_prefix = program_config.get('metadata_config', {}).get('prefix', self.default_metadata_prefix)
                
            relevant_files_with_metadata = self._get_relevant_files_from_metadata(
                enriched_query, max_files, metadata_prefix, metadata_filename, program_config)
            
            if not relevant_files_with_metadata:
                return json.dumps({
                    "answer": f"I couldn't find relevant data files that match your query. Please try a different question or provide more specific details about the data you're looking for.",
                    "chart_data": []
                })
            
            # Step 3: Generate Python code to process the relevant files
            python_code = self._generate_data_processing_code(enriched_query, relevant_files_with_metadata, program_config)
            
            # Step 4: Execute the generated code with error handling and self-correction
            max_correction_attempts = program_config.get('code_generation', {}).get('max_correction_attempts', 5)
            datasets = None
            error_message = None
            
            for attempt in range(max_correction_attempts):
                try:
                    self.logger.info(f"Executing generated Python code (attempt {attempt+1})")
                    datasets = self._execute_python_code(python_code, program_config)
                    error_message = None
                    break
                except Exception as e:
                    error_message = str(e)
                    traceback_str = traceback.format_exc()
                    self.logger.error(f"Error executing Python code (attempt {attempt+1}): {error_message}\n{traceback_str}")
                    
                    if attempt < max_correction_attempts - 1:
                        # Try to get improved code
                        python_code = self._correct_python_code(
                            python_code, error_message, traceback_str, attempt + 1, 
                            bedrock_service, program_config)
                    else:
                        self.logger.error(f"Maximum correction attempts reached. Unable to execute Python code.")
            
            # Step 5: Handle execution results
            if error_message or not datasets:
                return json.dumps({
                    "answer": f"I was unable to process the data files to answer your question after several attempts. Technical error: {error_message}",
                    "chart_data": []
                })
            
            # Step 6: Generate final response using datasets as context
            with open(prompt_template_file, 'r') as file:
                prompt_template = file.read()
            
            # Format the final answer using the template and datasets
            final_response = self._generate_final_response(
                enriched_query, datasets, prompt_template, chat_history, use_nova, 
                bedrock_service, program_config)
            
            return final_response

        except Exception as e:
            self.logger.error(f"Error during enhanced document search: {str(e)}\n{traceback.format_exc()}")
            return json.dumps({
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "chart_data": []
            })

    def _get_relevant_files_from_metadata(self, enriched_query: str, max_files: int, 
                                          metadata_prefix: str, metadata_filename: str,
                                          program_config: Dict) -> List[Dict]:
        """
        Query the knowledge base to find the most relevant files based on metadata.
        
        Args:
            enriched_query (str): The enriched user query
            max_files (int): Maximum number of files to return
            metadata_prefix (str): Prefix for metadata file
            metadata_filename (str): Filename for metadata
            program_config (Dict): Program configuration
            
        Returns:
            List[Dict]: List of relevant file metadata
        """
        try:
            self.logger.info(f"Finding relevant files for query: {enriched_query}")
            
            # Get bucket from program config or default
            metadata_bucket = program_config.get('metadata_config', {}).get('bucket', self.default_metadata_bucket)
            
            # Step 1: Get metadata file from S3
            metadata_key = f"{metadata_prefix}/{metadata_filename}"
            
            try:
                # Try to get metadata from S3
                response = self.s3_client.get_object(Bucket=metadata_bucket, Key=metadata_key)
                metadata_content = response['Body'].read().decode('utf-8')
                metadata = json.loads(metadata_content)
                self.logger.info(f"Retrieved metadata file directly from S3 with {len(metadata)} entries")
            except Exception as s3_error:
                # If S3 direct access fails, try to get metadata from Knowledge Base
                self.logger.warning(f"Failed to get metadata from S3 directly: {str(s3_error)}. Trying Knowledge Base...")
                kb_id = program_config.get('knowledge_base_id', self.knowledge_base_id)
                metadata = self._get_metadata_from_knowledge_base(metadata_filename, kb_id)
            
            if not metadata:
                self.logger.error("Failed to retrieve metadata file")
                return []
                
            # Step 2: Use LLM to identify the most relevant files
            model_selection = program_config.get('query_enrichment', {}).get('model', 'nova')
            model_id = self.default_nova_model_id if model_selection == 'nova' else self.default_model_id
            
            # Override with program-specific model if available
            if 'models' in program_config:
                models = program_config.get('models', {})
                default_model = models.get('default')
                
                # Map model name to model ID
                if default_model == "Claude" and model_selection != 'nova':
                    model_id = self.default_model_id
                elif default_model == "Nova" or model_selection == 'nova':
                    model_id = self.default_nova_model_id
            
            # Prepare prompt for file selection
            program_name = program_config.get('name', 'Program')
            file_selection_prompt = f"""
            I need to find the most relevant {program_name} data files to answer this question: 
            "{enriched_query}"
            
            Given the following metadata about our {program_name} data files, please identify the top {max_files} most relevant files 
            that would help answer this question.
            
            Here is the metadata for all available files:
            ```json
            {json.dumps(metadata, indent=2)}
            ```
            
            Focus on:
            1. Files with column names mentioned or implied in the query
            2. Files with appropriate scenarios if mentioned in the query
            3. Files with relevant data based on file path components
            
            Please return a JSON array with ONLY the s3_file_path values for the {max_files} most relevant files, sorted by relevance.
            Your response should be a valid JSON array of strings, nothing else. For example:
            ["s3://bucket/path/to/file1.csv", "s3://bucket/path/to/file2.csv"]
            """
            
            # Invoke model to find relevant files
            if model_selection == 'nova':
                response = self._make_request_with_retry(
                    self.bedrock_runtime.invoke_model,
                    modelId=model_id,
                    contentType='application/json',
                    accept='application/json',
                    body=json.dumps({
                        "prompt": file_selection_prompt,
                        "max_tokens": 1000,
                        "temperature": 0.0,
                        "top_p": 0.9
                    })
                )
                
                # Parse response
                response_body = json.loads(response['body'].read().decode('utf-8'))
                completion = response_body.get('completion', '')
            else:
                response = self._make_request_with_retry(
                    self.bedrock_runtime.invoke_model,
                    modelId=model_id,
                    contentType='application/json',
                    accept='application/json',
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1000,
                        "temperature": 0.0,
                        "messages": [
                            {
                                "role": "user",
                                "content": file_selection_prompt
                            }
                        ]
                    })
                )
                
                # Parse response
                response_body = json.loads(response['body'].read().decode('utf-8'))
                completion = response_body.get('content', [{}])[0].get('text', '')
            
            # Extract JSON array from the completion
            try:
                # Find all JSON array patterns in the text
                json_pattern = r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]'
                matches = re.findall(json_pattern, completion)
                
                if matches:
                    file_paths = json.loads(matches[0])
                    self.logger.info(f"Found {len(file_paths)} relevant files")
                else:
                    # Try to get anything that looks like a valid JSON from the text
                    json_start = completion.find('[')
                    json_end = completion.rfind(']') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_text = completion[json_start:json_end]
                        file_paths = json.loads(json_text)
                        self.logger.info(f"Found {len(file_paths)} relevant files using JSON extraction")
                    else:
                        self.logger.error("Could not extract file paths from model response")
                        return []
            except Exception as json_error:
                self.logger.error(f"Error parsing model response: {str(json_error)}")
                # Fallback to simple parsing
                try:
                    # Try to extract anything that looks like S3 paths
                    s3_pattern = r's3://[^\s,"\']+\.csv'
                    file_paths = re.findall(s3_pattern, completion)
                    self.logger.info(f"Found {len(file_paths)} relevant files using regex")
                except Exception:
                    self.logger.error("Failed to extract file paths using regex")
                    return []
            
            # Step 3: Get metadata for the selected files
            relevant_files_with_metadata = []
            for file_path in file_paths[:max_files]:  # Limit to max_files
                if file_path in metadata:
                    relevant_files_with_metadata.append(metadata[file_path])
                else:
                    # Try to match by file name if exact match fails
                    for key, value in metadata.items():
                        if file_path.split('/')[-1] in key:
                            relevant_files_with_metadata.append(value)
                            break
            
            self.logger.info(f"Returning metadata for {len(relevant_files_with_metadata)} files")
            return relevant_files_with_metadata
            
        except Exception as e:
            self.logger.error(f"Error getting relevant files from metadata: {str(e)}\n{traceback.format_exc()}")
            return []

    def _get_metadata_from_knowledge_base(self, metadata_filename: str, kb_id: str) -> Dict:
        """
        Retrieve metadata file from Knowledge Base.
        
        Args:
            metadata_filename (str): Name of the metadata file
            kb_id (str): Knowledge Base ID
            
        Returns:
            Dict: The metadata dictionary
        """
        try:
            # Create vector search config to find the metadata file
            vector_search_config = {
                "numberOfResults": 1,
                "filter": {
                    "equals": {
                        "key": "filename",
                        "value": metadata_filename
                    }
                }
            }
            
            # Query the knowledge base
            response = self._make_request_with_retry(
                self.bedrock_kb.retrieve,
                knowledgeBaseId=kb_id,
                retrievalQuery={"text": "data metadata"},
                retrievalConfiguration={
                    "vectorSearchConfiguration": vector_search_config
                }
            )
            
            metadata_content = None
            
            # Extract metadata content from response
            if response and "retrievalResults" in response:
                for result in response["retrievalResults"]:
                    content = result.get("content", {}).get("text", "")
                    if content and "data_metadata" in content:
                        metadata_content = content
                        break
            
            if not metadata_content:
                self.logger.error("Could not retrieve metadata file from knowledge base")
                return {}
            
            # Parse JSON content from the retrieved text
            metadata_json_start = metadata_content.find('{')
            metadata_json_end = metadata_content.rfind('}') + 1
            
            if metadata_json_start >= 0 and metadata_json_end > metadata_json_start:
                metadata_json = metadata_content[metadata_json_start:metadata_json_end]
                try:
                    return json.loads(metadata_json)
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse metadata JSON")
                    return {}
            else:
                self.logger.error("Could not extract valid JSON from metadata content")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error retrieving metadata from knowledge base: {str(e)}")
            return {}

    def _generate_data_processing_code(self, enriched_query: str, relevant_files: List[Dict], 
                                       program_config: Dict) -> str:
        """
        Generate Python code to process the relevant files using Bedrock.
        
        Args:
            enriched_query (str): The enriched user query
            relevant_files (List[Dict]): List of relevant file metadata
            program_config (Dict): Program configuration
            
        Returns:
            str: Generated Python code
        """
        try:
            # Get code generation parameters from program config
            code_gen_config = program_config.get('code_generation', {})
            model_selection = code_gen_config.get('model', 'claude')
            temperature = code_gen_config.get('temperature', 0.2)
            max_tokens = code_gen_config.get('max_tokens', 4000)
            
            # Determine model ID based on configuration
            if model_selection == 'nova':
                model_id = self.default_nova_model_id
            else:
                model_id = self.default_model_id
                
            # Override with program-specific model if available
            if 'models' in program_config:
                models = program_config.get('models', {})
                default_model = models.get('default')
                
                # Map model name to model ID
                if default_model == "Claude" and model_selection != 'nova':
                    model_id = self.default_model_id
                elif default_model == "Nova" or model_selection == 'nova':
                    model_id = self.default_nova_model_id
            
            # Get program name for prompt context
            program_name = program_config.get('name', 'Program')
            
            # Prepare code generation prompt
            code_prompt = f"""
            You are a data analyst working with {program_name} files. You need to write Python code 
            to answer this user question: "{enriched_query}"
            
            I've identified the following files that contain relevant data to answer this question:
            
            ```json
            {json.dumps(relevant_files, indent=2)}
            ```
            
            Write a Python function that:
            1. Uses polars to efficiently process these pipe-delimited CSV files from S3
            2. Filters, aggregates, and transforms the data as needed to answer the question
            3. Returns a structured dataset suitable for visualization and analysis
            
            Requirements:
            - Import all necessary libraries (polars as pl, s3fs, json, etc.)
            - Use polars' lazy API where possible for efficient processing
            - Process all relevant files efficiently, joining them if necessary
            - Filter and aggregate data intelligently based on the question
            - Handle large datasets efficiently (files could be multiple GB)
            - Return results in a standardized format that can be easily visualized
            - Include clear comments explaining the logic
            - Handle potential errors gracefully
            
            Return ONLY the Python code with no explanations or other text. Your code should be wrapped in a single function called `process_data`
            that takes no arguments and returns a dictionary with the following structure:
            
            ```python
            def process_data():
                # Your code here
                
                # Return results in this format
                return {
                    "datasets": [
                        {
                            "header": ["Column1", "Column2", "Column3"],  # Column names
                            "rows": [
                                ["Value1", "Value2", 123.45],  # Row 1
                                ["Value1", "Value2", 567.89],  # Row 2
                                # More rows...
                            ],
                            "description": "Description of this dataset"
                        }
                        # Additional datasets can be included if needed
                    ],
                    "summary": "A textual summary of the data analysis results"
                }
            ```
            """
            
            # Invoke model to generate code based on model selection
            if model_selection == 'nova':
                response = self._make_request_with_retry(
                    self.bedrock_runtime.invoke_model,
                    modelId=model_id,
                    contentType='application/json',
                    accept='application/json',
                    body=json.dumps({
                        "prompt": code_prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.9
                    })
                )
                
                # Parse response
                response_body = json.loads(response['body'].read().decode('utf-8'))
                completion = response_body.get('completion', '')
            else:
                response = self._make_request_with_retry(
                    self.bedrock_runtime.invoke_model,
                    modelId=model_id,
                    contentType='application/json',
                    accept='application/json',
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system": "You are an expert Python programmer specializing in data analysis with polars and s3fs. Write efficient, production-ready code.",
                        "messages": [
                            {
                                "role": "user",
                                "content": code_prompt
                            }
                        ]
                    })
                )
                
                # Parse response
                response_body = json.loads(response['body'].read().decode('utf-8'))
                completion = response_body.get('content', [{}])[0].get('text', '')
            
            # Extract code from response
            code_pattern = r'```python\s*(.*?)\s*```'
            code_matches = re.findall(code_pattern, completion, re.DOTALL)
            
            if code_matches:
                # Use the first code block that contains the function definition
                for code_match in code_matches:
                    if 'def process_data' in code_match:
                        python_code = code_match
                        break
                else:
                    python_code = code_matches[0]
            else:
                # If no code block with ```python tag, try to extract the function directly
                function_pattern = r'def process_data\(\):(.*?)(?=\n\n|$)'
                function_matches = re.findall(function_pattern, completion, re.DOTALL)
                
                if function_matches:
                    python_code = f"def process_data():{function_matches[0]}"
                else:
                    # If all else fails, return the whole completion
                    python_code = completion
            
            # Clean up any remaining markdown markers
            python_code = python_code.replace('```python', '').replace('```', '')
            
            self.logger.info(f"Generated Python code of length {len(python_code)}")
            return python_code
            
        except Exception as e:
            self.logger.error(f"Error generating data processing code: {str(e)}\n{traceback.format_exc()}")
            raise

    def _execute_python_code(self, python_code: str, program_config: Dict) -> Dict:
        """
        Execute the generated Python code in a safe environment.
        
        Args:
            python_code (str): The Python code to execute
            program_config (Dict): Program configuration
            
        Returns:
            Dict: The results returned by the code
        """
        try:
            # Get execution parameters from program config
            exec_config = program_config.get('code_execution', {})
            timeout_seconds = exec_config.get('timeout_seconds', 300)
            
            # Create a unique filename for this execution
            execution_id = str(uuid.uuid4())
            code_file_path = f"{self.temp_dir}/program_code_{execution_id}.py"
            
            # Write code to file
            with open(code_file_path, 'w') as f:
                f.write(python_code)
                
            self.logger.info(f"Saved code to {code_file_path}")
            
            # Prepare execution environment
            execution_namespace = {
                'pl': pl,
                's3fs': s3fs,
                'json': json,
                'StringIO': StringIO,
                'time': time,
                'random': random,
                'traceback': traceback
            }
            
            # Import required modules
            exec("import polars as pl", execution_namespace)
            exec("import s3fs", execution_namespace)
            exec("import json", execution_namespace)
            exec("import time", execution_namespace)
            exec("import random", execution_namespace)
            exec("from io import StringIO", execution_namespace)
            
            # Execute the code with timeout
            with open(code_file_path, 'r') as f:
                code_content = f.read()
            
            # Set timer for execution if supported
            if hasattr(time, 'timeout'):
                with time.timeout(timeout_seconds):
                    exec(code_content, execution_namespace)
            else:
                # Fallback if timeout context manager is not available
                exec(code_content, execution_namespace)
            
            # Call the process_data function
            if 'process_data' in execution_namespace:
                result = execution_namespace['process_data']()
                self.logger.info("Python code execution successful")
                return result
            else:
                raise ValueError("The code doesn't contain a process_data function")
                
        except Exception as e:
            self.logger.error(f"Error executing Python code: {str(e)}\n{traceback.format_exc()}")
            raise

    def _correct_python_code(self, python_code: str, error_message: str, traceback_str: str, 
                             attempt: int, bedrock_service, program_config: Dict) -> str:
        """
        Correct errors in the Python code using Bedrock.
        
        Args:
            python_code (str): The original Python code
            error_message (str): The error message
            traceback_str (str): The traceback string
            attempt (int): The current attempt number
            bedrock_service: The Bedrock service for model invocation
            program_config (Dict): Program configuration
            
        Returns:
            str: The corrected Python code
        """
        try:
            self.logger.info(f"Attempting to fix Python code (attempt {attempt})")
            
            # Get code correction parameters from program config
            code_gen_config = program_config.get('code_generation', {})
            model_selection = code_gen_config.get('model', 'claude')
            temperature = code_gen_config.get('correction_temperature', 0.1)
            
            # Determine model ID based on configuration
            if model_selection == 'nova':
                model_id = self.default_nova_model_id
            else:
                model_id = self.default_model_id
                
            # Override with program-specific model if available
            if 'models' in program_config:
                models = program_config.get('models', {})
                default_model = models.get('default')
                
                # Map model name to model ID
                if default_model == "Claude" and model_selection != 'nova':
                    model_id = self.default_model_id
                elif default_model == "Nova" or model_selection == 'nova':
                    model_id = self.default_nova_model_id
                    
            # Get program name for context
            program_name = program_config.get('name', 'Program')
            
            # Prepare correction prompt
            correction_prompt = f"""
            I tried to execute the following Python code for {program_name} data analysis, but encountered an error.
            Please fix the code to address the specific error. Focus only on fixing the error, not rewriting the entire solution.
            
            ## Python Code:
            ```python
            {python_code}
            ```
            
            ## Error:
            ```
            {error_message}
            ```
            
            ## Traceback:
            ```
            {traceback_str}
            ```
            
            Please provide ONLY the fixed Python code with no explanations or other text.
            """
            
            # Generate corrected code based on model selection
            if bedrock_service:
                # Use provided bedrock service if available
                if model_selection == 'nova':
                    corrected_code = bedrock_service.invoke_nova_model(correction_prompt)
                else:
                    corrected_code = bedrock_service.invoke_model_simple(correction_prompt)
            else:
                # Direct invocation if bedrock_service is not provided
                if model_selection == 'nova':
                    response = self._make_request_with_retry(
                        self.bedrock_runtime.invoke_model,
                        modelId=model_id,
                        contentType='application/json',
                        accept='application/json',
                        body=json.dumps({
                            "prompt": correction_prompt,
                            "max_tokens": 4000,
                            "temperature": temperature,
                            "top_p": 0.9
                        })
                    )
                    
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    corrected_code = response_body.get('completion', '')
                else:
                    response = self._make_request_with_retry(
                        self.bedrock_runtime.invoke_model,
                        modelId=model_id,
                        contentType='application/json',
                        accept='application/json',
                        body=json.dumps({
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 4000,
                            "temperature": temperature,
                            "system": "You are an expert Python debugger. Fix code errors without changing the overall structure.",
                            "messages": [
                                {
                                    "role": "user", 
                                    "content": correction_prompt
                                }
                            ]
                        })
                    )
                    
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    corrected_code = response_body.get('content', [{}])[0].get('text', '')
            
            # Clean up the response
            code_pattern = r'```python\s*(.*?)\s*```'
            code_matches = re.findall(code_pattern, corrected_code, re.DOTALL)
            
            if code_matches:
                cleaned_code = code_matches[0]
            else:
                # If no code block with ```python tag, check if the response already looks like code
                if 'def process_data' in corrected_code:
                    cleaned_code = corrected_code
                else:
                    # Fall back to just removing any ``` markers
                    cleaned_code = corrected_code.replace('```python', '').replace('```', '')
            
            self.logger.info(f"Generated corrected code of length {len(cleaned_code)}")
            return cleaned_code
            
        except Exception as e:
            self.logger.error(f"Error correcting Python code: {str(e)}")
            # Return the original code if correction fails
            return python_code

    def _generate_final_response(self, query: str, datasets: Dict, prompt_template: str, 
                                 chat_history: Optional[List[Dict]], use_nova: bool, 
                                 bedrock_service, program_config: Dict) -> str:
        """
        Generate the final response using the processed datasets.
        
        Args:
            query (str): The original query
            datasets (Dict): The processed datasets
            prompt_template (str): The prompt template
            chat_history (Optional[List[Dict]]): Chat history
            use_nova (bool): Whether to use Nova model
            bedrock_service: Bedrock service for model invocation
            program_config (Dict): Program configuration
            
        Returns:
            str: The final response in JSON format
        """
        try:
            # Format datasets as context
            datasets_context = json.dumps(datasets, indent=2)
            
            # Build conversation context if needed
            conversation_context = self._build_conversation_context(chat_history, program_config) if chat_history else ""
            
            # Replace placeholders in prompt template
            formatted_prompt = prompt_template.replace("$context$", datasets_context)
            formatted_prompt = formatted_prompt.replace("$user_query$", query)
            
            # Add conversation context if available
            if conversation_context:
                formatted_prompt += f"\n\nPrevious Conversation Context:\n{conversation_context}"
            
            # Get model parameters from program config
            response_config = program_config.get('response_generation', {})
            model_selection = response_config.get('model', 'claude')
            temperature = response_config.get('temperature', 0.7)
            
            # Determine model ID based on configuration
            if use_nova or model_selection == 'nova':
                model_id = self.default_nova_model_id
            else:
                model_id = self.default_model_id
                
            # Override with program-specific model if available
            if 'models' in program_config:
                models = program_config.get('models', {})
                default_model = models.get('default')
                
                # Map model name to model ID
                if default_model == "Claude" and not use_nova and model_selection != 'nova':
                    model_id = self.default_model_id
                elif default_model == "Nova" or use_nova or model_selection == 'nova':
                    model_id = self.default_nova_model_id
            
            # Invoke model to generate final response
            if use_nova or model_selection == 'nova':
                if bedrock_service:
                    final_response = bedrock_service.invoke_nova_model(formatted_prompt)
                else:
                    # Direct invocation if bedrock_service is not provided
                    response = self._make_request_with_retry(
                        self.bedrock_runtime.invoke_model,
                        modelId=model_id,
                        contentType='application/json',
                        accept='application/json',
                        body=json.dumps({
                            "prompt": formatted_prompt,
                            "max_tokens": 4000,
                            "temperature": temperature,
                            "top_p": 0.9
                        })
                    )
                    
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    final_response = response_body.get('completion', '')
            else:
                if bedrock_service:
                    final_response = bedrock_service.invoke_model_simple(formatted_prompt)
                else:
                    # Direct invocation if bedrock_service is not provided
                    response = self._make_request_with_retry(
                        self.bedrock_runtime.invoke_model,
                        modelId=model_id,
                        contentType='application/json',
                        accept='application/json',
                        body=json.dumps({
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 4000,
                            "temperature": temperature,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": formatted_prompt
                                }
                            ]
                        })
                    )
                    
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    final_response = response_body.get('content', [{}])[0].get('text', '')
            
            # Ensure response is valid JSON
            try:
                # First check if response is already valid JSON
                json.loads(final_response)
                return final_response
            except json.JSONDecodeError:
                # If not, try to extract valid JSON
                json_start = final_response.find('{')
                json_end = final_response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = final_response[json_start:json_end]
                    try:
                        json.loads(json_text)  # Validate it's proper JSON
                        return json_text
                    except json.JSONDecodeError:
                        # Clean JSON and try again
                        cleaned_json = re.sub(r'\s*#.*, '', json_text, flags=re.MULTILINE)
                        cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
                        
                        try:
                            json.loads(cleaned_json)
                            return cleaned_json
                        except json.JSONDecodeError:
                            # If all cleaning fails, create a minimal valid response
                            return json.dumps({
                                "answer": final_response,
                                "chart_data": []
                            })
                else:
                    # If no JSON structure found, create a valid one
                    return json.dumps({
                        "answer": final_response,
                        "chart_data": []
                    })
            
        except Exception as e:
            self.logger.error(f"Error generating final response: {str(e)}\n{traceback.format_exc()}")
            return json.dumps({
                "answer": f"I processed the data but encountered an error generating the final response: {str(e)}",
                "chart_data": []
            })

    def _enrich_user_query(self, user_query: str, bedrock_service, s3_service, program_config: dict) -> str:
        """
        Enrich user query with mapping codes.
        
        Args:
            user_query (str): The original user query
            bedrock_service: Bedrock service for model invocation
            s3_service: S3 service for file access
            program_config (dict): Program configuration
            
        Returns:
            str: The enriched query
        """
        mapping_config = program_config.get('mapping_config')
        if not mapping_config:
            return user_query

        try:
            # Initialize mapping information
            mapping_info = []
            
            # Process each mapping file
            for mapping_type, config in mapping_config.items():
                try:
                    is_enabled = config.get('enabled')
                    if is_enabled == "Y":
                        s3_uri = config.get('s3_location_uri')
                        if not s3_uri:
                            continue
                        
                        # Read file content using S3 service
                        response = s3_service.get_document_content(s3_uri)
                        if not response:
                            self.logger.error(f"No content found for file: {s3_uri}")
                            continue
                        content = response.decode('utf-8')

                        # Parse CSV content
                        csv_reader = csv.reader(StringIO(content), delimiter='|')
                        
                        # Add description from config
                        mapping_info.append(f"\n{config.get('description', mapping_type)}:")
                        
                        # Process mappings
                        for row in csv_reader:
                            if len(row) >= 2:
                                mapping_info.append(f"{row[0]}: {row[1]}")

                except Exception as e:
                    self.logger.error(f"Error processing mapping file {mapping_type}: {str(e)}")
                    continue

            # Get query enrichment parameters from program config
            query_config = program_config.get('query_enrichment', {})
            model_selection = query_config.get('model', 'nova')  # Default to Nova for query enrichment
            temperature = query_config.get('temperature', 0.0)  # Low temperature for deterministic results
            
            # Determine model ID based on configuration
            if model_selection == 'nova':
                model_id = self.default_nova_model_id
            else:
                model_id = self.default_model_id
                
            # Override with program-specific model if available
            if 'models' in program_config:
                models = program_config.get('models', {})
                default_model = models.get('default')
                
                # Map model name to model ID
                if default_model == "Claude" and model_selection != 'nova':
                    model_id = self.default_model_id
                elif default_model == "Nova" or model_selection == 'nova':
                    model_id = self.default_nova_model_id
                    
            # Get program name for context
            program_name = program_config.get('name', 'Program')
            
            # Build prompt with mapping information
            prompt = f"""
            User Question: {user_query}
            
            Background: Before querying the knowledgebase with actual user query about {program_name}, I want to enrich the user query with the below mapping codes and use the enriched user query to query knowledgebase for fetching most relevant data.
            Task: Use the above user query, interpret it with the below mapping codes and provide an enriched user query only with the most relevant mapping codes along with the description.
            
            Mapping Information:
            {mapping_info}
            """

            self.logger.info(f"Enrichment prompt: {prompt}")
            
            # Invoke model based on configuration
            if model_selection == 'claude' and bedrock_service:
                enriched_query = bedrock_service.invoke_model_simple(prompt)
            elif bedrock_service:
                enriched_query = bedrock_service.invoke_nova_model(prompt)
            else:
                # Direct invocation if bedrock_service is not provided
                if model_selection == 'claude':
                    response = self._make_request_with_retry(
                        self.bedrock_runtime.invoke_model,
                        modelId=model_id,
                        contentType='application/json',
                        accept='application/json',
                        body=json.dumps({
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 1000,
                            "temperature": temperature,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
                        })
                    )
                    
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    enriched_query = response_body.get('content', [{}])[0].get('text', '')
                else:
                    response = self._make_request_with_retry(
                        self.bedrock_runtime.invoke_model,
                        modelId=model_id,
                        contentType='application/json',
                        accept='application/json',
                        body=json.dumps({
                            "prompt": prompt,
                            "max_tokens": 1000,
                            "temperature": temperature,
                            "top_p": 0.9
                        })
                    )
                    
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    enriched_query = response_body.get('completion', '')
            
            self.logger.info(f"Enriched query: {enriched_query}")
            return enriched_query

        except Exception as e:
            self.logger.error(f"Error enriching query: {str(e)}")
            return user_query

    def _build_conversation_context(self, chat_history: List[Dict], program_config: Dict) -> str:
        """
        Build conversation context from chat history.
        
        Args:
            chat_history (List[Dict]): Chat history
            program_config (Dict): Program configuration
            
        Returns:
            str: Formatted conversation context
        """
        if not chat_history or len(chat_history) < 2:
            return ""
        
        # Get conversation history parameters from program config
        history_config = program_config.get('conversation_history', {})
        max_history = history_config.get('max_messages', 10)  # Default to 10 messages
        include_charts = history_config.get('include_charts', False)  # Whether to include chart data in history
            
        previous_messages = chat_history[-max_history:]  # Get up to max_history messages
        context_parts = []
        
        for i in range(0, len(previous_messages), 2):
            if i + 1 < len(previous_messages):
                user_msg = previous_messages[i]
                assistant_msg = previous_messages[i + 1]
                if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                    context_parts.append(f"Previous Question: {user_msg['content']}")
                    
                    # Try to parse assistant message as JSON
                    try:
                        assistant_content = json.loads(assistant_msg['content'])
                        
                        # Always include the answer
                        if "answer" in assistant_content:
                            context_parts.append(f"Previous Answer: {assistant_content['answer']}")
                            
                            # Optionally include chart data summary
                            if include_charts and "chart_data" in assistant_content and assistant_content["chart_data"]:
                                chart_data = assistant_content["chart_data"]
                                chart_summary = []
                                
                                for idx, chart in enumerate(chart_data):
                                    if "header" in chart and "rows" in chart:
                                        num_rows = len(chart["rows"])
                                        chart_summary.append(f"Chart {idx+1}: {len(chart['header'])} columns, {num_rows} rows")
                                
                                if chart_summary:
                                    context_parts.append(f"Previous Charts: {', '.join(chart_summary)}")
                        else:
                            context_parts.append(f"Previous Answer: {assistant_msg['content']}")
                    except json.JSONDecodeError:
                        context_parts.append(f"Previous Answer: {assistant_msg['content']}")
        
        return "\n\n".join(context_parts)

    # Alias the search_ccar_documents_latest method to search_documents_latest for backward compatibility
    def search_ccar_documents_latest(self, query: str, bedrock_service, s3_service, prompt_template_file: str, 
                                    program_config: dict, chat_history: Optional[List[Dict]] = None, 
                                    use_nova: bool = False) -> str:
        """
        Alias for search_documents_latest for backward compatibility.
        """
        return self.search_documents_latest(
            query=query,
            bedrock_service=bedrock_service,
            s3_service=s3_service,
            prompt_template_file=prompt_template_file,
            program_config=program_config,
            chat_history=chat_history,
            use_nova=use_nova,
            program_id="CCAR"  # Default to CCAR for backward compatibility
        )
    '''''
    def search_ccar_documents_new(self, query: str, bedrock_service, s3_service, prompt_template_file: str, program_config: dict, chat_history: Optional[List[Dict]] = None, use_nova: bool = False) -> str:
        """
        Search documents specifically for CCAR queries using specialized prompt template.
        
        Args:
            query (str): User query about CCAR documents
            prompt_template_file (str): Path to the CCAR prompt template file
            chat_history (Optional[List[Dict]]): Previous conversation history
            use_nova (bool): Whether to use Nova model instead of Claude
            
        Returns:
            str: JSON-formatted response with answer and optional chart data
        """
        try:
            self.logger.info(f"Original User Query: {query}")
            # Only enrich query if mapping_config exists in program_config
            if program_config and program_config.get('mapping_config'):
                enriched_query = self._enrich_user_query(query, bedrock_service, s3_service, program_config)
                self.logger.info(f"Searching documents with Enriched User Query: {enriched_query}")
            else:
                enriched_query = query
                self.logger.info("No mapping configuration found, using original query")
            
            # Use program-specific knowledge base and data source if provided
            knowledge_base_id = program_config.get('knowledge_base_id', self.knowledge_base_id)
            data_source_id = program_config.get('data_source_id', self.data_source_id)
            
            conversation_context = self._build_conversation_context(chat_history)
            
            # Load and format CCAR prompt template
            with open(prompt_template_file, 'r') as file:
                prompt_template = file.read()
            
            # Format the prompt template with user query and conversation context
            structured_query = prompt_template.replace("$user_query$", enriched_query)
            
            # Add conversation context if available
            if conversation_context:
                structured_query += f"\n\nPrevious Conversation Context:\n{conversation_context}"
            
            # Configure vector search
            vector_search_config = {
                "numberOfResults": self.max_results,
                "filter": {
                    "equals": {
                        "key": "x-amz-bedrock-kb-data-source-id",
                        "value": data_source_id
                    }
                }
            }

            # Configure model
            if use_nova:
                model_id = self.config.get('model_config', {}).get('nova_model_id', 'amazon.nova-pro-v1:0')
                model_arn = f"arn:aws:bedrock:us-east-1::foundation-model/{model_id}"
            else:
                model_id = self.config.get('model_config', {}).get('model_id', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
                model_arn = model_id

            print(f"Using model: {model_arn}")
            print(f"Using knowledge base ID: {knowledge_base_id}")
            print(f"Using data source ID: {data_source_id}")

            self._wait_for_rate_limit()

            try:
                if use_nova:
                    retrieve_config = {
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseConfiguration": {
                            "knowledgeBaseId": knowledge_base_id,
                            "modelArn": model_arn,
                            "retrievalConfiguration": {
                                "vectorSearchConfiguration": vector_search_config
                            }
                        }
                    }
                    print(f"Structured query for Nova:\n{structured_query}")
                    print(f"Nova Request:{json.dumps(retrieve_config, indent=4)}")
                    response = self._make_request_with_retry(
                        self.bedrock_kb.retrieve_and_generate,
                        input={"text": structured_query},
                        retrieveAndGenerateConfiguration=retrieve_config
                    )
                else:
                    base_config = {
                        "input": {
                            "text": structured_query
                        },
                        "retrieveAndGenerateConfiguration": {
                            "type": "KNOWLEDGE_BASE",
                            "knowledgeBaseConfiguration": {
                                "knowledgeBaseId": knowledge_base_id,
                                "modelArn": model_arn,
                                "retrievalConfiguration": {
                                    "vectorSearchConfiguration": vector_search_config
                                },
                                "generationConfiguration": {
                                    "promptTemplate": {
                                        "textPromptTemplate": structured_query
                                    },
                                    "inferenceConfig": {
                                        "textInferenceConfig": {
                                            "temperature": 0.7,
                                            "topP": 0.9,
                                            "maxTokens": 4000,
                                            "stopSequences": []
                                        }
                                    }
                                }
                            }
                        }
                    }
                    print(f"Structured query for Claude:\n{structured_query}")
                    print(f"Claude Request:{json.dumps(base_config, indent=4)}")
                    response = self._make_request_with_retry(
                        self.bedrock_kb.retrieve_and_generate,
                        **base_config
                    )

                print(f"{'Claude' if not use_nova else 'Nova'} Model Response: {json.dumps(response, indent=4)}")

                if not response:
                    return json.dumps({
                        "answer": "I apologize, but I couldn't generate a response. Please try again.",
                        "chart_data": []
                    })

                completion = response.get('output', {}).get('text', '')
                print(f"completion:\n{completion}\n")
                citations_data = response.get('citations', [])

                # Process response based on model
                if use_nova:
                    try:
                        structured_response = json.loads(completion)
                        print(f"Structured response from Nova:\n{structured_response}\n")
                        citations = self._format_citations(citations_data)
                        print(f"Citations from Nova:\n{citations}\n")
                        
                        if citations:
                            structured_response['answer'] = structured_response.get('answer', '') + "\n\n" + citations
                        else:
                            structured_response['answer'] = structured_response.get('answer', '')
                        
                        if 'chart_data' in structured_response:
                            structured_response['chart_data'] = self._clean_chart_data(structured_response['chart_data'])
                            
                            if not structured_response['chart_data']:
                                structured_response.pop('chart_data', None)
                                structured_response.pop('chart_type', None)
                                structured_response.pop('chart_attributes', None)

                        return json.dumps(structured_response)

                    except json.JSONDecodeError as e:
                        self.logger.error(f"Nova JSON parsing error: {str(e)}")
                        citations = self._format_citations(citations_data)
                        basic_response = completion + "\n\n" + citations if citations else completion
                        return json.dumps({"answer": basic_response})
                else:
                    try:
                        # Clean up the completion string and handle control characters
                        completion = completion.strip()
                        
                        # Clean special characters and escape sequences
                        #completion = completion.encode('ascii', 'ignore').decode('ascii')
                        #completion = ''.join(char for char in completion if ord(char) >= 32)
                        
                        if not completion.startswith('{'):
                            json_start = completion.find('{')
                            json_end = completion.rfind('}') + 1
                            if json_start >= 0 and json_end > json_start:
                                completion = completion[json_start:json_end]
                            else:
                                return json.dumps({
                                    "answer": "I apologize, but I couldn't parse the response. Please try again.",
                                    "chart_data": []
                                })
                        
                        try:
                            # First attempt to parse the JSON as is
                            structured_response = json.loads(completion)
                        except json.JSONDecodeError:
                            # If that fails, try to clean up common JSON formatting issues
                            # Remove any comments (e.g., # Choose from available chart types)
                            completion = re.sub(r'\s*#.*$', '', completion, flags=re.MULTILINE)
                            # Fix any trailing commas in arrays or objects
                            completion = re.sub(r',(\s*[}\]])', r'\1', completion)
                            # Try parsing again
                            structured_response = json.loads(completion)
                        
                        # Format and add citations
                        citations = self._format_citations(citations_data)
                        if citations:
                            structured_response['answer'] = structured_response.get('answer', '') + "\n\n" + citations
                        else:
                            structured_response['answer'] = structured_response.get('answer', '')
                        
                        # Handle chart data if present
                        if structured_response.get('chart_data'):
                            structured_response['chart_data'] = self._clean_chart_data(structured_response['chart_data'])
                            
                            if not structured_response['chart_data']:
                                structured_response.pop('chart_data', None)
                                structured_response.pop('chart_type', None)
                                structured_response.pop('chart_attributes', None)

                        # Final validation of the response
                        required_fields = ['answer']
                        for field in required_fields:
                            if field not in structured_response:
                                structured_response[field] = "No valid response could be generated."
                        
                        return json.dumps(structured_response, ensure_ascii=False)

                    except Exception as e:
                        self.logger.error(f"Claude response processing error: {str(e)}")
                        return json.dumps({
                            "answer": completion,
                            "chart_data": []
                        })

            except Exception as e:
                self.logger.error(f"Error making request: {str(e)}")
                return json.dumps({
                    "answer": f"I encountered an error while processing your request: {str(e)}",
                    "chart_data": []
                })

        except Exception as e:
            self.logger.error(f"Error during CCAR document search: {str(e)}")
            raise
    '''
    def search_ccar_documents_new(self, query: str, bedrock_service, s3_service, prompt_template_file: str, 
                            program_config: dict, chat_history: Optional[List[Dict]] = None, 
                            use_nova: bool = False) -> str:
        """
        Enhanced document search with separate retrieval and generation steps.
        """
        try:
            self.logger.info(f"Original User Query: {query}")
            
            # Enrich query if mapping config exists
            if program_config and program_config.get('mapping_config'):
                enriched_query = self._enrich_user_query(query, bedrock_service, s3_service, program_config)
                self.logger.info(f"Searching documents with Enriched User Query: {enriched_query}")
            else:
                enriched_query = query
                self.logger.info("No mapping configuration found, using original query")
            
            # Use program-specific knowledge base and data source if provided
            knowledge_base_id = program_config.get('knowledge_base_id', self.knowledge_base_id)
            data_source_id = program_config.get('data_source_id', self.data_source_id)
            
            # First step: Retrieve relevant chunks from knowledge base
            chunks = self._retrieve_chunks(enriched_query, knowledge_base_id, data_source_id)
            print(f"chunks:\n{chunks}")
            if not chunks:
                return json.dumps({
                    "answer": "I couldn't find relevant information to answer your question.",
                    "chart_data": []
                })
                
             # Load prompt template
            with open(prompt_template_file, 'r') as file:
                prompt_template = file.read()
            
            # Format prompt with chunks and query
            formatted_prompt = self._format_initial_prompt(chunks, enriched_query, prompt_template, chat_history)
            #formatted_prompt = self._load_prompt_template(prompt_template_file, enriched_query, chunks, chat_history)
            
            print(f"formatted_prompt:\n{formatted_prompt}")
            # Invoke model with retry logic
            completion = self._invoke_model_with_retry(bedrock_service, formatted_prompt, use_nova)
            
            if completion is None:
                return json.dumps({
                    "answer": "I apologize, but I'm currently experiencing high traffic. Please try your request again in a few moments.",
                    "chart_data": []
                })

            # Process the completion into structured response
            structured_response = self._process_completion(completion, chunks)
            return structured_response

        except Exception as e:
            self.logger.error(f"Error during document search: {str(e)}\n{traceback.format_exc()}")
            return json.dumps({
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "chart_data": []
            })

    def _retrieve_chunks(self, query: str, knowledge_base_id: str, data_source_id: str) -> List[Dict]:
        """
        Retrieve relevant chunks from the knowledge base.
        """
        try:
            vector_search_config = {
                "numberOfResults": 20,
                "overrideSearchType": "HYBRID",
                "filter": {
                    "equals": {
                        "key": "x-amz-bedrock-kb-data-source-id",
                        "value": data_source_id
                    }
                }
            }

            # Corrected API parameters
            response = self._make_request_with_retry(
                self.bedrock_kb.retrieve,
                knowledgeBaseId=knowledge_base_id,
                retrievalQuery={"text": query},  # Correct parameter structure
                retrievalConfiguration={
                    "vectorSearchConfiguration": vector_search_config  # Moved to retrievalConfiguration
                }
            )
            #print(f"response:\n{response}")
            chunks = []
            if response and 'retrievalResults' in response:
                for result in response['retrievalResults']:
                    chunk = {
                        'content': result.get('content', {}).get('text', ''),
                        'location': result.get('location', {}).get('s3Location', {}).get('uri', ''),
                        'metadata': result.get('metadata', {}),
                        'score': result.get('score', 0)
                    }
                    chunks.append(chunk)
            #print(f"chunks:\n{chunks}")
            
            # Sort chunks by relevance score
            chunks.sort(key=lambda x: x['score'], reverse=True)

            # Log retrieval metrics
            self.logger.info(f"Retrieved {len(chunks)} chunks with scores from "
                           f"{chunks[0]['score']:.3f} to {chunks[-1]['score']:.3f}")
            
            return chunks

        except Exception as e:
            self.logger.error(f"Error retrieving chunks: {str(e)}")
            raise

    def _format_chunks_for_generation(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks for use in generation prompt.
        """
        formatted_chunks = []
        for chunk in chunks:
            source = chunk['location'].split('/')[-1] if chunk['location'] else 'Unknown'
            page = chunk['metadata'].get('x-amz-bedrock-kb-document-page-number', 'N/A')
            formatted_chunks.append(
                f"""Source: {source}
                Page: {page}
                Content: {chunk['content']}
                Relevance Score: {chunk['score']}
                ---"""
            )
        return "\n\n".join(formatted_chunks)

    def _load_prompt_template(self, template_file: str, query: str, context: str, 
                            chat_history: Optional[List[Dict]] = None) -> str:
        """
        Load and format the prompt template with context and query.
        """
        try:
            with open(template_file, 'r') as file:
                template = file.read()
            
            # Build conversation context if available
            conversation_context = self._build_conversation_context(chat_history) if chat_history else ""
            
            # Replace placeholders in template
            formatted_template = template.replace("$user_query$", query)
            formatted_template = formatted_template.replace("$context$", context)
            
            if conversation_context:
                formatted_template += f"\n\nPrevious Conversation Context:\n{conversation_context}"
            
            return formatted_template

        except Exception as e:
            self.logger.error(f"Error loading prompt template: {str(e)}")
            raise

    def _process_completion(self, completion: str, chunks: List[Dict]) -> str:
        """
        Process and format the completion with citations.
        """
        try:
            completion = completion.strip()
            
            # Handle non-JSON responses
            if not completion.startswith('{'):
                json_start = completion.find('{')
                json_end = completion.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    completion = completion[json_start:json_end]
                else:
                    return json.dumps({
                        "answer": completion,
                        "chart_data": []
                    })
            
            # Parse JSON response
            try:
                structured_response = json.loads(completion)
            except json.JSONDecodeError:
                print(f"Inside JSONDecodeError:\n{completion}")
                # Clean up JSON formatting issues
                completion = re.sub(r'\s*#.*$', '', completion, flags=re.MULTILINE)
                completion = re.sub(r',(\s*[}\]])', r'\1', completion)
                structured_response = json.loads(completion)
            
            # Add citations
            #citations = self._format_citations_from_chunks(chunks)
            #if citations:
            #    structured_response['answer'] = structured_response.get('answer', '') + "\n\n" + citations
            structured_response['answer'] = structured_response.get('answer', '')
            # Process chart data if present
            if structured_response.get('chart_data'):
                structured_response['chart_data'] = self._clean_chart_data(structured_response['chart_data'])
                if not structured_response['chart_data']:
                    structured_response.pop('chart_data', None)
                    structured_response.pop('chart_type', None)
                    structured_response.pop('chart_attributes', None)
                    
            print(f"structured_response:\n{json.dumps(structured_response, ensure_ascii=False)}")
            return json.dumps(structured_response, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Error processing completion: {str(e)}")
            return json.dumps({
                "answer": completion,
                "chart_data": []
            })

    def _format_citations_from_chunks(self, chunks: List[Dict]) -> str:
        """
        Format citations from retrieved chunks.
        """
        doc_pages = {}
        
        for chunk in chunks:
            doc_uri = chunk['location']
            page_num = chunk['metadata'].get('x-amz-bedrock-kb-document-page-number', 'N/A')
            
            if doc_uri:
                doc_name = doc_uri.split('/')[-1]
                if isinstance(page_num, float):
                    page_num = int(page_num)
                
                if doc_name not in doc_pages:
                    doc_pages[doc_name] = set()
                doc_pages[doc_name].add(str(page_num))
        
        if doc_pages:
            citations = []
            for doc_name, pages in doc_pages.items():
                sorted_pages = sorted(pages, key=lambda x: int(x) if x.isdigit() else float('inf'))
                citations.append(f"{doc_name} (Page {', '.join(sorted_pages)})")
            
            return "**Sources:** " + ", ".join(citations)
        
        return ""
    
    def _format_initial_prompt(self, chunks: List[Dict], query: str, prompt_template: str, chat_history: Optional[List[Dict]] = None) -> str:
        """Insert raw chunks into the prompt template."""
        try:
            # Format the chunks section exactly as shown in document_content
            search_results = f"chunks:\n{chunks}"
            
            # Replace placeholders in template
            formatted_prompt = prompt_template.replace("$context$", search_results)
            formatted_prompt = formatted_prompt.replace("$user_query$", query)
            
            # Add conversation context if available
            conversation_context = self._build_conversation_context(chat_history) if chat_history else ""
            if conversation_context:
                formatted_prompt += f"\n\nPrevious Conversation Context:\n{conversation_context}"
            
            return formatted_prompt
                
        except Exception as e:
            self.logger.error(f"Error formatting prompt: {str(e)}")
            raise

    def _format_initial_prompt_old(self, chunks: List[Dict], query: str, prompt_template: str, chat_history: Optional[List[Dict]] = None) -> str:
        """Format chunks and query into initial prompt template."""
        try:
            # First, extract and organize relevant information from chunks
            organized_data = []
            for chunk in chunks:
                lines = chunk['content'].split('\r')
                for line in lines:
                    if line.strip():  # Skip empty lines
                        organized_data.append(line.strip())

            # Format context from chunks
            context = "\n".join(organized_data)
            
            # Replace placeholders in template
            formatted_prompt = prompt_template.replace("$user_query$", query)
            formatted_prompt = formatted_prompt.replace("$context$", context)
            
            # Build conversation context if available
            conversation_context = self._build_conversation_context(chat_history) if chat_history else ""
            if conversation_context:
                formatted_prompt += f"\n\nPrevious Conversation Context:\n{conversation_context}"
            
            return formatted_prompt
            
        except Exception as e:
            self.logger.error(f"Error formatting prompt: {str(e)}")
            raise

    def _invoke_model_with_retry(self, bedrock_service, prompt: str, use_nova, max_retries: int = 3, base_delay: float = 2.0) -> Optional[str]:
        """Invoke model with retry logic for throttling."""
        for attempt in range(max_retries):
            try:
                #response = bedrock_service.invoke_model_simple(prompt)
                if use_nova:
                    response = bedrock_service.invoke_nova_model(prompt)
                else:
                    response = bedrock_service.invoke_model_simple(prompt)
                if response:
                    return response
                    
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    self.logger.error(f"Max retries reached for model invocation: {str(e)}")
                    return None
                    
                # Calculate delay with exponential backoff and jitter
                delay = (base_delay ** attempt) + random.uniform(0.1, 1.0)
                self.logger.warning(f"Model invocation failed. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
        
        return None

    def _enrich_user_query(self, user_query: str, bedrock_service, s3_service, program_config: dict) -> str:
        mapping_config = program_config.get('mapping_config')
        if not mapping_config:
            return user_query

        try:
            # Initialize mapping information
            mapping_info = []
            
            # Process each mapping file
            for mapping_type, config in mapping_config.items():
                print(f"mapping_type:{mapping_type}\nconfig:{config}")
                try:
                    is_enabled = config.get('enabled')
                    if is_enabled == "Y":
                        s3_uri = config.get('s3_location_uri')
                        if not s3_uri:
                            continue
                        
                        # Read file content using boto3
                        response = s3_service.get_document_content(s3_uri)
                        if not response:
                            self.logger.error(f"No content found for file: {s3_uri}")
                            continue
                        content = response.decode('utf-8')

                        # Parse CSV content
                        csv_reader = csv.reader(StringIO(content), delimiter='|')
                        
                        # Add description from config
                        mapping_info.append(f"\n{config.get('description', mapping_type)}:")
                        
                        # Process mappings
                        for row in csv_reader:
                            if len(row) >= 2:
                                mapping_info.append(f"{row[0]}: {row[1]}")

                except Exception as e:
                    self.logger.error(f"Error processing mapping file {mapping_type}: {str(e)}")
                    continue

            # Build prompt with mapping information
            prompt = f"""
            User Question: {user_query}
            
            Background: Before querying the knowledgebase with actual user query, I want to enrich the user query with the below mapping codes and use the enriched user query to query knowledgebase for fetching most relevant data.
            Task: Use the above user query, interpret it with the below mapping codes and provide an enriched user query only with the most relevant mapping codes along with the description.
            
            Mapping Information:
            {mapping_info}
            """

            print(f"prompt for enrich query:\n{prompt}")
            return bedrock_service.invoke_nova_model(prompt)

        except Exception as e:
            self.logger.error(f"Error enriching query: {str(e)}")
            return user_query
    
    def search_ccar_documents(self, query: str, ccar_prompt_file: str, chat_history: Optional[List[Dict]] = None, use_nova: bool = False) -> str:
        """
        Search documents specifically for CCAR queries using specialized prompt template.
        
        Args:
            query (str): User query about CCAR documents
            ccar_prompt_file (str): Path to the CCAR prompt template file
            chat_history (Optional[List[Dict]]): Previous conversation history
            use_nova (bool): Whether to use Nova model instead of Claude
            
        Returns:
            str: JSON-formatted response with answer and optional chart data
        """
        try:
            self.logger.info(f"Searching CCAR documents with query: {query}")
            
            conversation_context = self._build_conversation_context(chat_history)
            
            # Load and format CCAR prompt template
            with open(ccar_prompt_file, 'r') as file:
                prompt_template = file.read()
            
            # Format the prompt template with user query and conversation context
            structured_query = prompt_template.replace("$user_query$", query)
            
            # Add conversation context if available
            if conversation_context:
                structured_query += f"\n\nPrevious Conversation Context:\n{conversation_context}"
            
            # Configure vector search
            vector_search_config = {
                "numberOfResults": self.max_results,
                "filter": {
                    "equals": {
                        "key": "x-amz-bedrock-kb-data-source-id",
                        "value": self.ccar_data_source_id
                    }
                }
            }

            # Configure model
            if use_nova:
                model_id = self.config.get('model_config', {}).get('nova_model_id', 'amazon.nova-pro-v1:0')
                model_arn = f"arn:aws:bedrock:us-east-1::foundation-model/{model_id}"
            else:
                model_id = self.config.get('model_config', {}).get('model_id', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
                model_arn = model_id

            print(f"Using model: {model_arn}")
            print(f"Using knowledge base ID: {self.ccar_knowledge_base_id}")
            print(f"Using data source ID: {self.ccar_data_source_id}")

            self._wait_for_rate_limit()

            try:
                if use_nova:
                    retrieve_config = {
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseConfiguration": {
                            "knowledgeBaseId": self.ccar_knowledge_base_id,
                            "modelArn": model_arn,
                            "retrievalConfiguration": {
                                "vectorSearchConfiguration": vector_search_config
                            }
                        }
                    }
                    print(f"Structured query for Nova:\n{structured_query}")
                    print(f"Nova Request:{json.dumps(retrieve_config, indent=4)}")
                    response = self._make_request_with_retry(
                        self.bedrock_kb.retrieve_and_generate,
                        input={"text": structured_query},
                        retrieveAndGenerateConfiguration=retrieve_config
                    )
                else:
                    base_config = {
                        "input": {
                            "text": structured_query
                        },
                        "retrieveAndGenerateConfiguration": {
                            "type": "KNOWLEDGE_BASE",
                            "knowledgeBaseConfiguration": {
                                "knowledgeBaseId": self.ccar_knowledge_base_id,
                                "modelArn": model_arn,
                                "retrievalConfiguration": {
                                    "vectorSearchConfiguration": vector_search_config
                                },
                                "generationConfiguration": {
                                    "promptTemplate": {
                                        "textPromptTemplate": structured_query
                                    },
                                    "inferenceConfig": {
                                        "textInferenceConfig": {
                                            "temperature": 0.7,
                                            "topP": 0.9,
                                            "maxTokens": 4000,
                                            "stopSequences": []
                                        }
                                    }
                                }
                            }
                        }
                    }
                    print(f"Structured query for Claude:\n{structured_query}")
                    print(f"Claude Request:{json.dumps(base_config, indent=4)}")
                    response = self._make_request_with_retry(
                        self.bedrock_kb.retrieve_and_generate,
                        **base_config
                    )

                print(f"{'Claude' if not use_nova else 'Nova'} Model Response: {json.dumps(response, indent=4)}")

                if not response:
                    return json.dumps({
                        "answer": "I apologize, but I couldn't generate a response. Please try again.",
                        "chart_data": []
                    })

                completion = response.get('output', {}).get('text', '')
                citations_data = response.get('citations', [])

                # Process response based on model
                if use_nova:
                    try:
                        structured_response = json.loads(completion)
                        citations = self._format_citations(citations_data)
                        
                        if citations:
                            structured_response['answer'] = structured_response.get('answer', '') + "\n\n" + citations
                        
                        if 'chart_data' in structured_response:
                            structured_response['chart_data'] = self._clean_chart_data(structured_response['chart_data'])
                            
                            if not structured_response['chart_data']:
                                structured_response.pop('chart_data', None)
                                structured_response.pop('chart_type', None)
                                structured_response.pop('chart_attributes', None)

                        return json.dumps(structured_response)

                    except json.JSONDecodeError as e:
                        self.logger.error(f"Nova JSON parsing error: {str(e)}")
                        citations = self._format_citations(citations_data)
                        basic_response = completion + "\n\n" + citations if citations else completion
                        return json.dumps({"answer": basic_response})
                else:
                    try:
                        # Clean up the completion string and handle control characters
                        completion = completion.strip()
                        
                        # Clean special characters and escape sequences
                        completion = completion.encode('ascii', 'ignore').decode('ascii')
                        completion = ''.join(char for char in completion if ord(char) >= 32)
                        
                        if not completion.startswith('{'):
                            json_start = completion.find('{')
                            json_end = completion.rfind('}') + 1
                            if json_start >= 0 and json_end > json_start:
                                completion = completion[json_start:json_end]
                            else:
                                return json.dumps({
                                    "answer": "I apologize, but I couldn't parse the response. Please try again.",
                                    "chart_data": []
                                })
                        
                        try:
                            # First attempt to parse the JSON as is
                            structured_response = json.loads(completion)
                        except json.JSONDecodeError:
                            # If that fails, try to clean up common JSON formatting issues
                            import re
                            # Remove any comments (e.g., # Choose from available chart types)
                            completion = re.sub(r'\s*#.*$', '', completion, flags=re.MULTILINE)
                            # Fix any trailing commas in arrays or objects
                            completion = re.sub(r',(\s*[}\]])', r'\1', completion)
                            # Try parsing again
                            structured_response = json.loads(completion)
                        
                        # Format and add citations
                        citations = self._format_citations(citations_data)
                        if citations:
                            structured_response['answer'] = structured_response.get('answer', '') + "\n\n" + citations
                        
                        # Handle chart data if present
                        if structured_response.get('chart_data'):
                            structured_response['chart_data'] = self._clean_chart_data(structured_response['chart_data'])
                            
                            if not structured_response['chart_data']:
                                structured_response.pop('chart_data', None)
                                structured_response.pop('chart_type', None)
                                structured_response.pop('chart_attributes', None)

                        # Final validation of the response
                        required_fields = ['answer']
                        for field in required_fields:
                            if field not in structured_response:
                                structured_response[field] = "No valid response could be generated."
                        
                        return json.dumps(structured_response, ensure_ascii=False)

                    except Exception as e:
                        self.logger.error(f"Claude response processing error: {str(e)}")
                        return json.dumps({
                            "answer": completion,
                            "chart_data": []
                        })

            except Exception as e:
                self.logger.error(f"Error making request: {str(e)}")
                return json.dumps({
                    "answer": f"I encountered an error while processing your request: {str(e)}",
                    "chart_data": []
                })

        except Exception as e:
            self.logger.error(f"Error during CCAR document search: {str(e)}")
            raise
    
    def search_documents(self, query: str, document_path: Optional[str] = None, chat_history: Optional[List[Dict]] = None, use_nova: bool = False) -> str:
        try:
            self.logger.info(f"Searching with query: {query}")
            self.logger.info(f"Document path: {document_path}")

            conversation_context = self._build_conversation_context(chat_history)
            print("conversation_context:")
            print(conversation_context)
            
            vector_search_config = {
                "numberOfResults": self.max_results
            }

            if document_path:
                document_uri = (f"s3://{self.config['s3_config']['bucket']}/{document_path}" 
                            if not document_path.startswith('https://') else document_path)
                vector_search_config["filter"] = {
                    "equals": {
                        "key": "x-amz-bedrock-kb-source-uri",
                        "value": document_uri
                    }
                }
                
            # Updated model configuration handling
            if use_nova:
                model_id = self.config.get('model_config', {}).get('nova_model_id', 'amazon.nova-pro-v1:0')
                model_arn = f"arn:aws:bedrock:us-east-1::foundation-model/{model_id}"
            else:
                model_id = self.config.get('model_config', {}).get('model_id', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
                model_arn = model_id  # Claude models expect just the model ID

            print(f"Using model: {model_arn}")

            # Modified prompt template for Claude
            if not use_nova:
                structured_query_template = f"""Here is the context and your task:
                
                context:
                
                $search_results$
                
                Task:
                
                Based on the provided context, please answer the following question:

                Please always structure your response as a JSON object with the following format with no other text:
                {{
                    "answer": "Provide a structured and comprehensive answer including definitions, explanations, specific values, amounts in millions with currency symbols, significant changes, trends, and supporting details where available.",
                    "chart_data": [
                        {{
                            "header": ["Category", "GroupAttribute", "Value"],
                            "rows": [
                                ["Category1", "Group1", NumericValue1],
                                ["Category2", "Group2", NumericValue2]
                            ]
                        }}
                    ],
                    "chart_type": "bar", # Choose from available chart types listed below
                    "chart_attributes": {{
                        "xAxis": "Category",
                        "yAxis": "Value",
                        "color": "GroupAttribute",
                        "title": "Chart Title"
                    }}
                }}

                Do not include any text before or after the JSON object. Just return the JSON object itself.
                
                Available Chart Types:
                1. Comparison Charts:
                - "bar" - Standard vertical bars for comparing values
                - "horizontal_bar" - Horizontal bars, good for long category names
                - "line" - For showing trends over time
                - "area" - For showing cumulative values or proportions

                2. Distribution Charts:
                - "pie" - For showing parts of a whole (best with 2-8 categories)
                - "donut" - Similar to pie but with a hole in the center
                - "treemap" - Hierarchical data as nested rectangles
                - "funnel" - For showing sequential process or stages

                3. Relationship Charts:
                - "scatter" - For showing correlations between values
                - "bubble" - Scatter plot with size dimension
            
                Format Requirements: Try to give the values in millions
                1. Data Structure:
                - First column: Main categories for comparison (e.g., Business Segment, Product, Region)
                - Second column: Grouping attribute for color differentiation (e.g., Year, Quarter, Type, Category)
                - Third column: Numeric values (no formatting, just numbers)

                2. Chart Type Selection:
                - Use "pie" or "donut" when showing parts of a whole
                - Use "horizontal_bar" for long category names
                - Use "line" for time series data
                - Use "treemap" for hierarchical data
                - Use "funnel" for sequential processes
                - Use "bubble" or "scatter" for relationships
                - Use "bar" for general comparisons
            
                2. For comparisons:
                - Use consistent naming in the grouping column
                - Sort data logically (chronologically for time, alphabetically for categories)
                - Include same categories for each group for proper comparison

                3. Numeric Values:
                - Use plain numbers without formatting (no currency symbols or special characters) in chart data
                - Convert to millions if dealing with large numbers
                - Round to 0-2 decimal places

                Example formats: Please note the below is just an example. Please include attributes that are more appropirate. If user did not ask for period/year/time based data, do not include the related attributes in the chart.
                Time-based comparison:
                "header": ["Business Segment", "Year", "Revenue"],
                "rows": [
                    ["Segment A", "H124", 5631],
                    ["Segment A", "H123", 5733],
                    ["Segment B", "H124", 486],
                    ["Segment B", "H123", 563]
                ]

                Category-based comparison:
                "header": ["Product", "Region", "Sales"],
                "rows": [
                    ["Product A", "North", 1200],
                    ["Product A", "South", 1400],
                    ["Product B", "North", 800],
                    ["Product B", "South", 900]
                ]

                Original question: {query}
                
                If the data does not support creating a chart, you can omit the chart_data, chart_type, and chart_attributes fields.                
                """
            else:
                # Original template for Nova
                structured_query_template = f"""Please provide your response in the following JSON structure:
                {{
                    "answer": "Provide a structured and comprehensive answer including definitions, explanations, specific values, amounts in millions with currency symbols, significant changes, trends, and supporting details where available.",
                    "chart_data": [
                        {{
                            "header": ["Category", "GroupAttribute", "Value"],
                            "rows": [
                                ["Category1", "Group1", NumericValue1],
                                ["Category1", "Group2", NumericValue2],
                                ["Category2", "Group1", NumericValue3],
                                ["Category2", "Group2", NumericValue4]
                            ]
                        }}
                    ],
                    "chart_type": "bar",  # Choose from available chart types listed below
                    "chart_attributes": {{
                        "xAxis": "Category Label",
                        "yAxis": "Value Label (with unit)",
                        "color": "GroupAttribute",  # Must match the second column header name
                        "title": "Descriptive Chart Title"
                    }}
                }}

                Available Chart Types:
                1. Comparison Charts:
                - "bar" - Standard vertical bars for comparing values
                - "horizontal_bar" - Horizontal bars, good for long category names
                - "line" - For showing trends over time
                - "area" - For showing cumulative values or proportions

                2. Distribution Charts:
                - "pie" - For showing parts of a whole (best with 2-8 categories)
                - "donut" - Similar to pie but with a hole in the center
                - "treemap" - Hierarchical data as nested rectangles
                - "funnel" - For showing sequential process or stages

                3. Relationship Charts:
                - "scatter" - For showing correlations between values
                - "bubble" - Scatter plot with size dimension
            
                Format Requirements: Try to give the values in millions
                1. Data Structure:
                - First column: Main categories for comparison (e.g., Business Segment, Product, Region)
                - Second column: Grouping attribute for color differentiation (e.g., Year, Quarter, Type, Category)
                - Third column: Numeric values (no formatting, just numbers)

                2. Chart Type Selection:
                - Use "pie" or "donut" when showing parts of a whole
                - Use "horizontal_bar" for long category names
                - Use "line" for time series data
                - Use "treemap" for hierarchical data
                - Use "funnel" for sequential processes
                - Use "bubble" or "scatter" for relationships
                - Use "bar" for general comparisons
            
                2. For comparisons:
                - Use consistent naming in the grouping column
                - Sort data logically (chronologically for time, alphabetically for categories)
                - Include same categories for each group for proper comparison

                3. Numeric Values:
                - Use plain numbers without formatting (no currency symbols or special characters) in chart data
                - Convert to millions if dealing with large numbers
                - Round to 0-2 decimal places

                Example formats: Please note the below is just an example. Please include attributes that are more appropirate. If user did not ask for period/year/time based data, do not include the related attributes in the chart.
                Time-based comparison:
                "header": ["Business Segment", "Year", "Revenue"],
                "rows": [
                    ["Segment A", "H124", 5631],
                    ["Segment A", "H123", 5733],
                    ["Segment B", "H124", 486],
                    ["Segment B", "H123", 563]
                ]

                Category-based comparison:
                "header": ["Product", "Region", "Sales"],
                "rows": [
                    ["Product A", "North", 1200],
                    ["Product A", "South", 1400],
                    ["Product B", "North", 800],
                    ["Product B", "South", 900]
                ]

                Original question: {query}
                """

            if conversation_context:
                structured_query_template += f"\nPrevious Conversation Context:\n{conversation_context}"

            if use_nova:
                retrieve_config = {
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.knowledge_base_id,
                        "modelArn": model_arn,
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": vector_search_config
                        }
                    }
                }
            
            self._wait_for_rate_limit()
            print("structured_query_template:")
            print(structured_query_template)

            try:
                if use_nova:
                    response = self._make_request_with_retry(
                        self.bedrock_kb.retrieve_and_generate,
                        input={"text": structured_query_template},
                        retrieveAndGenerateConfiguration=retrieve_config
                    )
                else:
                    # Prepare the common request configuration
                    base_config = {
                        "input": {
                            "text": structured_query_template
                        },
                        "retrieveAndGenerateConfiguration": {
                            "type": "KNOWLEDGE_BASE",
                            "knowledgeBaseConfiguration": {
                                "knowledgeBaseId": self.knowledge_base_id,
                                "modelArn": model_arn,
                                "retrievalConfiguration": {
                                    "vectorSearchConfiguration": {
                                        "numberOfResults": 10
                                    }
                                }
                            }
                        }
                    }

                    # Add Claude-specific configuration
                    base_config["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["generationConfiguration"] = {
                        "promptTemplate": {
                            "textPromptTemplate": structured_query_template
                        },
                        "inferenceConfig": {
                            "textInferenceConfig": {
                                "temperature": 0.7,
                                "topP": 0.9,
                                "maxTokens": 4000,
                                "stopSequences": []
                            }
                        }
                    }

                    # Add document filter if provided
                    if document_uri:
                        filter_config = {
                            "filter": {
                                "equals": {
                                    "key": "x-amz-bedrock-kb-source-uri",
                                    "value": document_uri
                                }
                            }
                        }
                        base_config["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["retrievalConfiguration"]["vectorSearchConfiguration"].update(filter_config)

                    # Make the API call
                    response = self._make_request_with_retry(
                        self.bedrock_kb.retrieve_and_generate,
                        **base_config
                    )
                print(f"{'Claude' if not use_nova else 'Nova'} Model Response:")
                print(json.dumps(response, indent=4))

                if not response:
                    return json.dumps({
                        "answer": "I apologize, but I couldn't generate a response. Please try again.",
                        "chart_data": []
                    })

                completion = response.get('output', {}).get('text', '')
                citations_data = response.get('citations', [])
                
                if use_nova:
                    try:
                        structured_response = json.loads(completion)
                        citations = self._format_citations(citations_data)
                        print(f"citations:{citations}")
                        if citations:
                            structured_response['answer'] = structured_response.get('answer', '') + "\n\n" + citations
                        else:
                            structured_response['answer'] = structured_response.get('answer', '')
                        
                        if 'chart_data' in structured_response:
                            structured_response['chart_data'] = self._clean_chart_data(structured_response['chart_data'])
                            
                            if not structured_response['chart_data']:
                                structured_response.pop('chart_data', None)
                                structured_response.pop('chart_type', None)
                                structured_response.pop('chart_attributes', None)

                        return json.dumps(structured_response)

                    except json.JSONDecodeError:
                        self.logger.error(f"Nova JSON parsing error: {str(e)}")
                        citations = self._format_citations(citations_data)
                        basic_response = completion + "\n\n" + citations if citations else completion
                        return json.dumps({"answer": basic_response})
                else:
                    try:
                        # Clean up the completion string for both models
                        completion = completion.strip()
                        
                        # For Claude model, handle the text prefix
                        if not completion.startswith('{'):
                            # Remove any text before the JSON
                            json_start = completion.find('{')
                            json_end = completion.rfind('}') + 1
                            if json_start >= 0 and json_end > json_start:
                                completion = completion[json_start:json_end]
                            else:
                                # If no JSON found, create a basic JSON response
                                return json.dumps({
                                    "answer": completion,
                                    "chart_data": []
                                })
                        
                        structured_response = json.loads(completion)
                        
                        # Format citations
                        citations = self._format_citations(citations_data)
                        print(f"citations:{citations}")
                        
                        # Add citations to the answer if available
                        if citations:
                            structured_response['answer'] = structured_response.get('answer', '') + "\n\n" + citations
                        
                        # Handle chart data
                        if structured_response.get('chart_data'):
                            structured_response['chart_data'] = self._clean_chart_data(structured_response['chart_data'])
                            
                            if not structured_response['chart_data']:
                                structured_response.pop('chart_data', None)
                                structured_response.pop('chart_type', None)
                                structured_response.pop('chart_attributes', None)

                        return json.dumps(structured_response)

                    except Exception as e:
                        self.logger.error(f"Claude response processing error: {str(e)}")
                        return json.dumps({
                            "answer": completion,
                            "chart_data": []
                        })

            except Exception as e:
                self.logger.error(f"Error making request: {str(e)}")
                return json.dumps({
                    "answer": f"I encountered an error while processing your request: {str(e)}",
                    "chart_data": []
                })

        except Exception as e:
            self.logger.error(f"Error during document search: {str(e)}")
            raise

    def _clean_chart_data(self, chart_data: List[Dict]) -> List[Dict]:
        """Clean and validate chart data."""
        print("Debug - Clean Chart Data Input:", chart_data)
        
        if not isinstance(chart_data, list):
            print("Debug - Chart data is not a list")
            return []

        cleaned_data = []
        for chart in chart_data:
            if not isinstance(chart, dict):
                print("Debug - Chart item is not a dict")
                continue

            header = chart.get('header', [])
            rows = chart.get('rows', [])
            
            print("Debug - Processing chart - Header:", header)
            print("Debug - Processing chart - Rows:", rows)

            if len(header) < 2 or not rows:  # Changed from 3 to 2 minimum columns
                print("Debug - Invalid header or empty rows")
                continue

            # Clean rows and convert numeric values
            cleaned_rows = []
            for row in rows:
                if len(row) != len(header):
                    print(f"Debug - Row length mismatch: {len(row)} vs {len(header)}")
                    continue

                try:
                    cleaned_row = []
                    # First column is always string (category/x-axis)
                    cleaned_row.append(str(row[0]))
                    
                    # Second column is numeric for 2-column data, or string for 3-column data
                    if len(header) == 2:
                        # For 2-column data, second column is numeric
                        numeric_value = str(row[1]).replace(',', '').replace('£', '').replace('$', '').replace('€', '').strip()
                        cleaned_row.append(float(numeric_value))
                    else:
                        # For 3-column data, second column is string (group) and third is numeric
                        cleaned_row.append(str(row[1]))
                        numeric_value = str(row[2]).replace(',', '').replace('£', '').replace('$', '').replace('€', '').strip()
                        cleaned_row.append(float(numeric_value))
                    
                    cleaned_rows.append(cleaned_row)
                    print("Debug - Cleaned row:", cleaned_row)
                    
                except (ValueError, IndexError) as e:
                    print(f"Debug - Error cleaning row: {str(e)}")
                    continue

            if cleaned_rows:
                cleaned_data.append({
                    'header': header,
                    'rows': cleaned_rows
                })
                print("Debug - Added cleaned chart data")

        print("Debug - Final cleaned data:", cleaned_data)
        return cleaned_data

    def _format_citations(self, citations_data: List[Dict]) -> str:
        """Format citations grouped by document name."""
        # Dictionary to store pages by document
        doc_pages = {}
        
        # Process citations and group pages by document
        for citation in citations_data:
            if 'retrievedReferences' in citation:
                for reference in citation['retrievedReferences']:
                    doc_uri = reference.get('location', {}).get('s3Location', {}).get('uri', '')
                    metadata = reference.get('metadata', {})
                    
                    if doc_uri:
                        # Get document name and page number
                        doc_name = doc_uri.split('/')[-1]
                        page_num = metadata.get('x-amz-bedrock-kb-document-page-number', 'N/A')
                        if isinstance(page_num, float):
                            page_num = int(page_num)
                        
                        # Add page to document's page list
                        if doc_name not in doc_pages:
                            doc_pages[doc_name] = set()
                        doc_pages[doc_name].add(str(page_num))
        
        # Format citations
        if doc_pages:
            citations = []
            for doc_name, pages in doc_pages.items():
                # Sort pages numerically
                sorted_pages = sorted(pages, key=lambda x: int(x) if x.isdigit() else float('inf'))
                # Format citation with comma-separated pages
                citations.append(f"{doc_name} (Page {', '.join(sorted_pages)})")
            
            # Join all citations with comma and space
            return "**Sources:** " + ", ".join(citations)
        
        return ""

    def _build_conversation_context(self, chat_history: Optional[List[Dict]]) -> str:
        """Build conversation context from chat history."""
        if not chat_history or len(chat_history) < 2:
            return ""
            
        previous_messages = chat_history[-10:]  # Get up to last 10 messages
        context_parts = []
        
        for i in range(0, len(previous_messages), 2):
            if i + 1 < len(previous_messages):
                user_msg = previous_messages[i]
                assistant_msg = previous_messages[i + 1]
                if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                    context_parts.append(f"Previous Question: {user_msg['content']}")
                    context_parts.append(f"Previous Answer: {assistant_msg['content']}")
        
        return "\n\n".join(context_parts)

    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history = []

    def get_chat_history(self) -> List[Dict]:
        """Get the current chat history."""
        return self.chat_history

    def index_document(self, chunks: List[Dict], s3_path: str, metadata: Dict) -> None:
        """Index document chunks in Knowledge Base."""
        try:
            time.sleep(60)  # Give time for indexing
            self.logger.info(f"Document submitted for indexing: {s3_path}")
            self.logger.info(f"Metadata: {json.dumps(metadata, default=str)}")
        except Exception as e:
            self.logger.error(f"Error indexing document: {str(e)}")
            raise

    def get_sync_status(self) -> Dict:
        """Get the current ingestion status of the data source."""
        try:
            response = self.bedrock_kb.get_knowledge_base(knowledgeBaseId=self.knowledge_base_id)
            return {
                'Status': response['status'],
                'LastUpdatedTime': response.get('lastUpdatedTime'),
                'CreatedTime': response.get('createdTime')
            }
        except Exception as e:
            self.logger.error(f"Error getting sync status: {str(e)}")
            raise