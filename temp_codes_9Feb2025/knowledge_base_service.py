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
            config=boto3_config
        )
        
        self.knowledge_base_id = config.get('knowledge_base_id')
        self.data_source_id = config.get('data_source_id')
        self.ccar_knowledge_base_id = config.get('ccar_knowledge_base_id')
        self.ccar_data_source_id = config.get('ccar_data_source_id')
        self.max_results = config.get('max_results', 10)
        self.model_arn = f"arn:aws:bedrock:us-east-1::foundation-model/{config.get('model_config', {}).get('nova_model_id', 'amazon.nova-pro-v1:0')}"
        self.max_history = 10
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum time between requests in seconds

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
                            "text": query
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
                print(f"{'Claude' if not use_nova else 'Nova'} Model Response: {response}")
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
        """Clean and validate chart data dynamically based on value types."""
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

            if len(header) < 2 or not rows:
                print("Debug - Invalid header or empty rows")
                continue

            # Determine column types based on first non-None row
            column_types = []
            for first_row in rows:
                if all(val is not None for val in first_row):
                    for value in first_row:
                        if isinstance(value, (int, float)) or (
                            isinstance(value, str) and 
                            value.replace('.', '', 1).replace('-', '', 1).replace('e', '', 1).replace('E', '', 1).isdigit()
                        ):
                            column_types.append('numeric')
                        else:
                            column_types.append('string')
                    break
            
            print("Debug - Detected column types:", column_types)

            # Clean rows and convert values based on detected types
            cleaned_rows = []
            for row in rows:
                if len(row) != len(header):
                    print(f"Debug - Row length mismatch: {len(row)} vs {len(header)}")
                    continue

                try:
                    cleaned_row = []
                    # Process each value in the row based on detected type
                    for i, value in enumerate(row):
                        # Handle None values
                        if value is None:
                            cleaned_row.append(None)
                            continue
                            
                        # Convert based on column type
                        if column_types[i] == 'numeric':
                            try:
                                # Clean numeric value and convert to float
                                if isinstance(value, (int, float)):
                                    cleaned_row.append(float(value))
                                else:
                                    numeric_value = str(value).replace(',', '').replace('£', '').replace('$', '').replace('€', '').strip()
                                    cleaned_row.append(float(numeric_value))
                            except ValueError:
                                print(f"Debug - Could not convert value to float: {value}")
                                # If conversion fails, treat as string
                                cleaned_row.append(str(value))
                        else:
                            # String type
                            cleaned_row.append(str(value))
                    
                    cleaned_rows.append(cleaned_row)
                    print("Debug - Cleaned row:", cleaned_row)
                    
                except Exception as e:
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