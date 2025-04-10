import streamlit as st  # Web app framework for Python
import pandas as pd  # Data manipulation and analysis
import os  # Operating system interfaces (file handling)
import json  # JSON data parsing and generation
import time  # Time-related functions
from datetime import datetime  # Date and time manipulation
import plotly.express as px  # High-level plotting library
import plotly.graph_objects as go  # Lower-level plotting library for custom visualizations
import asyncio  # Asynchronous I/O support
import functools  # Higher-order functions and operations on callable objects
import re  # Regular expression operations
import boto3  # AWS SDK for Python
from botocore.client import Config  # Configuration for AWS boto3 clients
from langchain_aws import ChatBedrock  # AWS Bedrock integration for LangChain
from langchain_aws import BedrockEmbeddings  # Embeddings from AWS Bedrock
from datasets import Dataset as HFDataset  # Hugging Face dataset library
from ragas import evaluate  # RAGAS evaluation framework
from ragas.metrics import (  # Specific RAGAS metrics for evaluation
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    context_entity_recall,
    answer_similarity,
    answer_correctness
)
import numpy as np  # Numerical operations
import sys  # System-specific parameters and functions
import traceback  # Stack trace utilities for debugging
import glob  # File pattern matching
from sklearn.feature_extraction.text import TfidfVectorizer  # Text feature extraction
from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity calculation
from langchain.llms.base import LLM  # Base LLM class from LangChain
from typing import Any, List, Optional, Mapping  # Type hints for better code clarity

# Import custom services from the src directory
try:
    sys.path.append('./')  # Add current directory to system path to locate src
    from src.services.s3_service import S3Service  # S3 interaction service
    from src.services.knowledge_base_service import KnowledgeBaseService  # Knowledge base management
    from src.services.bedrock_service import BedrockService  # Bedrock model interaction
    from src.services.guardrails_service import GuardrailsService  # Security and validation service
except ImportError as e:
    # Display error if imports fail
    st.error(f"Error importing services: {str(e)}")
    st.error("Please ensure your src directory is in the correct location.")

# Main application class for RAG evaluation
class RAGEvaluationApp:
    def __init__(self, config_path=None):
        """Initialize the RAG Evaluation Dashboard application."""
        # Set Streamlit page configuration
        st.set_page_config(
            page_title="RAG Evaluation Dashboard",  # Browser tab title
            page_icon="📊",  # Favicon
            layout="wide"  # Wide layout for better visualization
        )
        
        # Load configuration file
        self.config = self._load_config(config_path)
        
        # Initialize services if config is loaded successfully
        if self.config:
            try:
                self.s3_service = S3Service(self.config['s3_config'])  # S3 service instance
                self.kb_service = KnowledgeBaseService(self.config['knowledge_base_config'])  # Knowledge base service
                self.bedrock_service = BedrockService(self.config['model_config'])  # Bedrock service
                self.guardrails_service = GuardrailsService(self.config)  # Guardrails service
                self.eval_dir = "rag_evaluation_files"  # Directory for storing evaluation files
                os.makedirs(self.eval_dir, exist_ok=True)  # Create directory if it doesn’t exist
            except Exception as e:
                # Display error if service initialization fails
                st.error(f"Error initializing services: {str(e)}")
                st.error(traceback.format_exc())
        # Initialize session state variables
        self._initialize_session_state()
    
    def _load_config(self, config_path):
        """Load common configurations from a JSON file."""
        try:
            # Determine base directory for config file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_dir = os.path.join(base_dir, 'config')
            
            # Load common config file
            common_config_path = os.path.join(config_dir, 'common_config.json')
            if not os.path.exists(common_config_path):
                raise FileNotFoundError("Common config file not found")
                
            with open(common_config_path, 'r', encoding='utf-8') as f:
                common_config = json.load(f)  # Parse JSON config
            return common_config
        
        except Exception as e:
            # Display error if config loading fails
            st.error(f"Error loading configurations: {str(e)}")
            st.error(f"Current working directory: {os.getcwd()}")
            raise
    
    def _initialize_session_state(self):
        """Initialize session state variables for persistent app state."""
        if 'page' not in st.session_state:
            st.session_state.page = 'run_test'  # Default page
        if 'test_results' not in st.session_state:
            st.session_state.test_results = None  # Store test results
        if 'current_timestamp' not in st.session_state:
            st.session_state.current_timestamp = None  # Timestamp for current run
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None  # Store evaluation results
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = 'Claude-3.5-Sonnet'  # Default model
        if 'program_config' not in st.session_state:
            st.session_state.program_config = None  # Program-specific config
        if 'selected_program' not in st.session_state:
            st.session_state.selected_program = None  # Selected program
    
    def run(self):
        """Main entry point for running the application."""
        # Apply custom styling
        self._apply_custom_css()
        
        # Render sidebar for navigation
        self._render_sidebar()
        
        # Render content based on current page
        if st.session_state.page == 'run_test':
            self._render_run_test_page()
        elif st.session_state.page == 'evaluate_rag':
            self._render_evaluate_rag_page()
        elif st.session_state.page == 'view_results':
            self._render_view_results_page()
    
    def _render_sidebar(self):
        """Render the sidebar with navigation and configuration options."""
        with st.sidebar:
            st.title("RAG Evaluation Dashboard")  # Sidebar title
            
            # Navigation buttons
            st.markdown("---")
            if st.button("Run RAG Test", use_container_width=True, 
                        type="primary" if st.session_state.page == 'run_test' else "secondary"):
                st.session_state.page = 'run_test'
                st.rerun()  # Rerun app to update page
            if st.button("Evaluate RAG", use_container_width=True,
                        type="primary" if st.session_state.page == 'evaluate_rag' else "secondary"):
                st.session_state.page = 'evaluate_rag'
                st.rerun()
            if st.button("View RAG Results", use_container_width=True,
                        type="primary" if st.session_state.page == 'view_results' else "secondary"):
                st.session_state.page = 'view_results'
                st.rerun()
            st.markdown("---")
            
            # Program and model selection (only on Run Test page)
            if st.session_state.page == 'run_test':
                st.subheader("Program Selection")
                programs = self.config.get('programs', {})  # Get programs from config
                program_options = {k: v['name'] for k, v in programs.items()}  # Map program keys to names
                selected_program = st.selectbox(
                    "Select Program",
                    options=[''] + list(program_options.keys()),  # Include empty option
                    format_func=lambda x: program_options.get(x, 'Select a program...'),
                    key="program_select"
                )
                if selected_program != st.session_state.selected_program:
                    st.session_state.selected_program = selected_program
                    st.session_state.program_config = programs.get(selected_program)
                    st.rerun()  # Update app state
                
                # Model selection if a program is selected
                if st.session_state.selected_program:
                    st.subheader("Model Selection")
                    program_config = st.session_state.program_config
                    if program_config:
                        model_options = program_config.get('models', {}).get('allowed', 
                            ['Claude-3.5-Sonnet', 'Amazon-Nova-Pro-v1'])  # Default models
                        default_model = program_config.get('models', {}).get('default', 'Claude-3.5-Sonnet')
                        selected_model = st.radio(
                            "Select Model",
                            options=model_options,
                            index=model_options.index(default_model),  # Set default
                            key="program_model_select",
                            horizontal=True
                        )
                        if selected_model != st.session_state.selected_model:
                            st.session_state.selected_model = selected_model  # Update selected model
    
    def _render_run_test_page(self):
        """Render the page for running RAG tests."""
        st.title("Run RAG Test")
        if not st.session_state.selected_program:
            st.warning("Please select a program from the sidebar to continue")
            return
        
        # Section for uploading test questions
        st.header("Upload Test Questions")
        uploaded_file = st.file_uploader("Upload CSV file with test questions", type=["csv"])
        if uploaded_file:
            try:
                # Load CSV file into DataFrame
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_columns = ["Question ID", "Question", "Context", "Ground Truth"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    return
                
                # Show preview of uploaded data
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Button to start RAG test
                if st.button("Run RAG Test", type="primary"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp
                    st.session_state.current_timestamp = timestamp
                    progress_placeholder = st.empty()  # Placeholder for progress bar
                    status_placeholder = st.empty()  # Placeholder for status messages
                    results_placeholder = st.empty()  # Placeholder for intermediate results
                    progress_bar = progress_placeholder.progress(0)  # Initialize progress bar
                    status_placeholder.info("Starting RAG evaluation...")
                    results = []
                    total_questions = len(df)
                    
                    # Process each question
                    for i, (_, row) in enumerate(df.iterrows()):
                        progress = int((i / total_questions) * 100)
                        progress_bar.progress(progress)
                        status_placeholder.info(f"Processing question {i+1}/{total_questions}: {row['Question ID']}")
                        try:
                            answer = self._get_rag_response(row['Question'])  # Get RAG response
                            results.append({
                                "Question ID": row["Question ID"],
                                "Question": row["Question"],
                                "Context": row["Context"],
                                "Ground Truth": row["Ground Truth"],
                                "Answer Generated": answer
                            })
                            # Show intermediate results every 5 questions or at the end
                            if i % 5 == 0 or i == total_questions - 1:
                                temp_df = pd.DataFrame(results)
                                results_placeholder.dataframe(temp_df.head(), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error processing question {row['Question ID']}: {str(e)}")
                            results.append({
                                "Question ID": row["Question ID"],
                                "Question": row["Question"],
                                "Context": row["Context"],
                                "Ground Truth": row["Ground Truth"],
                                "Answer Generated": f"ERROR: {str(e)}"
                            })
                    
                    # Finalize progress and save results
                    progress_bar.progress(100)
                    status_placeholder.success("RAG test completed!")
                    results_df = pd.DataFrame(results)
                    output_path = os.path.join(self.eval_dir, f"question_answer_{timestamp}.csv")
                    results_df.to_csv(output_path, index=False)
                    st.session_state.test_results = results_df
                    
                    # Offer download option
                    st.download_button(
                        label="Download Results CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"question_answer_{timestamp}.csv",
                        mime="text/csv"
                    )
                    st.success(f"Test completed! Results saved to {output_path}")
                    st.info("Now you can go to 'Evaluate RAG' to evaluate these results.")
            except Exception as e:
                st.error(f"Error processing the uploaded file: {str(e)}")
                st.error(traceback.format_exc())
    
    def _render_evaluate_rag_page(self):
        """Render the page for evaluating RAG results with RAGAS."""
        st.title("Evaluate RAG")
        
        # Section for selecting test results
        st.header("Select Test Results")
        test_files = self._get_test_files()  # Get available test files
        
        if not test_files:
            st.warning("No test result files found. Please run a RAG test first.")
            return
        
        selected_file = st.selectbox(
            "Select a test results file",
            options=test_files,
            format_func=lambda x: os.path.basename(x)
        )
        if selected_file:
            try:
                df = pd.read_csv(selected_file)  # Load selected file
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Button to start evaluation
                if st.button("Evaluate with RAGAS", type="primary"):
                    filename = os.path.basename(selected_file)
                    timestamp = filename.replace("question_answer_", "").replace(".csv", "")
                    st.session_state.current_timestamp = timestamp
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    results_placeholder = st.empty()
                    progress_bar = progress_placeholder.progress(0)
                    status_placeholder.info("Starting RAGAS evaluation...")
                    evaluation_results = []
                    total_questions = len(df)
                    
                    # Evaluate each question
                    for i, (_, row) in enumerate(df.iterrows()):
                        progress = int((i / total_questions) * 100)
                        progress_bar.progress(progress)
                        status_placeholder.info(f"Evaluating question {i+1}/{total_questions}: {row['Question ID']}")
                        try:
                            # Try RAGAS evaluation first
                            result = self._evaluate_with_ragas(
                                question=row["Question"],
                                context=row["Context"],
                                ground_truth=row["Ground Truth"],
                                answer=row["Answer Generated"]
                            )
                        except Exception as ragas_error:
                            st.warning(f"RAGAS evaluation failed for {row['Question ID']}: {str(ragas_error)}. Trying LLM evaluation.")
                            try:
                                # Fallback to LLM evaluation
                                result = self._evaluate_with_llm(
                                    question=row["Question"],
                                    context=row["Context"],
                                    ground_truth=row["Ground Truth"],
                                    answer=row["Answer Generated"]
                                )
                            except Exception as llm_error:
                                st.warning(f"LLM evaluation failed for {row['Question ID']}: {str(llm_error)}. Using fallback scoring.")
                                # Final fallback scoring
                                result = self._fallback_scoring(
                                    question=row["Question"],
                                    context=row["Context"],
                                    ground_truth=row["Ground Truth"],
                                    answer=row["Answer Generated"]
                                )
                        evaluation_results.append({
                            "Question ID": row["Question ID"],
                            "Question": row["Question"],
                            "Context": row["Context"],
                            "Ground Truth": row["Ground Truth"],
                            "Answer Generated": row["Answer Generated"],
                            "Faithfulness Score": result["faithfulness"],
                            "Answer Relevancy Score": result["answer_relevancy"],
                            "Context Precision Score": result["context_precision"],
                            "Context Recall Score": result["context_recall"],
                            "Context Entity Recall Score": result["context_entity_recall"],
                            "Semantic Similarity Score": result["answer_similarity"],
                            "Answer Correctness Score": result["answer_correctness"],
                            "Overall Score": sum(result.values()) / 7  # Average of all metrics
                        })
                        if i % 5 == 0 or i == total_questions - 1:
                            temp_df = pd.DataFrame(evaluation_results)
                            results_placeholder.dataframe(
                                temp_df[["Question ID", "Faithfulness Score", "Answer Relevancy Score", 
                                         "Context Precision Score", "Context Recall Score", 
                                         "Context Entity Recall Score", "Semantic Similarity Score", 
                                         "Answer Correctness Score", "Overall Score"]].head(),
                                use_container_width=True
                            )
                    progress_bar.progress(100)
                    status_placeholder.success("RAGAS evaluation completed!")
                    
                    # Save and display results
                    results_df = pd.DataFrame(evaluation_results)
                    output_path = os.path.join(self.eval_dir, f"RAG_Results_{timestamp}.csv")
                    results_df.to_csv(output_path, index=False)
                    st.session_state.evaluation_results = results_df
                    
                    # Calculate and display overall scores
                    overall_scores = {
                        "Average Faithfulness": results_df["Faithfulness Score"].mean(),
                        "Average Answer Relevancy": results_df["Answer Relevancy Score"].mean(),
                        "Average Context Precision": results_df["Context Precision Score"].mean(),
                        "Average Context Recall": results_df["Context Recall Score"].mean(),
                        "Average Context Entity Recall": results_df["Context Entity Recall Score"].mean(),
                        "Average Semantic Similarity": results_df["Semantic Similarity Score"].mean(),
                        "Average Answer Correctness": results_df["Answer Correctness Score"].mean(),
                        "Average Overall Score": results_df["Overall Score"].mean()
                    }
                    st.subheader("Overall Scores")
                    overall_df = pd.DataFrame([overall_scores])
                    st.dataframe(overall_df, use_container_width=True)
                    
                    # Offer download option
                    st.download_button(
                        label="Download Evaluation Results CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"RAG_Results_{timestamp}.csv",
                        mime="text/csv"
                    )
                    st.success(f"Evaluation completed! Results saved to {output_path}")
                    st.info("Now you can go to 'View RAG Results' to visualize these results.")
            except Exception as e:
                st.error(f"Error processing the selected file: {str(e)}")
                st.error(traceback.format_exc())
    
    def _get_test_files(self):
        """Get list of test result files."""
        pattern = os.path.join(self.eval_dir, "question_answer_*.csv")
        return sorted(glob.glob(pattern), reverse=True)  # Sort newest first
    
    def _get_evaluation_files(self):
        """Get list of evaluation result files."""
        pattern = os.path.join(self.eval_dir, "RAG_Results_*.csv")
        return sorted(glob.glob(pattern), reverse=True)  # Sort newest first
    
    def _evaluate_with_ragas(self, question, context, ground_truth, answer):
        """Evaluate RAG performance using RAGAS framework."""
        try:
            if not all([answer, question, context, ground_truth]):  # Check for empty inputs
                return {
                    "faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5,
                    "context_entity_recall": 0.5, "answer_similarity": 0.5, "answer_correctness": 0.5
                }
            # Configure Bedrock client
            bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
            bedrock_client = boto3.client('bedrock-runtime', config=bedrock_config)
            llm = ChatBedrock(model_id="amazon.nova-pro-v1:0", client=bedrock_client)  # Use Nova model
            embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)
            # Prepare data for RAGAS
            data = {
                "question": [question],
                "contexts": [[str(context)]],
                "ground_truth": [ground_truth],
                "answer": [answer]
            }
            hf_dataset = HFDataset.from_dict(data)  # Convert to Hugging Face dataset
            metrics = [
                faithfulness, answer_relevancy, context_precision, context_recall,
                context_entity_recall, answer_similarity, answer_correctness
            ]
            result = evaluate(dataset=hf_dataset, metrics=metrics, llm=llm, embeddings=embeddings)  # Run evaluation
            result_dict = result.to_pandas().iloc[0].to_dict()
            # Map RAGAS results to expected keys with defaults
            return {
                "faithfulness": float(result_dict.get("faithfulness", 0.5)),
                "answer_relevancy": float(result_dict.get("answer_relevancy", 0.5)),
                "context_precision": float(result_dict.get("context_precision", 0.5)),
                "context_recall": float(result_dict.get("context_recall", 0.5)),
                "context_entity_recall": float(result_dict.get("context_entity_recall", 0.5)),
                "answer_similarity": float(result_dict.get("answer_similarity", 0.5)),
                "answer_correctness": float(result_dict.get("answer_correctness", 0.5))
            }
        except Exception as e:
            print(f"RAGAS evaluation failed: {str(e)}")
            raise
    
    def _evaluate_with_llm(self, question, context, ground_truth, answer):
        """Evaluate RAG performance using LLM scoring."""
        try:
            if not all([answer, question, context, ground_truth]):  # Check for empty inputs
                return {
                    "faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5,
                    "context_entity_recall": 0.5, "answer_similarity": 0.5, "answer_correctness": 0.5
                }
            # Configure Bedrock client
            bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
            bedrock_client = boto3.client('bedrock-runtime', config=bedrock_config)
            llm = ChatBedrock(model_id="amazon.nova-pro-v1:0", client=bedrock_client)  # Use Nova model
            
            # Prepare context text, truncate if too long
            context_text = context if isinstance(context, str) else "\n".join(context)
            max_context_length = 4000
            if len(context_text) > max_context_length:
                context_text = context_text[:max_context_length] + "..."
            # Define prompts for each metric
            prompts = {
                "faithfulness": f"Question: {question}\nContext: {context_text}\nAnswer: {answer}\nOn a scale from 0 to 1, how factually consistent is the answer with the context? Provide a single number between 0 and 1.",
                "answer_relevancy": f"Question: {question}\nAnswer: {answer}\nOn a scale from 0 to 1, how relevant is this answer to the question? Provide a single number between 0 and 1.",
                "context_precision": f"Question: {question}\nContext: {context_text}\nOn a scale from 0 to 1, how much of this context is relevant to answering the question? Provide a single number between 0 and 1.",
                "context_recall": f"Question: {question}\nContext: {context_text}\nGround Truth: {ground_truth}\nOn a scale from 0 to 1, how much of the info needed for the ground truth is in the context? Provide a single number between 0 and 1.",
                "context_entity_recall": f"Question: {question}\nContext: {context_text}\nGround Truth: {ground_truth}\nOn a scale from 0 to 1, what proportion of key entities from the ground truth are in the context? Provide a single number between 0 and 1.",
                "answer_similarity": f"Ground Truth: {ground_truth}\nAnswer: {answer}\nOn a scale from 0 to 1, how semantically similar is the answer to the ground truth? Provide a single number between 0 and 1.",
                "answer_correctness": f"Question: {question}\nGround Truth: {ground_truth}\nAnswer: {answer}\nOn a scale from 0 to 1, how factually correct and complete is the answer vs. ground truth? Provide a single number between 0 and 1."
            }
            scores = {}
            for metric, prompt in prompts.items():
                try:
                    response = llm.invoke(prompt)
                    # Extract the content from the response (ChatBedrock returns a message object)
                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
                    scores[metric] = self._extract_score(response_text)
                except Exception as e:
                    print(f"Error evaluating {metric}: {str(e)}")
                    scores[metric] = 0.5  # Default score on error
            print(f"Scores:{scores}")
            return scores
        except Exception as e:
            print(f"LLM evaluation failed: {str(e)}")
            raise
    
    def _fallback_scoring(self, question, context, ground_truth, answer):
        """Fallback scoring using text similarity metrics."""
        try:
            # Convert inputs to strings, handle None
            question = str(question) if question is not None else ""
            context = str(context) if context is not None else ""
            ground_truth = str(ground_truth) if ground_truth is not None else ""
            answer = str(answer) if answer is not None else ""
            
            # Debug inputs
            print(f"Debug _fallback_scoring - Question: '{question}'")
            print(f"Debug _fallback_scoring - Context: '{context}'")
            print(f"Debug _fallback_scoring - Ground Truth: '{ground_truth}'")
            print(f"Debug _fallback_scoring - Answer: '{answer}'")

            # Return default scores if all inputs are empty
            if not any([question.strip(), context.strip(), ground_truth.strip(), answer.strip()]):
                print("All inputs are empty or whitespace. Returning default scores.")
                return {
                    "faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5,
                    "context_entity_recall": 0.5, "answer_similarity": 0.5, "answer_correctness": 0.5
                }

            def jaccard_similarity(str1, str2):
                """Calculate Jaccard similarity between two strings."""
                if not str1.strip() or not str2.strip():
                    print(f"Jaccard similarity: One or both strings empty ('{str1}', '{str2}'). Returning 0.")
                    return 0.0
                set1, set2 = set(str1.lower().split()), set(str2.lower().split())
                union_len = len(set1 | set2)
                if union_len == 0:
                    print(f"Jaccard similarity: Union is empty ('{str1}', '{str2}'). Returning 0.")
                    return 0.0
                score = len(set1 & set2) / union_len
                print(f"Jaccard similarity ('{str1[:20]}...', '{str2[:20]}...'): {score}")
                return score

            vectorizer = TfidfVectorizer()
            try:
                texts = [question, answer, ground_truth, context]
                valid_texts = [t for t in texts if t.strip()]
                if len(valid_texts) < 2:  # Need at least 2 texts for similarity
                    print("Less than 2 valid texts for TF-IDF. Falling back to Jaccard only.")
                    scores = {
                        "faithfulness": jaccard_similarity(answer, context),
                        "answer_relevancy": jaccard_similarity(question, answer),
                        "context_precision": jaccard_similarity(context, ground_truth),
                        "context_recall": jaccard_similarity(context, ground_truth),
                        "context_entity_recall": jaccard_similarity(context, ground_truth),
                        "answer_similarity": jaccard_similarity(answer, ground_truth),
                        "answer_correctness": jaccard_similarity(answer, ground_truth)
                    }
                else:
                    tfidf_matrix = vectorizer.fit_transform(texts)  # Compute TF-IDF
                    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)  # Compute cosine similarity
                    print(f"Cosine similarity matrix:\n{cosine_sim}")
                    scores = {
                        "faithfulness": cosine_sim[1, 3],  # Answer vs Context
                        "answer_relevancy": cosine_sim[1, 0],  # Answer vs Question
                        "context_precision": cosine_sim[3, 2],  # Context vs Ground Truth
                        "context_recall": jaccard_similarity(context, ground_truth),
                        "context_entity_recall": jaccard_similarity(context, ground_truth),
                        "answer_similarity": cosine_sim[1, 2],  # Answer vs Ground Truth
                        "answer_correctness": (cosine_sim[1, 2] + jaccard_similarity(answer, ground_truth)) / 2
                    }
            except Exception as tfidf_error:
                print(f"TF-IDF failed: {str(tfidf_error)}. Using Jaccard similarity only.")
                scores = {
                    "faithfulness": jaccard_similarity(answer, context),
                    "answer_relevancy": jaccard_similarity(question, answer),
                    "context_precision": jaccard_similarity(context, ground_truth),
                    "context_recall": jaccard_similarity(context, ground_truth),
                    "context_entity_recall": jaccard_similarity(context, ground_truth),
                    "answer_similarity": jaccard_similarity(answer, ground_truth),
                    "answer_correctness": jaccard_similarity(answer, ground_truth)
                }

            # Ensure scores are between 0 and 1
            final_scores = {k: max(min(v, 1.0), 0.0) for k, v in scores.items()}
            print(f"Final scores: {final_scores}")
            return final_scores

        except Exception as e:
            print(f"_fallback_scoring failed entirely: {str(e)}. Returning default scores.")
            return {
                "faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5,
                "context_entity_recall": 0.5, "answer_similarity": 0.5, "answer_correctness": 0.5
            }
    
    def _extract_score(self, response):
        """Extract a numeric score from LLM response text."""
        try:
            numbers = re.findall(r"(\d+\.\d+|\d+)", response)  # Find all numbers
            if numbers:
                return max(0.0, min(float(numbers[-1]), 1.0))  # Use last number, clamp to [0, 1]
            if "0." in response:
                parts = response.split("0.")
                if len(parts) > 1:
                    decimal_digits = "".join(c for c in parts[1].strip() if c.isdigit())
                    if decimal_digits:
                        return float("0." + decimal_digits)
            return 0.5  # Default if no valid number found
        except Exception:
            return 0.5  # Default on error
    
    def _render_model_rag_explanation(self):
        """Render an explanation of RAG systems, their evaluation, and model assessment."""
        with st.expander("Understanding RAG Systems and Evaluation"):
            st.markdown("""
            # What is RAG and How Do We Evaluate It?

            ## Overview of RAG
            Retrieval-Augmented Generation (RAG) is a hybrid AI framework that integrates information retrieval with natural language generation to produce informed, contextually grounded responses:
            - **Retrieval**: A system searches a knowledge base (e.g., S3 buckets with CCAR documents) to fetch documents or snippets relevant to a user’s query, often using vector embeddings or keyword-based search.
            - **Generation**: A large language model (LLM), such as Claude-3.5-Sonnet or Amazon-Nova-Pro-v1, generates a coherent response by synthesizing the retrieved context with the query.

            Unlike standalone LLMs that rely solely on pre-trained knowledge (which can become outdated or lead to hallucinations), RAG leverages external, dynamic data sources. This makes it ideal for domains like finance, where regulations and data evolve rapidly.

            ### How RAG Works
            1. **Query Processing**: The user’s question (e.g., "What are the latest CCAR capital requirements?") is parsed to extract intent and key terms using embeddings or tokenization.
            2. **Retrieval**: A retrieval engine (e.g., via `KnowledgeBaseService`) ranks and selects relevant documents from the knowledge base, often using cosine similarity between query and document embeddings. For example, it might fetch a 2023 CCAR regulation PDF.
            3. **Context Integration**: The retrieved context is combined with the query in a prompt (e.g., via a template defined in `prompt_template_file`), ensuring the LLM has both the question and relevant data.
            4. **Response Generation**: The LLM processes the prompt and generates an answer, ideally sticking closely to the retrieved context while maintaining fluency.

            **Example**:
            - **Query**: "What are the latest CCAR capital requirements?"
            - **Retrieved Context**: "In 2023, the Federal Reserve updated CCAR rules, requiring Tier 1 capital ratios of at least 4.5% for all banks."
            - **Generated Answer**: "The latest CCAR capital requirements, updated in 2023, mandate a minimum Tier 1 capital ratio of 4.5% for all banks."

            ## Why Evaluate RAG?
            RAG systems are powerful but prone to errors at multiple stages, necessitating rigorous evaluation:
            - **Retrieval Failures**: The system might retrieve irrelevant documents (e.g., a 2015 regulation instead of 2023) or miss critical ones, leading to incomplete or outdated answers.
            - **Generation Issues**: The LLM might misinterpret the context, hallucinate details not in the retrieved data, or produce off-topic responses.
            - **Model Variability**: Different LLMs (e.g., Claude vs. Nova) have unique strengths—Claude might excel in nuanced reasoning, while Nova might prioritize speed—impacting answer quality.
            - **Integration Challenges**: Misalignment between retrieval and generation (e.g., context ignored by the model) can degrade performance.

            Evaluation ensures:
            - **Accuracy**: Answers reflect factual data from the knowledge base.
            - **Relevance**: Responses directly address the query without tangents.
            - **Robustness**: The system performs consistently across queries and models.
            - **Efficiency**: Retrieval fetches concise, relevant context, and generation avoids unnecessary verbosity.

            ## How This App Evaluates RAG
            This dashboard provides a comprehensive pipeline to test, evaluate, and analyze RAG systems, including model-specific assessments:
            1. **Run RAG Test**:
            - **Process**: Users upload a CSV with questions, contexts, and ground truths. The app generates answers using the selected model (via `BedrockService`) and program configuration (e.g., CCAR settings).
            - **Details**: The `_get_rag_response` method queries the knowledge base (`KnowledgeBaseService`), applies guardrails (`GuardrailsService`), and generates responses, saving results as a CSV (e.g., `question_answer_20230410_123456.csv`).
            - **Model Role**: The chosen LLM (e.g., Claude-3.5-Sonnet) interprets the context and query, with its performance logged for later analysis.

            2. **Evaluate RAG**:
            - **Process**: The app evaluates generated answers using RAGAS metrics, with fallbacks to LLM scoring or text similarity if RAGAS fails.
            - **RAGAS Metrics**: Seven metrics (detailed below) assess retrieval and generation quality, scored from 0 to 1. For example, `faithfulness` checks if the answer aligns with the context, while `answer_correctness` compares it to the ground truth.
            - **Fallbacks**: If RAGAS fails (e.g., due to API timeouts), the app uses LLM-based scoring (prompting Nova to rate metrics) or text similarity (TF-IDF/Jaccard) as robust alternatives.
            - **Model Evaluation**: Scores reflect not just RAG performance but also model behavior—e.g., does Claude hallucinate less than Nova? Results are saved (e.g., `RAG_Results_20230410_123456.csv`).

            3. **View Results**:
            - **Process**: Visualizations (gauges, histograms, radar charts) and per-question breakdowns reveal performance trends.
            - **Model Insights**: Users can compare how different models perform across metrics—e.g., Nova might score higher in `answer_relevancy` due to concise outputs, while Claude excels in `semantic_similarity` due to nuanced phrasing.
            - **Diagnostics**: Low scores in `context_recall` might indicate retrieval issues, while low `faithfulness` suggests model-specific hallucinations.

            ### Key Components
            - **Knowledge Base**: Managed by `KnowledgeBaseService`, storing CCAR documents or other domain-specific data in S3, accessed via `S3Service`.
            - **Models**: Integrated via `BedrockService`, supporting LLMs like Claude-3.5-Sonnet (strong in reasoning) and Amazon-Nova-Pro-v1 (optimized for speed). Model choice is configurable in the sidebar.
            - **Guardrails**: `GuardrailsService` validates prompts and responses, redacting PII (e.g., Social Security numbers) and logging security events.
            - **Evaluation Metrics**: RAGAS provides a standardized framework, augmented by custom fallbacks for resilience.

            ## Model Evaluation in RAG
            Evaluating the LLM within a RAG system is critical, as model characteristics influence overall performance:
            - **Model Strengths**:
            - **Claude-3.5-Sonnet**: Excels in complex reasoning and context understanding, potentially boosting `faithfulness` and `answer_correctness`. However, it may be slower or over-elaborate.
            - **Amazon-Nova-Pro-v1**: Prioritizes efficiency and concise outputs, likely improving `answer_relevancy` and runtime, but might miss subtle nuances, lowering `semantic_similarity`.
            - **Evaluation Metrics**:
            - Each RAGAS metric indirectly assesses model performance. For instance, high `faithfulness` indicates the model sticks to the context, while low `answer_similarity` might reveal paraphrasing or factual drift.
            - The app’s LLM fallback (`_evaluate_with_llm`) directly queries the model for metric scores, providing a self-assessment lens (e.g., "On a scale of 0-1, how correct is this answer?").
            - **Challenges**:
            - **Hallucination**: Models might generate plausible but unsupported details, detectable via low `faithfulness`.
            - **Context Overload**: If retrieval provides too much context, models like Nova might struggle to filter, reducing `context_precision`.
            - **Bias**: Pre-trained biases in models can skew answers, requiring ground truth comparison (`answer_correctness`).

            **Practical Use Case**:
            - Testing a financial RAG system with queries like "What are the 2023 CCAR stress test scenarios?" The app retrieves relevant regulations, generates answers with Claude or Nova, and evaluates them. If Claude scores 0.9 in `faithfulness` but Nova scores 0.7, users might prefer Claude for accuracy-critical tasks, despite Nova’s speed advantage.

            ## Evaluation Challenges and Solutions
            - **Retrieval Quality**: Poor embeddings or indexing can miss key documents. Solution: Tune retrieval algorithms (e.g., adjust similarity thresholds).
            - **Model Variability**: Different LLMs handle context differently. Solution: Test multiple models and compare via this app.
            - **Ground Truth Dependency**: Metrics like `context_recall` need accurate ground truths. Solution: Ensure test data is well-curated.
            - **Scalability**: Evaluating many questions can be slow. Solution: The app’s progress bars and intermediate results keep users informed.

            By combining RAG testing, metric-based evaluation, and model analysis, this app helps refine retrieval strategies, optimize model selection, and ensure reliable, compliant outputs for applications like CCAR compliance.
            """)
        
    def _render_metrics_explanation(self):
        """Render detailed explanations of RAGAS metrics."""
        with st.expander("Understanding RAGAS Metrics"):
            st.markdown("""
            # RAGAS Metrics Explained
            RAGAS (Retrieval-Augmented Generation Assessment Suite) evaluates Retrieval-Augmented Generation (RAG) systems by providing seven metrics, each scored from 0 to 1 (1 being the best). Below, we dive into each metric with detailed explanations, evaluation processes, examples, and considerations.

            ## Faithfulness
            **What it measures**: Faithfulness assesses how factually consistent the generated answer is with the retrieved context, detecting "hallucinations" where the model invents unsupported details.

            **How it's evaluated**: 
            1. The generated answer is broken into individual statements or "atomic facts" (e.g., "The sky is blue" and "It rained yesterday").
            2. Each statement is cross-checked against the retrieved context to verify if it’s explicitly supported or inferable.
            3. The score is the proportion of statements fully supported by the context. For instance, if 3 out of 4 statements are supported, the score is 0.75.

            **Example**:
            - **Question**: "What is the capital of France?"
            - **Context**: "France is a country in Europe. Its capital is Paris."
            - **Answer**: "The capital of France is Paris, and it’s known for its wine."
            - **Evaluation**: "The capital of France is Paris" is supported (1.0), but "known for its wine" isn’t explicitly in the context (0.0). Score: ~0.5.

            **Considerations**: 
            - High faithfulness requires precise retrieval and generation alignment.
            - Ambiguous context or overly creative models can lower scores.

            ## Answer Relevancy
            **What it measures**: This metric evaluates how well the answer addresses the question, focusing on relevance, topicality, and completeness.

            **How it's evaluated**: 
            1. Semantic similarity between the question and answer is analyzed using embeddings or LLM scoring.
            2. It checks if the answer avoids extraneous details and directly responds to the query.
            3. Completeness is assessed—does the answer cover all key aspects of the question?

            **Example**:
            - **Question**: "What’s the weather like today?"
            - **Answer**: "It’s sunny and 75°F."
            - **Evaluation**: Fully relevant and complete (1.0).
            - **Bad Answer**: "The weather is nice, and I like dogs." (Relevance drops due to unrelated content, ~0.6).

            **Considerations**: 
            - Irrelevant tangents or incomplete responses lower the score.
            - Requires nuanced understanding of question intent.

            ## Context Precision
            **What it measures**: Context precision gauges the relevance of the retrieved context to the question, assessing retrieval efficiency.

            **How it's evaluated**: 
            1. The context is analyzed to determine which parts contribute to answering the question.
            2. Irrelevant sections are identified and penalized.
            3. The score reflects the proportion of context that’s useful (e.g., 80% relevant = 0.8).

            **Example**:
            - **Question**: "Who won the 2020 election?"
            - **Context**: "The 2020 election was held on November 3. Joe Biden won. Polls opened at 7 AM."
            - **Evaluation**: "Joe Biden won" is relevant; "Polls opened at 7 AM" is not. Score: ~0.67.

            **Considerations**: 
            - Overly broad retrieval lowers precision.
            - Ideal retrieval fetches only what’s needed.

            ## Context Recall
            **What it measures**: This metric checks if the retrieved context contains all necessary information to answer the question correctly, based on the ground truth.

            **How it's evaluated**: 
            1. The ground truth is broken into key information units (e.g., facts, entities).
            2. The context is checked for the presence of these units.
            3. The score is the fraction of ground truth information found in the context.

            **Example**:
            - **Question**: "What’s the tallest mountain?"
            - **Ground Truth**: "Mount Everest is the tallest mountain at 29,032 feet."
            - **Context**: "Mount Everest is the tallest mountain."
            - **Evaluation**: Missing height (29,032 feet). Score: ~0.5.

            **Considerations**: 
            - Missing critical details reduces recall.
            - Retrieval must be comprehensive.

            ## Context Entity Recall
            **What it measures**: Focuses on whether key entities (e.g., names, places) from the ground truth are present in the context.

            **How it's evaluated**: 
            1. Entities are extracted from the ground truth (e.g., "Mount Everest," "29,032 feet").
            2. The context is scanned for these entities.
            3. The score is the proportion of entities recalled.

            **Example**:
            - **Ground Truth**: "Elon Musk founded Tesla in 2003."
            - **Context**: "Tesla was founded in 2003."
            - **Evaluation**: Missing "Elon Musk." Score: ~0.67 (2/3 entities).

            **Considerations**: 
            - Entity-focused, less about full facts.
            - Sensitive to named entity recognition accuracy.

            ## Semantic Similarity
            **What it measures**: Compares the meaning of the generated answer to the ground truth, ignoring exact wording.

            **How it's evaluated**: 
            1. Both answer and ground truth are converted to embeddings (vector representations).
            2. Cosine similarity is calculated between these vectors.
            3. The score reflects meaning alignment (e.g., 0.9 = very similar).

            **Example**:
            - **Ground Truth**: "The cat is black."
            - **Answer**: "The feline is dark-colored."
            - **Evaluation**: Semantically identical (1.0).

            **Considerations**: 
            - Robust to paraphrasing but sensitive to nuance loss.
            - Embedding quality affects accuracy.

            ## Answer Correctness
            **What it measures**: Assesses factual accuracy and completeness of the answer against the ground truth.

            **How it's evaluated**: 
            1. Factual errors are identified (e.g., wrong dates, names).
            2. Omissions of key details are noted.
            3. A combined score balances accuracy and completeness (e.g., weighted average).

            **Example**:
            - **Ground Truth**: "Python was released in 1991 by Guido van Rossum."
            - **Answer**: "Python came out in 1991."
            - **Evaluation**: Accurate but misses "Guido van Rossum." Score: ~0.75.

            **Considerations**: 
            - Penalizes both errors and omissions.
            - Requires clear ground truth for comparison.
            """)
    
    def _render_view_results_page(self):
        """Render the page for viewing and analyzing RAG evaluation results."""
        st.title("RAG Results")
        st.markdown("---")
        
        # Section for selecting evaluation results
        st.header("Select Evaluation Results")
        
        # Get list of evaluation result files
        eval_files = self._get_evaluation_files()
        
        if not eval_files:
            st.warning("No evaluation result files found. Please evaluate a RAG test first.")
            return
        
        selected_file = st.selectbox(
            "Select an evaluation results file",
            options=eval_files,
            format_func=lambda x: os.path.basename(x)
        )
        
        if selected_file:
            try:
                df = pd.read_csv(selected_file)  # Load evaluation results
                self._render_results_dashboard(df)  # Render dashboard
            except Exception as e:
                st.error(f"Error processing the selected file: {str(e)}")
                st.error(traceback.format_exc())
    
    def _render_results_dashboard(self, df):
        """Render a dashboard with visualizations of evaluation results."""
        st.header("Overall RAGAS Metrics")
        
        # Calculate average scores for each metric
        avg_scores = {
            "Faithfulness": df["Faithfulness Score"].mean(),
            "Answer Relevancy": df["Answer Relevancy Score"].mean(),
            "Context Precision": df["Context Precision Score"].mean(),
            "Context Recall": df["Context Recall Score"].mean(),
            "Context Entity Recall": df["Context Entity Recall Score"].mean(),
            "Semantic Similarity": df["Semantic Similarity Score"].mean(),
            "Answer Correctness": df["Answer Correctness Score"].mean(),
            "Overall": df["Overall Score"].mean()
        }
        
        # Display metrics in four columns
        col1, col2, col3, col4 = st.columns(4)
        
        # Display metrics in columns with gauges
        with col1:
            self._render_metric_gauge("Faithfulness", avg_scores["Faithfulness"])
            self._render_metric_gauge("Semantic Similarity", avg_scores["Semantic Similarity"])
        with col2:
            self._render_metric_gauge("Answer Relevancy", avg_scores["Answer Relevancy"])
            self._render_metric_gauge("Answer Correctness", avg_scores["Answer Correctness"])
        with col3:
            self._render_metric_gauge("Context Precision", avg_scores["Context Precision"])
            self._render_metric_gauge("Context Entity Recall", avg_scores["Context Entity Recall"])
        with col4:
            self._render_metric_gauge("Context Recall", avg_scores["Context Recall"])
        
        # Overall performance gauge
        st.markdown("---")
        st.header("Overall RAG System Performance")
        st.markdown("""
        The Overall Score represents the average of all seven RAGAS metrics, providing a single measure of your RAG system's performance.
        A higher score indicates better overall performance across all evaluation dimensions.
        """)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_scores["Overall"],
            title={"text": "Overall RAG Performance Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": self._get_color_for_score(avg_scores["Overall"])},
                "steps": [
                    {"range": [0, 0.33], "color": "lightgray"},
                    {"range": [0.33, 0.67], "color": "gray"},
                    {"range": [0.67, 1], "color": "darkgray"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 0.7
                }
            }
        ))

        fig.update_layout(
            width=300,  # Adjust width as needed
            height=350, # Adjust height as needed
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add RAG system explanation first
        self._render_model_rag_explanation()
        
        # Then add metrics explanation
        self._render_metrics_explanation()
        
        # Radar chart for metric comparison
        st.markdown("---")
        st.subheader("RAGAS Metrics Comparison")
        fig = go.Figure()
        
        # Add radar chart
        fig.add_trace(go.Scatterpolar(
            r=[avg_scores["Faithfulness"], avg_scores["Answer Relevancy"], avg_scores["Context Precision"], 
               avg_scores["Context Recall"], avg_scores["Context Entity Recall"], avg_scores["Semantic Similarity"], 
               avg_scores["Answer Correctness"], avg_scores["Overall"]],
            theta=["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", 
                   "Context Entity Recall", "Semantic Similarity", "Answer Correctness", "Overall"],
            fill='toself',
            name='Average Scores'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=700,
            width=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Score distribution histograms
        st.markdown("---")
        st.header("Score Distributions")
        
        # Create two columns for distribution charts
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="Faithfulness Score", nbins=20, histfunc="count", title="Faithfulness Score Distribution", 
                               color_discrete_sequence=['#1f77b4'])
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            fig = px.histogram(df, x="Context Precision Score", nbins=20, histfunc="count", title="Context Precision Score Distribution", 
                               color_discrete_sequence=['#2ca02c'])
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            fig = px.histogram(df, x="Context Entity Recall Score", nbins=20, histfunc="count", title="Context Entity Recall Score Distribution", 
                               color_discrete_sequence=['#9467bd'])
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x="Answer Relevancy Score", nbins=20, histfunc="count", title="Answer Relevancy Score Distribution", 
                               color_discrete_sequence=['#ff7f0e'])
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            fig = px.histogram(df, x="Context Recall Score", nbins=20, histfunc="count", title="Context Recall Score Distribution", 
                               color_discrete_sequence=['#d62728'])
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            fig = px.histogram(df, x="Semantic Similarity Score", nbins=20, histfunc="count", title="Semantic Similarity Score Distribution", 
                               color_discrete_sequence=['#17becf'])
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            fig = px.histogram(df, x="Answer Correctness Score", nbins=20, histfunc="count", title="Answer Correctness Score Distribution", 
                               color_discrete_sequence=['#bcbd22'])
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        # Per-question analysis section
        st.markdown("---")
        st.header("Per-Question Analysis")
        # Question selection for detailed view
        question_ids = df["Question ID"].unique()
        selected_question = st.selectbox(
            "Select a question for detailed analysis",
            options=question_ids
        )
        
        if selected_question:
            with st.expander("Question and scores"):
                # Get the selected question data
                question_data = df[df["Question ID"] == selected_question].iloc[0]
                
                # Create columns for question and scores
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("Question Details")
                    st.markdown(f"**Question**: {question_data['Question']}")
                    st.markdown(f"**Ground Truth**: {question_data['Ground Truth']}")
                    st.markdown(f"**Generated Answer**: {question_data['Answer Generated']}")
                with col2:
                    st.subheader("Question Scores")
                    
                    # Create a score table
                    scores_df = pd.DataFrame({
                        "Metric": ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", 
                                   "Context Entity Recall", "Semantic Similarity", "Answer Correctness", "Overall"],
                        "Score": [
                            question_data["Faithfulness Score"], question_data["Answer Relevancy Score"],
                            question_data["Context Precision Score"], question_data["Context Recall Score"],
                            question_data["Context Entity Recall Score"], question_data["Semantic Similarity Score"],
                            question_data["Answer Correctness Score"], question_data["Overall Score"]
                        ]
                    })
                    st.dataframe(scores_df, use_container_width=True)
                    
                    # Create bar chart for scores
                    fig = px.bar(
                        scores_df, x="Metric", y="Score", title="RAGAS Scores", barmode="group", color="Metric",
                        color_discrete_map={
                            "Faithfulness": "#1f77b4", "Answer Relevancy": "#ff7f0e", "Context Precision": "#2ca02c",
                            "Context Recall": "#d62728", "Context Entity Recall": "#9467bd", "Semantic Similarity": "#17becf",
                            "Answer Correctness": "#bcbd22", "Overall": "#8c564b"
                        }
                    )
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
            # Display context if expanded
            with st.expander("Show Context"):
                st.text(question_data['Context'])
                
        # Correlation analysis
        st.markdown("---")
        st.header("Score Correlation Analysis")
        constant_columns = [col for col in ["Faithfulness Score", "Answer Relevancy Score", "Context Precision Score", 
                                            "Context Recall Score", "Context Entity Recall Score", "Semantic Similarity Score", 
                                            "Answer Correctness Score"] if df[col].nunique() == 1]
        correlation_matrix = df[["Faithfulness Score", "Answer Relevancy Score", "Context Precision Score", 
                                 "Context Recall Score", "Context Entity Recall Score", "Semantic Similarity Score", 
                                 "Answer Correctness Score"]].corr()
        fig = px.imshow(correlation_matrix, labels=dict(x="Metrics", y="Metrics", color="Correlation"),
                        x=correlation_matrix.columns, y=correlation_matrix.index, color_continuous_scale="RdBu",
                        color_continuous_midpoint=0)
        fig.update_layout(width=800, height=600, xaxis_tickangle=-45)
        title_text = "Correlation Matrix of RAGAS Metrics"
        if constant_columns:
            title_text += f"<br><sup>NaN values indicate zero variance in: {', '.join(constant_columns)}</sup>"
        fig.update_layout(title_text=title_text)
        st.plotly_chart(fig, use_container_width=False)
        
        # Full results table with filtering
        st.markdown("---")
        st.header("Complete Results")
        
        # Allow filtering by score thresholds
        with st.expander("Filter Results by Score"):
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            with col1:
                min_faithfulness = st.slider("Min Faithfulness Score", 0.0, 1.0, 0.0, 0.05)
            with col2:
                min_answer_relevancy = st.slider("Min Answer Relevancy Score", 0.0, 1.0, 0.0, 0.05)
            with col3:
                min_context_precision = st.slider("Min Context Precision Score", 0.0, 1.0, 0.0, 0.05)
            with col4:
                min_context_recall = st.slider("Min Context Recall Score", 0.0, 1.0, 0.0, 0.05)
            with col5:
                min_context_entity_recall = st.slider("Min Context Entity Recall Score", 0.0, 1.0, 0.0, 0.05)
            with col6:
                min_semantic_similarity = st.slider("Min Semantic Similarity Score", 0.0, 1.0, 0.0, 0.05)
            with col7:
                min_answer_correctness = st.slider("Min Answer Correctness Score", 0.0, 1.0, 0.0, 0.05)
        filtered_df = df[
            (df["Faithfulness Score"] >= min_faithfulness) &
            (df["Answer Relevancy Score"] >= min_answer_relevancy) &
            (df["Context Precision Score"] >= min_context_precision) &
            (df["Context Recall Score"] >= min_context_recall) &
            (df["Context Entity Recall Score"] >= min_context_entity_recall) &
            (df["Semantic Similarity Score"] >= min_semantic_similarity) &
            (df["Answer Correctness Score"] >= min_answer_correctness)
        ]
        
        # Display filtered results
        st.dataframe(
            filtered_df[["Question ID", "Question", "Faithfulness Score", "Answer Relevancy Score", 
                         "Context Precision Score", "Context Recall Score", "Context Entity Recall Score", 
                         "Semantic Similarity Score", "Answer Correctness Score", "Overall Score"]],
            use_container_width=True
        )
        
        # Provide download option for filtered results
        if not filtered_df.equals(df):
            st.download_button(
                label="Download Filtered Results",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name="filtered_rag_results.csv",
                mime="text/csv"
            )
    
    def _render_metric_gauge(self, title, value):
        """Render a gauge visualization for a metric."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "green"},
                'steps': [{'range': [0, 0.5], 'color': "red"}, {'range': [0.5, 0.75], 'color': "yellow"}, 
                          {'range': [0.75, 1], 'color': "green"}],
                'bgcolor': "white",
                'bordercolor': "gray"
            }
        ))
        fig.update_layout(width=300, height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=False)
    
    def _get_color_for_score(self, score):
        """Determine color based on score value."""
        if score < 0.33:
            return "red"
        elif score < 0.67:
            return "yellow"
        else:
            return "green"
    
    def _get_rag_response(self, prompt):
        """Generate a RAG response for a given prompt."""
        try:
            # Use the existing _get_ccar_response method to get the response
            # This ensures all configs are set properly
            response = self._get_ccar_response(prompt)
            
            # Parse the response to extract just the answer
            try:
                response_data = json.loads(response)  # Parse JSON response
                if "answer" in response_data:
                    return response_data["answer"]  # Extract answer
                return response
            except json.JSONDecodeError:
                return response  # Return raw response if not JSON
        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            print(f"RAG response error: {error_msg}")
            print(traceback.format_exc())
            return error_msg
    
    def _get_ccar_response(self, prompt: str) -> str:
        """Generate a response for CCAR-related queries."""
        try:
            program_config = st.session_state.program_config
            if not program_config:
                return json.dumps({
                    "answer": "Please select a program to continue.",
                    "chart_data": []
                })
                
            # Validate prompt using guardrails
            is_valid, error_message, redacted_prompt = self.guardrails_service.validate_prompt(prompt)
            
            if not is_valid:
                return json.dumps({
                    "answer": f"Security Alert: {error_message}. Please revise your question",
                    "chart_data": []
                })

            # Use redacted prompt if PII was detected and not redacted
            safe_prompt = redacted_prompt if redacted_prompt else prompt
            
            # Get program-specific configuration
            prompt_template_file = program_config.get('prompt_template_file')
            if not prompt_template_file:
                raise ValueError(f"Prompt template file not configured for program {st.session_state.selected_program}")

            # Get response from Knowledge Base service
            response = self.kb_service.search_ccar_documents_new(
                query=safe_prompt,
                bedrock_service=self.bedrock_service,
                s3_service=self.s3_service,
                prompt_template_file=prompt_template_file,
                program_config=program_config,
                chat_history=[],  # Empty chat history for evaluation
                use_nova=st.session_state.selected_model == 'Amazon-Nova-Pro-v1'                
            )

            if not response:
                return json.dumps({
                    "answer": "I couldn't find relevant information in the CCAR documents to answer your question.",
                    "chart_data": []
                })

            # Validate prompt using guardrails
            is_valid, error_message, redacted_response = self.guardrails_service.validate_prompt(response)
            
            if not is_valid:
                return json.dumps({
                    "answer": f"Security Alert: {error_message}. Please revise your question",
                    "chart_data": []
                })

            # Use redacted prompt if PII was detected and not redacted
            final_response = redacted_response if redacted_response else response
            
            # Log successful interaction with proper variable reference
            self.guardrails_service.log_security_event(
                "successful_interaction",
                {
                    "prompt_length": len(safe_prompt),
                    "response_length": len(final_response),
                    "pii_redacted": bool(redacted_prompt or redacted_response)
                }
            )
            
            return final_response

        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            return json.dumps({
                "answer": error_msg,
                "chart_data": []
            })
            
    def _apply_custom_css(self):
        """Apply custom CSS styles to the Streamlit app."""
        st.markdown("""
        <style>
        .main { padding: 1rem 1rem; }
        h1, h2, h3, h4 { color: #0051A2; }
        .stButton > button { background-color: #0051A2; color: white; border-radius: 4px; padding: 0.5rem 1rem; border: none; transition: all 0.3s ease; }
        .stButton > button:hover { background-color: #003D82; }
        .css-1r6slb0, .css-1y4p8pa { border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
        .metric-card { background-color: white; border-radius: 6px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
        .metric-value { font-size: 2rem; font-weight: bold; color: #0051A2; }
        .metric-label { font-size: 0.9rem; color: #666; }
        .stProgress > div > div { background-color: #0051A2; }
        [data-testid="stSidebar"] { background-color: #f5f7f9; }
        [data-testid="stSidebar"] h1 { padding-left: 1rem; }
        .success-message { background-color: #d1e7dd; color: #0f5132; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; }
        .info-message { background-color: #cff4fc; color: #055160; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; }
        .warning-message { background-color: #fff3cd; color: #664d03; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; }
        .error-message { background-color: #f8d7da; color: #842029; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; }
        </style>
        """, unsafe_allow_html=True)

def main():
    """Main function to instantiate and run the app."""
    app = RAGEvaluationApp()
    app.run()

if __name__ == "__main__":
    main()  # Run the app if script is executed directly
