import streamlit as st
import pandas as pd
import os
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import functools
import re
import boto3
from botocore.client import Config
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from datasets import Dataset as HFDataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    context_entity_recall,
    answer_similarity,
    answer_correctness
)
import numpy as np
import sys
import traceback
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.llms.base import LLM
from typing import Any, List, Optional, Mapping

# Patch asyncio for RAGAS compatibility
def _patch_asyncio():
    original_get_event_loop = asyncio.get_event_loop
    @functools.wraps(original_get_event_loop)
    def patched_get_event_loop():
        try:
            return original_get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    asyncio.get_event_loop = patched_get_event_loop

_patch_asyncio()

# Import custom services
try:
    sys.path.append('./')  # Adjust this path to where your src directory is located
    from src.services.s3_service import S3Service
    from src.services.knowledge_base_service import KnowledgeBaseService
    from src.services.bedrock_service import BedrockService
    from src.services.guardrails_service import GuardrailsService
except ImportError as e:
    st.error(f"Error importing services: {str(e)}")
    st.error("Please ensure your src directory is in the correct location.")

class RAGEvaluationApp:
    def __init__(self, config_path=None):
        st.set_page_config(
            page_title="RAG Evaluation Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        self.config = self._load_config(config_path)
        
        # Initialize services if config loaded successfully
        if self.config:
            try:
                self.s3_service = S3Service(self.config['s3_config'])
                self.kb_service = KnowledgeBaseService(self.config['knowledge_base_config'])
                self.bedrock_service = BedrockService(self.config['model_config'])
                self.guardrails_service = GuardrailsService(self.config)
                self.eval_dir = "rag_evaluation_files"
                os.makedirs(self.eval_dir, exist_ok=True)
            except Exception as e:
                st.error(f"Error initializing services: {str(e)}")
                st.error(traceback.format_exc())
        self._initialize_session_state()
    
    def _load_config(self, config_path):
        """Load common configurations."""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_dir = os.path.join(base_dir, 'config')
            
            # Load common config
            common_config_path = os.path.join(config_dir, 'common_config.json')
            if not os.path.exists(common_config_path):
                raise FileNotFoundError("Common config file not found")
                
            with open(common_config_path, 'r', encoding='utf-8') as f:
                common_config = json.load(f)
                
            return common_config
        
        except Exception as e:
            st.error(f"Error loading configurations: {str(e)}")
            st.error(f"Current working directory: {os.getcwd()}")
            raise
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'page' not in st.session_state:
            st.session_state.page = 'run_test'
        if 'test_results' not in st.session_state:
            st.session_state.test_results = None
        if 'current_timestamp' not in st.session_state:
            st.session_state.current_timestamp = None
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = 'Claude-3.5-Sonnet'
        if 'program_config' not in st.session_state:
            st.session_state.program_config = None
        if 'selected_program' not in st.session_state:
            st.session_state.selected_program = None
            
    def run(self):
        """Main entry point for the application."""
        # Apply custom CSS
        self._apply_custom_css()
        
        # Sidebar for navigation
        self._render_sidebar()
        
        # Main content based on selected page
        if st.session_state.page == 'run_test':
            self._render_run_test_page()
        elif st.session_state.page == 'evaluate_rag':
            self._render_evaluate_rag_page()
        elif st.session_state.page == 'view_results':
            self._render_view_results_page()
    
    def _render_sidebar(self):
        """Render the sidebar with navigation options."""
        with st.sidebar:
            st.title("RAG Evaluation Dashboard")
            
            # Navigation
            st.markdown("---")
            if st.button("Run RAG Test", use_container_width=True, 
                        type="primary" if st.session_state.page == 'run_test' else "secondary"):
                st.session_state.page = 'run_test'
                st.rerun()
            if st.button("Evaluate RAG", use_container_width=True,
                        type="primary" if st.session_state.page == 'evaluate_rag' else "secondary"):
                st.session_state.page = 'evaluate_rag'
                st.rerun()
            if st.button("View RAG Results", use_container_width=True,
                        type="primary" if st.session_state.page == 'view_results' else "secondary"):
                st.session_state.page = 'view_results'
                st.rerun()
            st.markdown("---")
            
            # Program Selection (only on Run Test page)
            if st.session_state.page == 'run_test':
                st.subheader("Program Selection")
                programs = self.config.get('programs', {})
                program_options = {k: v['name'] for k, v in programs.items()}
                selected_program = st.selectbox(
                    "Select Program",
                    options=[''] + list(program_options.keys()),
                    format_func=lambda x: program_options.get(x, 'Select a program...'),
                    key="program_select"
                )
                if selected_program != st.session_state.selected_program:
                    st.session_state.selected_program = selected_program
                    st.session_state.program_config = programs.get(selected_program)
                    st.rerun()
                
                # Model Selection
                if st.session_state.selected_program:
                    st.subheader("Model Selection")
                    program_config = st.session_state.program_config
                    if program_config:
                        model_options = program_config.get('models', {}).get('allowed', 
                            ['Claude-3.5-Sonnet', 'Amazon-Nova-Pro-v1'])
                        default_model = program_config.get('models', {}).get('default', 'Claude-3.5-Sonnet')
                        selected_model = st.radio(
                            "Select Model",
                            options=model_options,
                            index=model_options.index(default_model),
                            key="program_model_select",
                            horizontal=True
                        )
                        if selected_model != st.session_state.selected_model:
                            st.session_state.selected_model = selected_model
    
    def _render_run_test_page(self):
        """Render the Run RAG Test page."""
        st.title("Run RAG Test")
        if not st.session_state.selected_program:
            st.warning("Please select a program from the sidebar to continue")
            return
        
        # File upload section
        st.header("Upload Test Questions")
        uploaded_file = st.file_uploader("Upload CSV file with test questions", type=["csv"])
        if uploaded_file:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_columns = ["Question ID", "Question", "Context", "Ground Truth"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    return
                # Display the data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Run test button
                if st.button("Run RAG Test", type="primary"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state.current_timestamp = timestamp
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    results_placeholder = st.empty()
                    progress_bar = progress_placeholder.progress(0)
                    status_placeholder.info("Starting RAG evaluation...")
                    results = []
                    total_questions = len(df)
                    for i, (_, row) in enumerate(df.iterrows()):
                        # Update progress
                        progress = int((i / total_questions) * 100)
                        progress_bar.progress(progress)
                        status_placeholder.info(f"Processing question {i+1}/{total_questions}: {row['Question ID']}")
                        try:
                            # Process the question
                            answer = self._get_rag_response(row['Question'])
                            
                            # Store results
                            results.append({
                                "Question ID": row["Question ID"],
                                "Question": row["Question"],
                                "Context": row["Context"],
                                "Ground Truth": row["Ground Truth"],
                                "Answer Generated": answer
                            })
                            
                            # Display intermediate results
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
                    
                    # Complete the progress bar
                    progress_bar.progress(100)
                    status_placeholder.success("RAG test completed!")
                    
                    # Create final DataFrame and save to file
                    results_df = pd.DataFrame(results)
                    output_path = os.path.join(self.eval_dir, f"question_answer_{timestamp}.csv")
                    results_df.to_csv(output_path, index=False)
                    
                    # Store results in session state
                    st.session_state.test_results = results_df
                    
                    # Provide download option
                    st.download_button(
                        label="Download Results CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"question_answer_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                    # Add notification about next steps
                    st.success(f"Test completed! Results saved to {output_path}")
                    st.info("Now you can go to 'Evaluate RAG' to evaluate these results.")
            except Exception as e:
                st.error(f"Error processing the uploaded file: {str(e)}")
                st.error(traceback.format_exc())
    
    def _render_evaluate_rag_page(self):
        """Render the Evaluate RAG page."""
        st.title("Evaluate RAG")
        
        # File selection section
        st.header("Select Test Results")
        
        # Get list of test result files
        test_files = self._get_test_files()
        
        if not test_files:
            st.warning("No test result files found. Please run a RAG test first.")
            return
        
        # File selection
        selected_file = st.selectbox(
            "Select a test results file",
            options=test_files,
            format_func=lambda x: os.path.basename(x)
        )
        if selected_file:
            try:
                # Load the selected file
                df = pd.read_csv(selected_file)
                
                # Display the data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Evaluate button
                if st.button("Evaluate with RAGAS", type="primary"):
                    # Extract timestamp from filename
                    filename = os.path.basename(selected_file)
                    timestamp = filename.replace("question_answer_", "").replace(".csv", "")
                    st.session_state.current_timestamp = timestamp
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    results_placeholder = st.empty()
                    progress_bar = progress_placeholder.progress(0)
                    status_placeholder.info("Starting RAGAS evaluation...")
                    
                    # Prepare data for RAGAS evaluation
                    evaluation_results = []
                    total_questions = len(df)
                    
                    for i, (_, row) in enumerate(df.iterrows()):
                        # Update progress
                        progress = int((i / total_questions) * 100)
                        progress_bar.progress(progress)
                        status_placeholder.info(f"Evaluating question {i+1}/{total_questions}: {row['Question ID']}")
                        try:
                            # Primary evaluation with RAGAS
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
                                # Final fallback
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
                            "Overall Score": sum(result.values()) / 7  # Average of all 7 metrics
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
                    
                    # Create final DataFrame and save to file
                    results_df = pd.DataFrame(evaluation_results)
                    output_path = os.path.join(self.eval_dir, f"RAG_Results_{timestamp}.csv")
                    results_df.to_csv(output_path, index=False)
                    
                    # Store results in session state
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
                    
                    # Provide download option
                    st.download_button(
                        label="Download Evaluation Results CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"RAG_Results_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                    # Add notification about next steps
                    st.success(f"Evaluation completed! Results saved to {output_path}")
                    st.info("Now you can go to 'View RAG Results' to visualize these results.")
                    
            except Exception as e:
                st.error(f"Error processing the selected file: {str(e)}")
                st.error(traceback.format_exc())
    
    def _get_test_files(self):
        pattern = os.path.join(self.eval_dir, "question_answer_*.csv")
        return sorted(glob.glob(pattern), reverse=True)
    
    def _get_evaluation_files(self):
        pattern = os.path.join(self.eval_dir, "RAG_Results_*.csv")
        return sorted(glob.glob(pattern), reverse=True)
    
    def _evaluate_with_ragas(self, question, context, ground_truth, answer):
        try:
            if not all([answer, question, context, ground_truth]):
                return {
                    "faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5,
                    "context_entity_recall": 0.5, "answer_similarity": 0.5, "answer_correctness": 0.5
                }
            bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
            bedrock_client = boto3.client('bedrock-runtime', config=bedrock_config)
            llm = ChatBedrock(model_id="amazon.nova-pro-v1:0", client=bedrock_client)
            embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)
            data = {
                "question": [question],
                "contexts": [[str(context)]],
                "ground_truth": [ground_truth],
                "answer": [answer]
            }
            hf_dataset = HFDataset.from_dict(data)
            metrics = [
                faithfulness, answer_relevancy, context_precision, context_recall,
                context_entity_recall, answer_similarity, answer_correctness
            ]
            result = evaluate(dataset=hf_dataset, metrics=metrics, llm=llm, embeddings=embeddings)
            result_dict = result.to_pandas().iloc[0].to_dict()
            # Explicitly map RAGAS metric names to our expected keys
            return {
                "faithfulness": float(result_dict.get("faithfulness", 0.5)),
                "answer_relevancy": float(result_dict.get("answer_relevancy", 0.5)),
                "context_precision": float(result_dict.get("context_precision", 0.5)),
                "context_recall": float(result_dict.get("context_recall", 0.5)),
                "context_entity_recall": float(result_dict.get("context_entity_recall", 0.5)),
                "answer_similarity": float(result_dict.get("answer_similarity", 0.5)),  # Ensure this matches RAGAS output
                "answer_correctness": float(result_dict.get("answer_correctness", 0.5))
            }
        except Exception as e:
            print(f"RAGAS evaluation failed: {str(e)}")
            raise
    
    def _evaluate_with_llm(self, question, context, ground_truth, answer):
        try:
            if not all([answer, question, context, ground_truth]):
                return {
                    "faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5,
                    "context_entity_recall": 0.5, "answer_similarity": 0.5, "answer_correctness": 0.5
                }
            bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
            bedrock_client = boto3.client('bedrock-runtime', config=bedrock_config)
            llm = ChatBedrock(model_id="amazon.nova-pro-v1:0", client=bedrock_client)
            
            context_text = context if isinstance(context, str) else "\n".join(context)
            max_context_length = 4000
            if len(context_text) > max_context_length:
                context_text = context_text[:max_context_length] + "..."
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
                    # Use invoke instead of deprecated __call__
                    response = llm.invoke(prompt)
                    # Extract the content from the response (ChatBedrock returns a message object)
                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
                    scores[metric] = self._extract_score(response_text)
                except Exception as e:
                    print(f"Error evaluating {metric}: {str(e)}")
                    scores[metric] = 0.5
            print(f"Scores:{scores}")
            return scores
        except Exception as e:
            print(f"LLM evaluation failed: {str(e)}")
            raise
    
    def _fallback_scoring(self, question, context, ground_truth, answer):
        try:
            # Convert inputs to strings and handle None/empty cases
            question = str(question) if question is not None else ""
            context = str(context) if context is not None else ""
            ground_truth = str(ground_truth) if ground_truth is not None else ""
            answer = str(answer) if answer is not None else ""
            
            # Debug: Print inputs to check their values
            print(f"Debug _fallback_scoring - Question: '{question}'")
            print(f"Debug _fallback_scoring - Context: '{context}'")
            print(f"Debug _fallback_scoring - Ground Truth: '{ground_truth}'")
            print(f"Debug _fallback_scoring - Answer: '{answer}'")

            # If all inputs are empty, return default scores
            if not any([question.strip(), context.strip(), ground_truth.strip(), answer.strip()]):
                print("All inputs are empty or whitespace. Returning default scores.")
                return {
                    "faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5,
                    "context_entity_recall": 0.5, "answer_similarity": 0.5, "answer_correctness": 0.5
                }

            def jaccard_similarity(str1, str2):
                # Handle empty strings to avoid division by zero
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
                # Ensure there’s enough valid text for TF-IDF
                texts = [question, answer, ground_truth, context]
                valid_texts = [t for t in texts if t.strip()]
                if len(valid_texts) < 2:
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
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
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

            # Clamp scores between 0 and 1
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
        try:
            numbers = re.findall(r"(\d+\.\d+|\d+)", response)
            if numbers:
                return max(0.0, min(float(numbers[-1]), 1.0))
            if "0." in response:
                parts = response.split("0.")
                if len(parts) > 1:
                    decimal_digits = "".join(c for c in parts[1].strip() if c.isdigit())
                    if decimal_digits:
                        return float("0." + decimal_digits)
            return 0.5
        except Exception:
            return 0.5
        
    def _render_metrics_explanation(self):
        """Render detailed explanations of RAGAS metrics."""
        with st.expander("Understanding RAGAS Metrics"):
            st.markdown("""
    # RAGAS Metrics Explained
    RAGAS provides seven key metrics to evaluate the quality of RAG systems, each scoring from 0 to 1, where 1 is the best possible score:

    ## Faithfulness
    **What it measures**: How factually consistent the generated answer is with the provided context.

    **How it's evaluated**: Faithfulness detects hallucinations by measuring if the generated answer contains information not supported by the retrieved context. The metric works by:
    1. Breaking down the answer into atomic facts or statements.
    2. Verifying each statement against the provided context.
    3. Scoring based on the proportion of statements that are supported by the context.

    **Interpretation**:
    - Score of 1.0: All information in the answer is supported by the retrieved context.
    - Score of 0.0: None of the information in the answer is supported by the context (complete hallucination).

    ## Answer Relevancy
    **What it measures**: How relevant the generated answer is to the question asked.

    **How it's evaluated**: Answer relevancy analyzes semantic similarity between the question and the answer. It evaluates:
    1. Whether the answer directly addresses what was asked.
    2. If the answer stays on topic and doesn’t include irrelevant information.
    3. If the answer is complete relative to what was asked.

    **Interpretation**:
    - Score of 1.0: The answer perfectly addresses the question with high relevance.
    - Score of 0.0: The answer is completely unrelated to the question asked.

    ## Context Precision
    **What it measures**: How much of the retrieved context is actually relevant and used in generating the answer.

    **How it's evaluated**: Context precision measures if the RAG system is retrieving concise, relevant context. It works by:
    1. Analyzing how much of the retrieved context is actually relevant to the question.
    2. Identifying portions of context that don’t contribute to answering the question.
    3. Measuring efficiency of information retrieval.

    **Interpretation**:
    - Score of 1.0: All retrieved context is relevant and helpful for answering the question.
    - Score of 0.0: None of the retrieved context is relevant to the question.

    ## Context Recall
    **What it measures**: How much of the information needed to answer the question is present in the retrieved context.

    **How it's evaluated**: Context recall determines if the RAG system is retrieving all necessary information by:
    1. Comparing the retrieved context with the ground truth answer.
    2. Identifying key information needed to answer the question correctly.
    3. Measuring how much of that key information is present in the retrieved context.

    **Interpretation**:
    - Score of 1.0: The retrieved context contains all information needed to answer the question.
    - Score of 0.0: The retrieved context is missing all information needed to answer the question.

    ## Context Entity Recall
    **What it measures**: How well the retrieved context captures the key entities (e.g., people, places, things) present in the ground truth.

    **How it's evaluated**: Context entity recall focuses on entity-level completeness by:
    1. Extracting key entities from the ground truth answer.
    2. Checking the presence of these entities in the retrieved context.
    3. Scoring based on the proportion of ground truth entities found in the context.

    **Interpretation**:
    - Score of 1.0: All key entities from the ground truth are present in the retrieved context.
    - Score of 0.0: None of the key entities from the ground truth are present in the retrieved context.

    ## Semantic Similarity
    **What it measures**: How similar the meaning of the generated answer is to the ground truth answer.

    **How it's evaluated**: Semantic similarity uses embeddings to compare the generated answer and ground truth. It works by:
    1. Converting both the answer and ground truth into vector representations.
    2. Calculating cosine similarity between these vectors.
    3. Scoring based on how closely the meanings align, regardless of exact wording.

    **Interpretation**:
    - Score of 1.0: The generated answer has identical meaning to the ground truth.
    - Score of 0.0: The generated answer has no semantic overlap with the ground truth.

    ## Answer Correctness
    **What it measures**: How factually correct and complete the generated answer is compared to the ground truth.

    **How it's evaluated**: Answer correctness assesses both accuracy and completeness by:
    1. Checking factual alignment between the answer and ground truth.
    2. Evaluating if the answer includes all necessary details from the ground truth.
    3. Combining these into a single score, often weighting factual errors and omissions.

    **Interpretation**:
    - Score of 1.0: The answer is fully factually correct and complete compared to the ground truth.
    - Score of 0.0: The answer is completely incorrect or missing key information from the ground truth.
    """)
    
    def _render_view_results_page(self):
        """Render the View RAG Results page."""
        st.title("RAG Results")
        st.markdown("---")
        
        # File selection section
        st.header("Select Evaluation Results")
        
        # Get list of evaluation result files
        eval_files = self._get_evaluation_files()
        
        if not eval_files:
            st.warning("No evaluation result files found. Please evaluate a RAG test first.")
            return
        
        # File selection
        selected_file = st.selectbox(
            "Select an evaluation results file",
            options=eval_files,
            format_func=lambda x: os.path.basename(x)
        )
        
        if selected_file:
            try:
                # Load the selected file
                df = pd.read_csv(selected_file)
                
                # Display results dashboard
                self._render_results_dashboard(df)
                
            except Exception as e:
                st.error(f"Error processing the selected file: {str(e)}")
                st.error(traceback.format_exc())
    
    def _render_results_dashboard(self, df):
        """Render a dashboard visualizing the evaluation results."""
        # Overall metrics section
        st.header("Overall RAGAS Metrics")
        
        # Calculate average scores
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
        
        # Create columns for metrics
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

        # Add the metrics explanation section after the overall score gauge
        self._render_metrics_explanation()
        
        # Display radar chart for overall metrics
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
        
        # Score distribution section
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
        
        # Per-question analysis
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
                
        # Score correlation analysis
        st.markdown("---")
        st.header("Score Correlation Analysis")
        constant_columns = []
        for col in ["Faithfulness Score", "Answer Relevancy Score", "Context Precision Score", "Context Recall Score",
                    "Context Entity Recall Score", "Semantic Similarity Score", "Answer Correctness Score"]:
            if df[col].nunique() == 1:
                constant_columns.append(col)
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
        
        # Display full results table
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
        """Render a gauge metric with a title and value."""
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
        """Get color based on score value."""
        if score < 0.33:
            return "red"
        elif score < 0.67:
            return "yellow"
        else:
            return "green"
    
    def _get_rag_response(self, prompt):
        """Get RAG response for a prompt."""
        try:
            # Use the existing _get_ccar_response method to get the response
            # This ensures all configs are set properly
            response = self._get_ccar_response(prompt)
            
            # Parse the response to extract just the answer
            try:
                response_data = json.loads(response)
                if "answer" in response_data:
                    return response_data["answer"]
                return response
            except json.JSONDecodeError:
                return response

        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            print(f"RAG response error: {error_msg}")
            print(traceback.format_exc())
            return error_msg
            
    def _get_ccar_response(self, prompt: str) -> str:
        """Generate response for CCAR queries."""
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
    # Create and run the application
    app = RAGEvaluationApp()
    app.run()

if __name__ == "__main__":
    main()
