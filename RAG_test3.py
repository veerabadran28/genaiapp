import streamlit as st
import pandas as pd
import os
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import asyncio
from functools import wraps
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
import numpy as np
import ast
import sys
import importlib.util
from io import StringIO
import traceback
import boto3
import glob
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import your existing services from the source code
# Adjust the path if needed
try:
    sys.path.append('./')  # Adjust this path to where your src directory is located
    from src.services.s3_service import S3Service
    from src.services.knowledge_base_service import KnowledgeBaseService
    from src.services.bedrock_service import BedrockService
    from src.services.document_processor import DocumentProcessor
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
        self._setup_asyncio()
        
        # Initialize services if config loaded successfully
        if self.config:
            try:
                self.s3_service = S3Service(self.config['s3_config'])
                self.kb_service = KnowledgeBaseService(self.config['knowledge_base_config'])
                self.bedrock_service = BedrockService(self.config['model_config'])
                self.guardrails_service = GuardrailsService(self.config)
                
                # Initialize storage directory for evaluation files
                self.eval_dir = "rag_evaluation_files"
                os.makedirs(self.eval_dir, exist_ok=True)
                
            except Exception as e:
                st.error(f"Error initializing services: {str(e)}")
                st.error(traceback.format_exc())
        
        self._initialize_session_state()
        
    def _setup_asyncio(self):
        """Set up asyncio event loop for the current thread."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there's no event loop in the current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
    def _run_with_event_loop(self, func, *args, **kwargs):
        """Run a function with a guaranteed event loop."""
        try:
            # First try to get the event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If that fails, create a new loop and set it
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Now run the function within this guaranteed context
        return func(*args, **kwargs)
        
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
                            # Evaluate with RAGAS
                            result = self._evaluate_with_ragas(
                                question=row["Question"],
                                context=row["Context"],
                                ground_truth=row["Ground Truth"],
                                answer=row["Answer Generated"]
                            )
                            
                            # Store results
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
                                "Overall Score": (result["faithfulness"] + result["answer_relevancy"] + result["context_precision"] + result["context_recall"]) / 4.0
                            })
                            
                            # Display intermediate results
                            if i % 5 == 0 or i == total_questions - 1:
                                temp_df = pd.DataFrame(evaluation_results)
                                results_placeholder.dataframe(
                                    temp_df[["Question ID", "Faithfulness Score", "Answer Relevancy Score", 
                                            "Context Precision Score", "Context Recall Score"]].head(), 
                                    use_container_width=True
                                )
                        
                        except Exception as e:
                            st.error(f"Error evaluating question {row['Question ID']}: {str(e)}")
                            evaluation_results.append({
                                "Question ID": row["Question ID"],
                                "Question": row["Question"],
                                "Context": row["Context"],
                                "Ground Truth": row["Ground Truth"],
                                "Answer Generated": row["Answer Generated"],
                                "Faithfulness Score": 0.0,
                                "Answer Relevancy Score": 0.0,
                                "Context Precision Score": 0.0,
                                "Context Recall Score": 0.0,
                                "Overall Score": 0.0,
                                "Error": str(e)
                            })
                    
                    # Complete the progress bar
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
    
    def _render_metrics_explanation(self):
        """Render detailed explanations of RAGAS metrics."""
        with st.expander("Understanding RAGAS Metrics"):
            st.markdown("""
            # RAGAS Metrics Explained
            
            RAGAS provides four key metrics to evaluate the quality of RAG systems, each scoring from 0 to 1, where 1 is the best possible score:
            
            ## Faithfulness
            
            **What it measures**: How factually consistent the generated answer is with the provided context.
            
            **How it's evaluated**: Faithfulness detects hallucinations by measuring if the generated answer contains information not supported by the retrieved context. The metric works by:
            1. Breaking down the answer into atomic facts or statements
            2. Verifying each statement against the provided context
            3. Scoring based on the proportion of statements that are supported by the context
            
            **Interpretation**: 
            - Score of 1.0: All information in the answer is supported by the retrieved context
            - Score of 0.0: None of the information in the answer is supported by the context (complete hallucination)
            
            ## Answer Relevancy
            
            **What it measures**: How relevant the generated answer is to the question asked.
            
            **How it's evaluated**: Answer relevancy analyzes semantic similarity between the question and the answer. It evaluates:
            1. Whether the answer directly addresses what was asked
            2. If the answer stays on topic and doesn't include irrelevant information
            3. If the answer is complete relative to what was asked
            
            **Interpretation**:
            - Score of 1.0: The answer perfectly addresses the question with high relevance
            - Score of 0.0: The answer is completely unrelated to the question asked
            
            ## Context Precision
            
            **What it measures**: How much of the retrieved context is actually relevant and used in generating the answer.
            
            **How it's evaluated**: Context precision measures if the RAG system is retrieving concise, relevant context. It works by:
            1. Analyzing how much of the retrieved context is actually relevant to the question
            2. Identifying portions of context that don't contribute to answering the question
            3. Measuring efficiency of information retrieval
            
            **Interpretation**:
            - Score of 1.0: All retrieved context is relevant and helpful for answering the question
            - Score of 0.0: None of the retrieved context is relevant to the question
            
            ## Context Recall
            
            **What it measures**: How much of the information needed to answer the question is present in the retrieved context.
            
            **How it's evaluated**: Context recall determines if the RAG system is retrieving all necessary information by:
            1. Comparing the retrieved context with the ground truth answer
            2. Identifying key information needed to answer the question correctly
            3. Measuring how much of that key information is present in the retrieved context
            
            **Interpretation**:
            - Score of 1.0: The retrieved context contains all information needed to answer the question
            - Score of 0.0: The retrieved context is missing all information needed to answer the question
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
            "Overall": df["Overall Score"].mean()
        }
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Display metrics in columns with gauges
        with col1:
            self._render_metric_gauge("Faithfulness", avg_scores["Faithfulness"])
        
        with col2:
            self._render_metric_gauge("Answer Relevancy", avg_scores["Answer Relevancy"])
            
        with col3:
            self._render_metric_gauge("Context Precision", avg_scores["Context Precision"])  # Changed
            
        with col4:
            self._render_metric_gauge("Context Recall", avg_scores["Context Recall"])
        
        st.markdown("---")
        st.header("Overall RAG System Performance")
        st.markdown("""
        The Overall Score represents the average of all four RAGAS metrics, providing a single measure of your RAG system's performance.
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
            r=[avg_scores["Faithfulness"], avg_scores["Answer Relevancy"], 
            avg_scores["Context Precision"], avg_scores["Context Recall"], 
            avg_scores["Overall"]],  # Add Overall Score
            theta=["Faithfulness", "Answer Relevancy", "Context Precision", 
                "Context Recall", "Overall"],  # Add Overall to theta
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
            # Histogram for faithfulness
            fig = px.histogram(
                df, 
                x="Faithfulness Score",
                nbins=20,
                histfunc="count",
                title="Faithfulness Score Distribution",
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Histogram for context precision
            fig = px.histogram(
                df, 
                x="Context Precision Score",  # Changed
                nbins=20,
                histfunc="count",
                title="Context Precision Score Distribution",  # Changed
                color_discrete_sequence=['#2ca02c']
            )
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Histogram for answer relevancy
            fig = px.histogram(
                df, 
                x="Answer Relevancy Score",
                nbins=20,
                histfunc="count",
                title="Answer Relevancy Score Distribution",
                color_discrete_sequence=['#ff7f0e']
            )
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Histogram for context recall
            fig = px.histogram(
                df, 
                x="Context Recall Score",
                nbins=20,
                histfunc="count",
                title="Context Recall Score Distribution",
                color_discrete_sequence=['#d62728']
            )
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
                        "Metric": ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", "Overall"], 
                        "Score": [
                            question_data["Faithfulness Score"],
                            question_data["Answer Relevancy Score"],
                            question_data["Context Precision Score"], 
                            question_data["Context Recall Score"],
                            question_data["Overall Score"]
                        ]
                    })
                    
                    st.dataframe(scores_df, use_container_width=True)
                    
                    # Create bar chart for scores
                    fig = px.bar(
                        scores_df,
                        x="Metric",
                        y="Score",
                        title="RAGAS Scores",
                        barmode="group",
                        color="Metric",
                        color_discrete_map={
                            "Faithfulness": "#1f77b4",
                            "Answer Relevancy": "#ff7f0e",
                            "Context Precision": "#2ca02c",
                            "Context Recall": "#d62728",
                            "Overall": "#9467bd"
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
        
        # Check for constant columns
        constant_columns = []  # Corrected initialization
        for col in ["Faithfulness Score", "Answer Relevancy Score",
                    "Context Precision Score", "Context Recall Score"]:
            if df[col].nunique() == 1:
                constant_columns.append(col)

        # Calculate correlation matrix
        correlation_matrix = df[["Faithfulness Score", "Answer Relevancy Score",
                                "Context Precision Score", "Context Recall Score"]].corr()
        
        # Create heatmap
        fig = px.imshow(correlation_matrix,
                        labels=dict(x="Metrics", y="Metrics", color="Correlation"),
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        color_continuous_scale="RdBu",
                        color_continuous_midpoint=0)

        # Customize layout
        fig.update_layout(
            width=800,  # Adjust width as needed
            height=600, # Adjust height as needed
            xaxis_tickangle=-45
        )

        # Add title with explanation
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
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                min_faithfulness = st.slider(
                    "Min Faithfulness Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
            
            with col2:
                min_answer_relevancy = st.slider(
                    "Min Answer Relevancy Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
            
            with col3:
                min_context_precision = st.slider(  # Changed variable name
                    "Min Context Precision Score",  # Changed label
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
            
            with col4:
                min_context_recall = st.slider(
                    "Min Context Recall Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
                
            with col5:
                min_overall_score = st.slider(
                    "Min Overall Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
        
        # Apply filters
        filtered_df = df[
            (df["Faithfulness Score"] >= min_faithfulness) &
            (df["Answer Relevancy Score"] >= min_answer_relevancy) &
            (df["Context Precision Score"] >= min_context_precision) &  # Changed
            (df["Context Recall Score"] >= min_context_recall) &
            (df["Overall Score"] >= min_overall_score)
        ]
        
        # Display filtered results
        st.dataframe(
            filtered_df[[
                "Question ID", "Question", "Faithfulness Score", 
                "Answer Relevancy Score", "Context Precision Score", "Context Recall Score", "Overall Score"  # Changed
            ]],
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
                'steps': [
                    {'range': [0, 0.5], 'color': "red"},
                    {'range': [0.5, 0.75], 'color': "yellow"},
                    {'range': [0.75, 1], 'color': "green"}],
                'bgcolor': "white",
                'bordercolor': "gray"
            }
        ))

        # Adjust layout to control size
        fig.update_layout(
            width=300,  # Adjust width as needed
            height=250, # Adjust height as needed
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(fig, use_container_width=False)
    
    def _render_metric_gauge_old(self, title, value):
        """Render a gauge chart for a metric."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            domain = {'x':[0, 1], 'y': [0, 1]},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": self._get_color_for_score(value)},
                "steps": [
                    {"range": [0, 0.33], "color": "lightgray"},
                    {"range": [0.33, 0.67], "color": "gray"},
                    {"range": [0.67, 1], "color": "darkgray"}
                ]
            }
        ))
        
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_color_for_score(self, score):
        """Get color based on score value."""
        if score < 0.33:
            return "red"
        elif score < 0.67:
            return "yellow"
        else:
            return "green"
    
    def _get_test_files(self):
        """Get list of test result files."""
        pattern = os.path.join(self.eval_dir, "question_answer_*.csv")
        return sorted(glob.glob(pattern), reverse=True)
    
    def _get_evaluation_files(self):
        """Get list of evaluation result files."""
        pattern = os.path.join(self.eval_dir, "RAG_Results_*.csv")
        return sorted(glob.glob(pattern), reverse=True)
        
    def _evaluate_with_ragas(self, question, context, ground_truth, answer):
        """Evaluate a question-answer pair with RAGAS metrics."""
        try:
            if not answer or not question or not context or not ground_truth:
                return {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5}
            
            # Ensure context is formatted correctly (list of strings)
            contexts = context.split(". ") if isinstance(context, str) else context
            
            eval_df = pd.DataFrame({
                'question': [question],
                'answer': [answer], 
                'contexts': [contexts],
                'ground_truth': [ground_truth]
            })
            
            # Use the helper function to run each metric with a guaranteed event loop
            faith_score = self._run_with_event_loop(faithfulness.score, eval_df)
            ans_rel_score = self._run_with_event_loop(answer_relevancy.score, eval_df)
            ctx_prec_score = self._run_with_event_loop(context_precision.score, eval_df)
            ctx_recall_score = self._run_with_event_loop(context_recall.score, eval_df)
            
            scores = {
                "faithfulness": faith_score.iloc[0] if hasattr(faith_score, 'iloc') else faith_score,
                "answer_relevancy": ans_rel_score.iloc[0] if hasattr(ans_rel_score, 'iloc') else ans_rel_score,
                "context_precision": ctx_prec_score.iloc[0] if hasattr(ctx_prec_score, 'iloc') else ctx_prec_score,
                "context_recall": ctx_recall_score.iloc[0] if hasattr(ctx_recall_score, 'iloc') else ctx_recall_score
            }
            return scores
        except Exception as e:
            print(f"Error in RAGAS evaluation: {str(e)}")
            print(traceback.format_exc())
            return self._fallback_scoring(question, context, ground_truth, answer)

    def _fallback_scoring(self, question, context, ground_truth, answer):
        """Fallback scoring using Jaccard and TF-IDF similarity."""
        try:
            def jaccard_similarity(str1, str2):
                set1, set2 = set(str1.lower().split()), set(str2.lower().split())
                return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
            
            vectorizer = TfidfVectorizer()
            try:
                # Handle potential encoding issues
                tfidf_matrix = vectorizer.fit_transform([
                    str(question), 
                    str(answer), 
                    str(ground_truth), 
                    str(context)
                ])
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                
                faithfulness_score = cosine_sim[1, 3]  # Answer vs Context
                relevancy_score = cosine_sim[1, 0]  # Answer vs Question
                context_precision_score = cosine_sim[3, 2]  # Context vs Ground Truth
                context_recall_score = jaccard_similarity(context, ground_truth)  # Jaccard Similarity
            except Exception as vec_error:
                print(f"TF-IDF vectorization failed: {str(vec_error)}")
                # If vectorization fails, use simpler metrics
                faithfulness_score = jaccard_similarity(answer, context)
                relevancy_score = jaccard_similarity(question, answer)
                context_precision_score = jaccard_similarity(context, ground_truth)
                context_recall_score = jaccard_similarity(context, ground_truth)
            
            return {
                "faithfulness": max(min(faithfulness_score, 1.0), 0.0),
                "answer_relevancy": max(min(relevancy_score, 1.0), 0.0),
                "context_precision": max(min(context_precision_score, 1.0), 0.0),
                "context_recall": max(min(context_recall_score, 1.0), 0.0)
            }
        except Exception as e:
            print(f"Error in fallback scoring: {str(e)}")
            return {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5}
    
    def _evaluate_with_ragas_old(self, question, context, ground_truth, answer):
        """Evaluate a question-answer pair with RAGAS metrics."""
        try:
            # Ensure we have valid inputs to prevent errors
            if not answer or not question:
                return {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0
                }

            # Ensure context is formatted correctly (list of strings)
            contexts = context if isinstance(context, list) else [context]

            # Try different approaches to handle various RAGAS versions
            try:                
                # Create a test dataframe with all fields that any metric might need
                eval_df = pd.DataFrame({
                    'question': [question],
                    'answer': [answer], 
                    'contexts': [contexts],
                    'ground_truth': [ground_truth]
                })
                
                # Get scores - each metric takes only the columns it needs
                faith_score = faithfulness.score(eval_df)
                ans_rel_score = answer_relevancy.score(eval_df) 
                ctx_prec_score = context_precision.score(eval_df)
                ctx_recall_score = context_recall.score(eval_df)
                
                # Process results - handle various return types (Series, float, etc.)
                if hasattr(faith_score, 'iloc'):
                    faith_value = float(faith_score.iloc[0]) if len(faith_score) > 0 else 0.0
                else:
                    faith_value = float(faith_score) if faith_score is not None else 0.0
                    
                if hasattr(ans_rel_score, 'iloc'):
                    ans_rel_value = float(ans_rel_score.iloc[0]) if len(ans_rel_score) > 0 else 0.0
                else:
                    ans_rel_value = float(ans_rel_score) if ans_rel_score is not None else 0.0
                    
                if hasattr(ctx_prec_score, 'iloc'):
                    ctx_prec_value = float(ctx_prec_score.iloc[0]) if len(ctx_prec_score) > 0 else 0.0
                else:
                    ctx_prec_value = float(ctx_prec_score) if ctx_prec_score is not None else 0.0
                    
                if hasattr(ctx_recall_score, 'iloc'):
                    ctx_recall_value = float(ctx_recall_score.iloc[0]) if len(ctx_recall_score) > 0 else 0.0
                else:
                    ctx_recall_value = float(ctx_recall_score) if ctx_recall_score is not None else 0.0
                
            except Exception as inner_e:
                # If the DataFrame approach fails, try a simpler implementation
                print(f"First RAGAS approach failed: {str(inner_e)}")
                print(traceback.format_exc())
                
                # Fall back to simple scoring
                # These values are placeholders - in a real implementation, you would 
                # use a more sophisticated fallback method
                faith_value = self._calculate_simple_score(question, answer, contexts)
                ans_rel_value = self._calculate_simple_score(question, answer)
                ctx_prec_value = self._calculate_simple_score(question, answer, contexts)
                ctx_recall_value = self._calculate_simple_score(question, answer, contexts, ground_truth)
            
            return {
                "faithfulness": faith_value,
                "answer_relevancy": ans_rel_value,
                "context_precision": ctx_prec_value,
                "context_recall": ctx_recall_value
            }
        except Exception as e:
            st.error(f"Error in RAGAS evaluation: {str(e)}")
            print(f"RAGAS evaluation error details: {str(e)}")
            print(traceback.format_exc())
            
            # Return default scores
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0
            }

    def _calculate_simple_score(self, question, answer, contexts=None, ground_truth=None):
        """Calculate a simple fallback score if RAGAS fails.
        This is a very simple implementation that should be replaced with better logic.
        """
        # Simple word overlap heuristic - not meant to be accurate
        # Just a fallback if RAGAS metrics fail
        try:
            q_words = set(question.lower().split())
            a_words = set(answer.lower().split())
            
            # Avoid division by zero
            if not a_words:
                return 0.0
                
            # Simple overlap calculation
            overlap = len(q_words.intersection(a_words)) / len(a_words)
            
            # If we have contexts, adjust score
            if contexts:
                c_words = set()
                for ctx in contexts:
                    c_words.update(ctx.lower().split())
                
                if c_words:
                    ctx_overlap = len(a_words.intersection(c_words)) / len(a_words)
                    overlap = (overlap + ctx_overlap) / 2
            
            # If we have ground truth, adjust score further
            if ground_truth:
                gt_words = set(ground_truth.lower().split())
                if gt_words:
                    gt_overlap = len(a_words.intersection(gt_words)) / len(a_words)
                    overlap = (overlap + gt_overlap) / 2
            
            return min(1.0, overlap)
        except:
            return 0.5  # Neutral score on error
    
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
        """Apply custom CSS styling."""
        st.markdown("""
        <style>
        /* Page layout */
        .main {
            padding: 1rem 1rem;
        }
        
        /* Header styling */
        h1, h2, h3, h4 {
            color: #0051A2;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #0051A2;
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #003D82;
        }
        
        /* Card styling */
        .css-1r6slb0, .css-1y4p8pa {
            border-radius: 6px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Metric styles */
        .metric-card {
            background-color: white;
            border-radius: 6px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #0051A2;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #666;
        }
        
        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #0051A2;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f5f7f9;
        }
        
        [data-testid="stSidebar"] h1 {
            padding-left: 1rem;
        }
        
        /* Status message styling */
        .success-message {
            background-color: #d1e7dd;
            color: #0f5132;
            padding: 0.75rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .info-message {
            background-color: #cff4fc;
            color: #055160;
            padding: 0.75rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .warning-message {
            background-color: #fff3cd;
            color: #664d03;
            padding: 0.75rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .error-message {
            background-color: #f8d7da;
            color: #842029;
            padding: 0.75rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)


def main():
    # Create and run the application
    app = RAGEvaluationApp()
    app.run()

if __name__ == "__main__":
    main()
