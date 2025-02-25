import boto3
import json
import os
import polars as pl
import s3fs
from typing import Dict, List, Set, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class MetadataGenerator:
    """A utility class to generate metadata for CCAR CSV files."""
    
    def __init__(self, bucket_name: str, ccar_prefix: str, metadata_prefix: str, metadata_filename: str):
        """
        Initialize the metadata generator.
        
        Args:
            bucket_name: S3 bucket name
            ccar_prefix: Prefix for CCAR data files
            metadata_prefix: Prefix for storing metadata file
            metadata_filename: Name of the metadata file
        """
        self.bucket_name = bucket_name
        self.ccar_prefix = ccar_prefix
        self.metadata_prefix = metadata_prefix
        self.metadata_filename = metadata_filename
        self.s3_client = boto3.client('s3')
        self.s3_fs = s3fs.S3FileSystem()
        
        # Set up logging
        self.logger = logging.getLogger('MetadataGenerator')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def generate_metadata(self, max_workers: int = 5, sample_limit: int = 10) -> Dict:
        """
        Generate metadata for all CCAR CSV files.
        
        Args:
            max_workers: Maximum number of concurrent workers for file processing
            sample_limit: Maximum number of sample values to include for each column
            
        Returns:
            Dict: The complete metadata
        """
        try:
            self.logger.info(f"Starting metadata generation for CCAR files in s3://{self.bucket_name}/{self.ccar_prefix}")
            
            # Step 1: List all CSV files
            csv_files = self._list_csv_files()
            self.logger.info(f"Found {len(csv_files)} CSV files to process")
            
            # Step 2: Process files in parallel to extract metadata
            metadata = {}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self._process_file, file_path, sample_limit): file_path 
                    for file_path in csv_files
                }
                
                # Process results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_metadata = future.result()
                        if file_metadata:
                            # Use the full S3 URI as the key
                            s3_uri = f"s3://{self.bucket_name}/{file_path}"
                            metadata[s3_uri] = file_metadata
                            self.logger.info(f"Processed metadata for {s3_uri}")
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {str(e)}")
            
            # Step 3: Save metadata to S3
            self._save_metadata(metadata)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error generating metadata: {str(e)}")
            raise
    
    def _list_csv_files(self) -> List[str]:
        """
        List all CSV files in the CCAR prefix.
        
        Returns:
            List[str]: List of file paths
        """
        csv_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.ccar_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.csv'):
                        csv_files.append(key)
        
        return csv_files
    
    def _process_file(self, file_path: str, sample_limit: int) -> Dict:
        """
        Process a single CSV file to extract metadata.
        
        Args:
            file_path: S3 path to the CSV file
            sample_limit: Maximum number of sample values to include
            
        Returns:
            Dict: Metadata for the file
        """
        try:
            # Extract path components
            path_parts = file_path.split('/')
            file_name = path_parts[-1]
            
            # Extract cycle, scenario, run_id, process
            metadata = self._extract_path_components(file_path)
            
            # Set basic file information
            metadata.update({
                'file_name': file_name,
                'file_type': 'csv',
                'delimiter': '|',
                's3_file_path': f"s3://{self.bucket_name}/{file_path}"
            })
            
            # Read file with Polars for better performance
            s3_path = f"s3://{self.bucket_name}/{file_path}"
            try:
                # First try to scan with Polars lazy API
                df = pl.scan_csv(
                    s3_path,
                    separator='|',
                    infer_schema_length=1000,
                    n_rows=1000  # Limit rows for schema inference
                ).collect()
            except Exception as e:
                # Fallback to eager loading if lazy fails
                self.logger.warning(f"Lazy loading failed for {s3_path}, falling back to eager loading: {str(e)}")
                with self.s3_fs.open(s3_path, 'rb') as f:
                    df = pl.read_csv(f, separator='|', infer_schema_length=1000, n_rows=1000)
            
            # Extract column metadata
            columns = {}
            for col_name in df.columns:
                col_values = df[col_name].unique()
                # Determine data type
                if df[col_name].dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.Float64, pl.Float32]:
                    data_type = 'float' if df[col_name].dtype.is_float() else 'int'
                    # For numeric columns, just take a few sample values
                    sample_values = col_values.head(min(5, len(col_values))).to_list()
                else:
                    data_type = 'string'
                    # For string columns, take more representative samples
                    sample_values = col_values.head(min(sample_limit, len(col_values))).to_list()
                
                # Make all sample values JSON serializable
                sample_values = [str(val) if val is not None else None for val in sample_values]
                
                columns[col_name] = {
                    'sample_values': sample_values,
                    'data_type': data_type
                }
            
            metadata['columns'] = columns
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _extract_path_components(self, file_path: str) -> Dict[str, str]:
        """
        Extract metadata components from file path.
        
        Args:
            file_path: S3 path to the file
            
        Returns:
            Dict: Extracted metadata components
        """
        # Expected path format: ccar_reports/dev/ccar/cycle3/internal_baseline/run001/ihc_balance_sheet/output/file.csv
        parts = file_path.split('/')
        metadata = {}
        
        # Extract components based on position
        for i, part in enumerate(parts):
            if 'cycle' in part:
                metadata['cycle'] = part
            elif part.startswith('internal_') or part.startswith('supervisory_'):
                metadata['scenario'] = part
            elif part.startswith('run'):
                metadata['run_id'] = part
            elif i > 0 and i < len(parts) - 2:  # Not first or last two components
                if part not in ['dev', 'ccar', 'output', 'input'] and not metadata.get('process'):
                    metadata['process'] = part
        
        return metadata
    
    def _save_metadata(self, metadata: Dict) -> None:
        """
        Save metadata to S3.
        
        Args:
            metadata: The complete metadata dictionary
        """
        metadata_key = f"{self.metadata_prefix}/{self.metadata_filename}"
        
        try:
            # Convert to JSON
            metadata_json = json.dumps(metadata, indent=2)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=metadata_json,
                ContentType='application/json'
            )
            
            self.logger.info(f"Metadata saved to s3://{self.bucket_name}/{metadata_key}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
            raise

    def add_to_knowledge_base(self, kb_id: str, source_id: str) -> None:
        """
        Add the metadata file to a Bedrock Knowledge Base.
        
        Args:
            kb_id: Knowledge Base ID
            source_id: Data Source ID
        """
        try:
            metadata_key = f"{self.metadata_prefix}/{self.metadata_filename}"
            s3_uri = f"s3://{self.bucket_name}/{metadata_key}"
            
            # Create Bedrock Agent Runtime client
            bedrock_agent = boto3.client('bedrock-agent')
            
            # Start ingestion job
            response = bedrock_agent.start_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=source_id,
                description=f"Ingest CCAR data metadata file: {self.metadata_filename}",
                documentFilter={
                    "source": s3_uri
                }
            )
            
            job_id = response.get('ingestionJobId')
            self.logger.info(f"Started ingestion job {job_id} for metadata file {s3_uri}")
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Error adding metadata to Knowledge Base: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    generator = MetadataGenerator(
        bucket_name="myawstests3buckets1",
        ccar_prefix="ccar_reports/dev/ccar",
        metadata_prefix="metadata",
        metadata_filename="ccar_data_metadata.json"
    )
    
    metadata = generator.generate_metadata()
    
    # Optionally add to Knowledge Base
    # generator.add_to_knowledge_base(kb_id="YOUR_KB_ID", source_id="YOUR_SOURCE_ID")