"""
Generic metadata generator utility for structured data files.

This script generates a metadata file containing information about all data files
in a specified S3 bucket and prefix. The metadata file is saved to S3 and can optionally
be added to a Bedrock Knowledge Base.

Usage:
    python generate_metadata.py --program PROGRAM_ID --config CONFIG_FILE [--force]

Example:
    python generate_metadata.py --program CCAR --config config/common_config.json --force
"""

import argparse
import logging
import sys
import boto3
import json
import os
import time
import polars as pl
import s3fs
from typing import Dict, List, Set, Any
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('metadata_generator')

class MetadataGenerator:
    """Generic utility class to generate metadata for structured data files."""
    
    def __init__(self, bucket_name: str, data_prefix: str, metadata_prefix: str, metadata_filename: str):
        """
        Initialize the metadata generator.
        
        Args:
            bucket_name: S3 bucket name
            data_prefix: Prefix for data files
            metadata_prefix: Prefix for storing metadata file
            metadata_filename: Name of the metadata file
        """
        self.bucket_name = bucket_name
        self.data_prefix = data_prefix
        self.metadata_prefix = metadata_prefix
        self.metadata_filename = metadata_filename
        self.s3_client = boto3.client('s3')
        self.s3_fs = s3fs.S3FileSystem()
        
    def generate_metadata(self, max_workers: int = 5, sample_limit: int = 10, file_extensions: List[str] = ['.csv']) -> Dict:
        """
        Generate metadata for all data files.
        
        Args:
            max_workers: Maximum number of concurrent workers for file processing
            sample_limit: Maximum number of sample values to include for each column
            file_extensions: List of file extensions to process
            
        Returns:
            Dict: The complete metadata
        """
        try:
            logger.info(f"Starting metadata generation for files in s3://{self.bucket_name}/{self.data_prefix}")
            
            # Step 1: List all data files with supported extensions
            data_files = self._list_data_files(file_extensions)
            logger.info(f"Found {len(data_files)} files to process")
            
            # Step 2: Process files in parallel to extract metadata
            metadata = {}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self._process_file, file_path, sample_limit): file_path 
                    for file_path in data_files
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
                            logger.info(f"Processed metadata for {s3_uri}")
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")
            
            # Step 3: Save metadata to S3
            self._save_metadata(metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            raise
    
    def _list_data_files(self, file_extensions: List[str]) -> List[str]:
        """
        List all data files with the specified extensions in the data prefix.
        
        Args:
            file_extensions: List of file extensions to include
            
        Returns:
            List[str]: List of file paths
        """
        data_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.data_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Check if the file has one of the supported extensions
                    if any(key.lower().endswith(ext.lower()) for ext in file_extensions):
                        data_files.append(key)
        
        return data_files
    
    def _process_file(self, file_path: str, sample_limit: int) -> Dict:
        """
        Process a single file to extract metadata.
        
        Args:
            file_path: S3 path to the file
            sample_limit: Maximum number of sample values to include
            
        Returns:
            Dict: Metadata for the file
        """
        try:
            # Extract path components
            path_parts = file_path.split('/')
            file_name = path_parts[-1]
            file_extension = os.path.splitext(file_name)[1].lower()
            
            # Extract path components as metadata
            metadata = self._extract_path_components(file_path)
            
            # Set basic file information
            metadata.update({
                'file_name': file_name,
                'file_type': file_extension.replace('.', ''),
                's3_file_path': f"s3://{self.bucket_name}/{file_path}"
            })
            
            # Process based on file type
            if file_extension.lower() == '.csv':
                metadata = self._process_csv_file(file_path, metadata, sample_limit)
            elif file_extension.lower() in ['.json', '.jsonl']:
                metadata = self._process_json_file(file_path, metadata, sample_limit)
            elif file_extension.lower() in ['.parquet']:
                metadata = self._process_parquet_file(file_path, metadata, sample_limit)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _process_csv_file(self, file_path: str, metadata: Dict, sample_limit: int) -> Dict:
        """
        Process a CSV file to extract metadata.
        
        Args:
            file_path: S3 path to the CSV file
            metadata: Existing metadata dictionary
            sample_limit: Maximum number of sample values
            
        Returns:
            Dict: Enhanced metadata
        """
        s3_path = f"s3://{self.bucket_name}/{file_path}"
        
        # Determine delimiter based on file extension or existing metadata
        delimiter = metadata.get('delimiter', ',')
        if file_path.endswith('.psv') or file_path.endswith('.pipe'):
            delimiter = '|'
        elif file_path.endswith('.tsv'):
            delimiter = '\t'
        
        # Update metadata with delimiter
        metadata['delimiter'] = delimiter
        
        try:
            # Use Polars lazy API for efficiency
            df = pl.scan_csv(
                s3_path,
                separator=delimiter,
                infer_schema_length=1000,
                n_rows=1000  # Limit rows for schema inference
            ).collect()
        except Exception as e:
            logger.warning(f"Lazy loading failed for {s3_path}, falling back to eager loading: {str(e)}")
            try:
                # Fallback to eager loading with sample
                with self.s3_fs.open(s3_path, 'rb') as f:
                    df = pl.read_csv(f, separator=delimiter, infer_schema_length=1000, n_rows=1000)
            except Exception as e2:
                logger.error(f"Failed to read CSV file {s3_path}: {str(e2)}")
                # Add basic metadata without column information
                metadata['error'] = f"Failed to read file: {str(e2)}"
                return metadata
        
        # Extract column metadata
        columns = {}
        for col_name in df.columns:
            try:
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
            except Exception as col_err:
                # Skip problematic columns but log the error
                logger.warning(f"Error processing column {col_name} in {s3_path}: {str(col_err)}")
        
        metadata['columns'] = columns
        return metadata

    def _process_json_file(self, file_path: str, metadata: Dict, sample_limit: int) -> Dict:
        """
        Process a JSON file to extract metadata.
        
        Args:
            file_path: S3 path to the JSON file
            metadata: Existing metadata dictionary
            sample_limit: Maximum number of sample values
            
        Returns:
            Dict: Enhanced metadata
        """
        s3_path = f"s3://{self.bucket_name}/{file_path}"
        
        try:
            # For JSON files, read a sample to determine structure
            with self.s3_fs.open(s3_path, 'rb') as f:
                # Try to detect if it's JSON Lines format
                sample = f.read(10000).decode('utf-8').strip()
                
                # Check if it's JSON Lines (one JSON object per line)
                if sample.startswith('{') and '\n{' in sample:
                    metadata['format'] = 'jsonl'
                    # Read with Polars as JSON Lines
                    df = pl.read_json(s3_path, lines=True, n_rows=1000)
                else:
                    metadata['format'] = 'json'
                    # Read regular JSON (assumes an array of objects)
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        # Convert to DataFrame if it's an array of objects
                        df = pl.from_dicts(data[:1000])
                    else:
                        # Not a supported JSON structure
                        metadata['error'] = "Unsupported JSON structure (not an array of objects)"
                        return metadata
            
            # Extract column metadata
            columns = {}
            for col_name in df.columns:
                try:
                    col_values = df[col_name].unique()
                    
                    # Determine data type
                    if df[col_name].dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.Float64, pl.Float32]:
                        data_type = 'float' if df[col_name].dtype.is_float() else 'int'
                        sample_values = col_values.head(min(5, len(col_values))).to_list()
                    else:
                        data_type = 'string'
                        sample_values = col_values.head(min(sample_limit, len(col_values))).to_list()
                    
                    # Make all sample values JSON serializable
                    sample_values = [str(val) if val is not None else None for val in sample_values]
                    
                    columns[col_name] = {
                        'sample_values': sample_values,
                        'data_type': data_type
                    }
                except Exception as col_err:
                    # Skip problematic columns but log the error
                    logger.warning(f"Error processing column {col_name} in {s3_path}: {str(col_err)}")
            
            metadata['columns'] = columns
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to read JSON file {s3_path}: {str(e)}")
            metadata['error'] = f"Failed to read file: {str(e)}"
            return metadata

    def _process_parquet_file(self, file_path: str, metadata: Dict, sample_limit: int) -> Dict:
        """
        Process a Parquet file to extract metadata.
        
        Args:
            file_path: S3 path to the Parquet file
            metadata: Existing metadata dictionary
            sample_limit: Maximum number of sample values
            
        Returns:
            Dict: Enhanced metadata
        """
        s3_path = f"s3://{self.bucket_name}/{file_path}"
        
        try:
            # Use Polars lazy API for efficiency with Parquet
            df = pl.scan_parquet(
                s3_path,
                n_rows=1000  # Limit rows for schema inference
            ).collect()
            
            # Extract column metadata
            columns = {}
            for col_name in df.columns:
                try:
                    col_values = df[col_name].unique()
                    
                    # Determine data type
                    if df[col_name].dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.Float64, pl.Float32]:
                        data_type = 'float' if df[col_name].dtype.is_float() else 'int'
                        sample_values = col_values.head(min(5, len(col_values))).to_list()
                    else:
                        data_type = 'string'
                        sample_values = col_values.head(min(sample_limit, len(col_values))).to_list()
                    
                    # Make all sample values JSON serializable
                    sample_values = [str(val) if val is not None else None for val in sample_values]
                    
                    columns[col_name] = {
                        'sample_values': sample_values,
                        'data_type': data_type
                    }
                except Exception as col_err:
                    # Skip problematic columns but log the error
                    logger.warning(f"Error processing column {col_name} in {s3_path}: {str(col_err)}")
            
            metadata['columns'] = columns
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to read Parquet file {s3_path}: {str(e)}")
            metadata['error'] = f"Failed to read file: {str(e)}"
            return metadata
    
    def _extract_path_components(self, file_path: str) -> Dict[str, str]:
        """
        Extract metadata components from file path.
        
        Args:
            file_path: S3 path to the file
            
        Returns:
            Dict: Extracted metadata components
        """
        # Extract meaningful components from the path
        parts = file_path.split('/')
        metadata = {}
        
        # Remove the file name and the base prefix from consideration
        data_prefix_parts = self.data_prefix.split('/')
        path_parts = parts[len(data_prefix_parts):-1]
        
        # Map each path component to a metadata field
        # This is generic and can be customized for specific path structures
        if len(path_parts) >= 1:
            metadata['level_1'] = path_parts[0]
        if len(path_parts) >= 2:
            metadata['level_2'] = path_parts[1]
        if len(path_parts) >= 3:
            metadata['level_3'] = path_parts[2]
        if len(path_parts) >= 4:
            metadata['level_4'] = path_parts[3]
            
        # Add special handling for common path patterns
        # Look for components that match specific patterns
        for part in path_parts:
            # Example: detect version numbers, dates, run IDs
            if part.startswith('v') and part[1:].isdigit():
                metadata['version'] = part
            elif part.startswith('run') and part[3:].isdigit():
                metadata['run_id'] = part
            elif part in ['input', 'output', 'intermediate']:
                metadata['stage'] = part
            elif part in ['baseline', 'adverse', 'severely_adverse']:
                metadata['scenario'] = part
            elif part.startswith('cycle') and part[5:].isdigit():
                metadata['cycle'] = part
                
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
            
            logger.info(f"Metadata saved to s3://{self.bucket_name}/{metadata_key}")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise

    def add_to_knowledge_base(self, kb_id: str, source_id: str) -> str:
        """
        Add the metadata file to a Bedrock Knowledge Base.
        
        Args:
            kb_id: Knowledge Base ID
            source_id: Data Source ID
            
        Returns:
            str: Ingestion job ID
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
                description=f"Ingest data metadata file: {self.metadata_filename}",
                documentFilter={
                    "source": s3_uri
                }
            )
            
            job_id = response.get('ingestionJobId')
            logger.info(f"Started ingestion job {job_id} for metadata file {s3_uri}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error adding metadata to Knowledge Base: {str(e)}")
            raise

def load_config(config_file: str) -> Dict:
    """Load configuration from file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        sys.exit(1)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate metadata for structured data files')
    
    # Required arguments
    parser.add_argument('--program', required=True, help='Program ID from config')
    parser.add_argument('--config', required=True, help='Path to config file')
    
    # Optional arguments
    parser.add_argument('--force', action='store_true', help='Force metadata regeneration')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of concurrent workers')
    parser.add_argument('--sample-limit', type=int, default=10, help='Maximum number of sample values per column')
    parser.add_argument('--extensions', nargs='+', default=['.csv', '.json', '.jsonl', '.parquet'], 
                       help='File extensions to process')
    parser.add_argument('--ingest', action='store_true', help='Add metadata to Knowledge Base after generation')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Get program configuration
        if args.program not in config.get('programs', {}):
            logger.error(f"Program '{args.program}' not found in configuration")
            sys.exit(1)
            
        program_config = config['programs'][args.program]
        
        # Get metadata configuration
        metadata_config = program_config.get('metadata_config', {})
        bucket = metadata_config.get('bucket', config.get('s3_config', {}).get('bucket'))
        data_prefix = metadata_config.get('data_prefix')
        metadata_prefix = metadata_config.get('prefix', 'metadata')
        metadata_filename = metadata_config.get('filename', f"{args.program}_data_metadata.json")
        
        if not bucket:
            logger.error("S3 bucket not specified in configuration")
            sys.exit(1)
            
        if not data_prefix:
            logger.error("Data prefix not specified in metadata configuration")
            sys.exit(1)
        
        # Check if metadata file already exists
        s3_client = boto3.client('s3')
        metadata_key = f"{metadata_prefix}/{metadata_filename}"
        
        try:
            if not args.force:
                # Check if metadata file already exists
                s3_client.head_object(Bucket=bucket, Key=metadata_key)
                logger.info(f"Metadata file already exists: s3://{bucket}/{metadata_key}")
                
                # Get last modified time
                response = s3_client.get_object(Bucket=bucket, Key=metadata_key)
                last_modified = response['LastModified']
                
                logger.info(f"Last modified: {last_modified}")
                logger.info("Use --force to regenerate metadata")
                sys.exit(0)
        except s3_client.exceptions.ClientError:
            # File doesn't exist, continue with generation
            pass
        
        # Create metadata generator
        generator = MetadataGenerator(
            bucket_name=bucket,
            data_prefix=data_prefix,
            metadata_prefix=metadata_prefix,
            metadata_filename=metadata_filename
        )
        
        # Generate metadata
        logger.info(f"Generating metadata for {args.program} files in s3://{bucket}/{data_prefix}")
        metadata = generator.generate_metadata(
            max_workers=args.max_workers,
            sample_limit=args.sample_limit,
            file_extensions=args.extensions
        )
        
        # Get file counts by level
        level_counts = {}
        for file_path, file_metadata in metadata.items():
            for level in ['level_1', 'level_2', 'level_3', 'level_4']:
                if level in file_metadata:
                    level_value = file_metadata[level]
                    if level not in level_counts:
                        level_counts[level] = {}
                    level_counts[level][level_value] = level_counts[level].get(level_value, 0) + 1
        
        logger.info("Generated metadata summary:")
        logger.info(f"  Total files: {len(metadata)}")
        for level, counts in level_counts.items():
            logger.info(f"  {level.replace('_', ' ').title()} counts:")
            for value, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:  # Show top 10
                logger.info(f"    {value}: {count} files")
        
        # Add to Knowledge Base if requested
        if args.ingest:
            kb_id = program_config.get('knowledge_base_id')
            source_id = program_config.get('data_source_id')
            
            if kb_id and source_id:
                logger.info(f"Adding metadata to Knowledge Base {kb_id}, Data Source {source_id}")
                job_id = generator.add_to_knowledge_base(kb_id=kb_id, source_id=source_id)
                logger.info(f"Started ingestion job {job_id}")
            else:
                logger.warning("Knowledge Base ID or Data Source ID not specified, skipping ingestion")
        
        logger.info("Metadata generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error generating metadata: {str(e)}")
        for line in traceback.format_exc().splitlines():
            logger.error(line)
        sys.exit(1)

if __name__ == "__main__":
    main()