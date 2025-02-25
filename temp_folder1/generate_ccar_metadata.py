"""
Command-line tool to generate metadata for CCAR CSV files.

This script generates a metadata file containing information about all CCAR CSV files
in a specified S3 bucket and prefix. The metadata file is saved to S3 and can optionally
be added to a Bedrock Knowledge Base.

Usage:
    python generate_ccar_metadata.py --bucket BUCKET_NAME --prefix CCAR_PREFIX --metadata-prefix METADATA_PREFIX [--kb-id KB_ID --source-id SOURCE_ID]

Example:
    python generate_ccar_metadata.py --bucket myawstests3buckets1 --prefix ccar_reports/dev/ccar --metadata-prefix metadata --kb-id MEGMDJFEJS --source-id XFKUEMGLCM
"""

import argparse
import logging
import sys
import boto3
import json
import os
from metadata_generator import MetadataGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('generate_ccar_metadata')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate metadata for CCAR CSV files')
    
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', required=True, help='S3 prefix for CCAR data files')
    parser.add_argument('--metadata-prefix', default='metadata', help='S3 prefix for metadata file')
    parser.add_argument('--metadata-filename', default='ccar_data_metadata.json', help='Metadata filename')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of concurrent workers')
    parser.add_argument('--sample-limit', type=int, default=10, help='Maximum number of sample values per column')
    parser.add_argument('--kb-id', help='Knowledge Base ID (optional)')
    parser.add_argument('--source-id', help='Data Source ID (optional)')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Create metadata generator
        generator = MetadataGenerator(
            bucket_name=args.bucket,
            ccar_prefix=args.prefix,
            metadata_prefix=args.metadata_prefix,
            metadata_filename=args.metadata_filename
        )
        
        # Generate metadata
        logger.info(f"Generating metadata for CCAR files in s3://{args.bucket}/{args.prefix}")
        metadata = generator.generate_metadata(
            max_workers=args.max_workers,
            sample_limit=args.sample_limit
        )
        
        # Print summary statistics
        logger.info(f"Generated metadata for {len(metadata)} files")
        
        # Get file counts by scenario
        scenarios = {}
        for file_path, file_metadata in metadata.items():
            scenario = file_metadata.get('scenario', 'unknown')
            scenarios[scenario] = scenarios.get(scenario, 0) + 1
        
        logger.info("File counts by scenario:")
        for scenario, count in scenarios.items():
            logger.info(f"  {scenario}: {count} files")
        
        # Add to Knowledge Base if IDs are provided
        #if args.kb_id and args.source_id:
        #    logger.info(f"Adding metadata to Knowledge Base {args.kb_id}, Data Source {args.source_id}")
        #    job_id = generator.add_to_knowledge_base(kb_id=args.kb_id, source_id=args.source_id)
        #    logger.info(f"Started ingestion job {job_id}")
        
        #logger.info("Metadata generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error generating metadata: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()