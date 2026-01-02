import boto3
import json

# Initialize the Bedrock Agent Runtime client
bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name='us-east-1'  # Change to your region
)

def retrieve_and_generate_with_sonnet_45(query, knowledge_base_id, region='us-east-1'):
    """
    Query a knowledge base using Claude Sonnet 4.5
    
    Args:
        query: The question to ask
        knowledge_base_id: Your knowledge base ID
        region: AWS region (default: us-east-1)
    
    Returns:
        Generated response and citations
    """
    
    # Use inference profile ARN for Claude Sonnet 4.5
    # Option 1: Regional inference profile (recommended for production)
    model_arn = f'arn:aws:bedrock:{region}::inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    
    # Option 2: Global inference profile (for automatic cross-region routing)
    # model_arn = f'arn:aws:bedrock:{region}::inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0'
    
    try:
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={
                'text': query
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': model_arn,
                    'generationConfiguration': {
                        'inferenceConfig': {
                            'textInferenceConfig': {
                                'maxTokens': 4096,
                                'temperature': 0.7,
                                'topP': 0.9
                            }
                        }
                    },
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 5,  # Number of chunks to retrieve
                            'overrideSearchType': 'SEMANTIC'  # or 'HYBRID'
                        }
                    }
                }
            }
        )
        
        return response
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Replace with your actual knowledge base ID
    KB_ID = "YOUR_KNOWLEDGE_BASE_ID"
    
    # Your query
    question = "What is AWS Lambda and how does it work?"
    
    # Call the function
    result = retrieve_and_generate_with_sonnet_45(
        query=question,
        knowledge_base_id=KB_ID
    )
    
    # Extract and print the response
    generated_text = result['output']['text']
    print("Generated Response:")
    print(generated_text)
    print("\n" + "="*80 + "\n")
    
    # Print citations
    if 'citations' in result:
        print("Citations:")
        for i, citation in enumerate(result['citations'], 1):
            print(f"\nCitation {i}:")
            if 'retrievedReferences' in citation:
                for ref in citation['retrievedReferences']:
                    print(f"  Source: {ref.get('location', {}).get('s3Location', {}).get('uri', 'N/A')}")
                    print(f"  Content: {ref.get('content', {}).get('text', 'N/A')[:200]}...")
    
    # Print session ID for multi-turn conversations
    print(f"\nSession ID: {result.get('sessionId', 'N/A')}")
