import os
import json
import glob
import re
import time
import argparse
from tqdm import tqdm
import pinecone
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
import uuid

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API")

# Configuration
DISCOURSE_DIR = "downloads/discourse_json"
DISCOURSE_SUMM_DIR = "downloads/discourse_summ"  # New directory for summarized QA pairs
TDS_PAGES_DIR = "downloads/tds_pages_md"
PINECONE_INDEX_NAME = "geton-rag"
PINECONE_DIMENSION = 1024  # llama-text-embed-v2 dimension
PINECONE_METRIC = "cosine"
PINECONE_ENVIRONMENT = "gcp-starter"  # Change if using a different environment
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PROGRESS_FILE = "rag_process_progress.json"  # File to track progress

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

# Progress tracking
def load_progress():
    """Load progress from progress file if it exists."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading progress file: {e}")
    
    # Return default progress object if file doesn't exist or there's an error
    return {
        "discourse_files_processed": [],
        "discourse_summ_files_processed": [],
        "tds_files_processed": [],
        "last_processed_document_id": None,
        "documents_uploaded": 0,
        "total_documents": 0,
        "completed": False
    }

def save_progress(progress):
    """Save progress to progress file."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"Error saving progress: {e}")

def extract_frontmatter(md_content):
    """Extract frontmatter from markdown content."""
    frontmatter_match = re.match(r"^---\n(.*?)\n---\n", md_content, re.DOTALL)
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        try:
            metadata = yaml.safe_load(frontmatter_text)
            content = md_content[frontmatter_match.end():]
            return metadata, content
        except Exception as e:
            print(f"Error parsing frontmatter: {e}")
    
    # Return empty metadata and original content if no frontmatter
    return {}, md_content

def process_discourse_files(progress=None):
    """Process discourse JSON files and return list of documents with metadata."""
    if progress is None:
        progress = {"discourse_files_processed": []}
    
    print("Processing Discourse JSON files...")
    all_documents = []
    
    # Get all JSON files in the discourse directory
    json_files = glob.glob(os.path.join(DISCOURSE_DIR, "*.json"))
    
    # Filter out already processed files
    files_to_process = [f for f in json_files if os.path.basename(f) not in progress["discourse_files_processed"]]
    
    if len(files_to_process) < len(json_files):
        print(f"Skipping {len(json_files) - len(files_to_process)} already processed discourse files")
    
    for json_file in tqdm(files_to_process, desc="Processing Discourse files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                topic_data = json.load(f)
            
            # Extract topic metadata
            topic_id = topic_data.get('id')
            topic_title = topic_data.get('title', '')
            
            # Properly format the topic URL with the base Discourse URL
            discourse_base_url = "https://discourse.onlinedegree.iitm.ac.in"
            topic_slug = topic_data.get('slug', '')
            topic_url = f"{discourse_base_url}/t/{topic_slug}/{topic_id}"
            
            # Process each post in the topic
            if 'post_stream' in topic_data and 'posts' in topic_data['post_stream']:
                for post in topic_data['post_stream']['posts']:
                    post_id = post.get('id')
                    post_number = post.get('post_number')
                    post_content = post.get('cooked', '')  # HTML content
                    
                    # Remove HTML tags for plain text
                    post_text = re.sub(r'<[^>]+>', ' ', post_content)
                    post_text = re.sub(r'\s+', ' ', post_text).strip()
                    
                    # Skip empty posts
                    if not post_text:
                        continue
                    
                    # Create post URL with proper format
                    post_url = f"{topic_url}/{post_number}"
                    
                    # Split post into chunks
                    chunks = text_splitter.split_text(post_text)
                    
                    # Create document for each chunk with metadata
                    for i, chunk in enumerate(chunks):
                        # Ensure text doesn't exceed token limits (approx 2048 tokens = ~8000 chars)
                        if len(chunk) > 8000:
                            chunk = chunk[:8000]
                            
                        # Flatten metadata as simple key-value pairs
                        doc_id = f"discourse_{topic_id}_{post_id}_{i}"
                        
                        all_documents.append({
                            "id": doc_id,
                            "text": chunk,
                            # Metadata as simple key-value pairs (no nested objects)
                            "source": "discourse",
                            "topic_id": str(topic_id),
                            "topic_title": topic_title,
                            "post_id": str(post_id),
                            "post_number": str(post_number),
                            "chunk_index": str(i),
                            "url": post_url,
                            "created_at": post.get('created_at', ""),
                            "username": post.get('username', "")
                        })
            
            # Mark this file as processed
            progress["discourse_files_processed"].append(os.path.basename(json_file))
            save_progress(progress)
        
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
    
    print(f"Processed {len(all_documents)} discourse document chunks")
    return all_documents

def process_discourse_summ_files(progress=None):
    """Process summarized discourse JSON files with QA pairs."""
    if progress is None:
        progress = {"discourse_summ_files_processed": []}
    
    print("Processing summarized Discourse QA pairs...")
    all_documents = []
    
    # Get all JSON files in the discourse_summ directory
    json_files = glob.glob(os.path.join(DISCOURSE_SUMM_DIR, "*.json"))
    
    # Filter out already processed files
    files_to_process = [f for f in json_files if os.path.basename(f) not in progress["discourse_summ_files_processed"]]
    
    if len(files_to_process) < len(json_files):
        print(f"Skipping {len(json_files) - len(files_to_process)} already processed QA files")
    
    for json_file in tqdm(files_to_process, desc="Processing QA pairs"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            # Extract topic metadata
            topic_id = qa_data.get('topic_id')
            topic_title = qa_data.get('topic_title', '')
            topic_url = qa_data.get('topic_url', '')
            
            # Make sure we have a valid URL
            if not topic_url or topic_url == "/":
                discourse_base_url = "https://discourse.onlinedegree.iitm.ac.in"
                topic_slug = qa_data.get('topic_slug', '') or topic_title.lower().replace(' ', '-')
                topic_url = f"{discourse_base_url}/t/{topic_slug}/{topic_id}"
            
            # Process each QA pair
            qa_pairs = qa_data.get('qa_pairs', [])
            for i, qa_pair in enumerate(qa_pairs):
                question = qa_pair.get('question', '')
                answer = qa_pair.get('answer', '')
                post_id = qa_pair.get('post_id')
                post_number = qa_pair.get('post_number')
                username = qa_pair.get('username', '')
                created_at = qa_pair.get('created_at', '')
                
                # Skip empty pairs
                if not question or not answer:
                    continue
                
                # Create post URL
                post_url = f"{topic_url}/{post_number}" if post_number else topic_url
                
                # Create document for the question
                question_id = f"qa_question_{topic_id}_{post_id}_{i}"
                all_documents.append({
                    "id": question_id,
                    "text": question,
                    "source": "discourse_qa",
                    "content_type": "question",
                    "topic_id": str(topic_id),
                    "topic_title": topic_title,
                    "post_id": str(post_id) if post_id else "",
                    "post_number": str(post_number) if post_number else "",
                    "qa_pair_index": str(i),
                    "url": post_url,
                    "created_at": created_at,
                    "username": username
                })
                
                # Create document for the answer
                answer_id = f"qa_answer_{topic_id}_{post_id}_{i}"
                all_documents.append({
                    "id": answer_id,
                    "text": answer,
                    "source": "discourse_qa",
                    "content_type": "answer",
                    "topic_id": str(topic_id),
                    "topic_title": topic_title,
                    "post_id": str(post_id) if post_id else "",
                    "post_number": str(post_number) if post_number else "",
                    "qa_pair_index": str(i),
                    "url": post_url,
                    "created_at": created_at,
                    "username": username,
                    "question_id": question_id  # Reference to the corresponding question
                })
            
            # Mark this file as processed
            progress["discourse_summ_files_processed"].append(os.path.basename(json_file))
            save_progress(progress)
        
        except Exception as e:
            print(f"Error processing QA file {json_file}: {e}")
    
    print(f"Processed {len(all_documents)} QA documents")
    return all_documents

def process_tds_md_files(progress=None):
    """Process TDS markdown files and return list of documents with metadata."""
    if progress is None:
        progress = {"tds_files_processed": []}
    
    print("Processing TDS markdown files...")
    all_documents = []
    
    # Get all MD files in the TDS pages directory
    md_files = glob.glob(os.path.join(TDS_PAGES_DIR, "*.md"))
    
    # Filter out already processed files
    files_to_process = [f for f in md_files if os.path.basename(f) not in progress["tds_files_processed"]]
    
    if len(files_to_process) < len(md_files):
        print(f"Skipping {len(md_files) - len(files_to_process)} already processed TDS files")
    
    for md_file in tqdm(files_to_process, desc="Processing TDS files"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract frontmatter metadata and content
            metadata, md_content = extract_frontmatter(content)
            
            # Skip metadata.json as it's not a content file
            if os.path.basename(md_file) == "metadata.json":
                continue
            
            # Split content into chunks
            chunks = text_splitter.split_text(md_content)
            
            # Create document for each chunk with metadata
            filename = os.path.basename(md_file)
            for i, chunk in enumerate(chunks):
                # Ensure text doesn't exceed token limits (approx 2048 tokens = ~8000 chars)
                if len(chunk) > 8000:
                    chunk = chunk[:8000]
                    
                doc_id = f"tds_{filename.replace('.', '_')}_{i}"
                
                all_documents.append({
                    "id": doc_id,
                    "text": chunk,
                    # Metadata as simple key-value pairs (no nested objects)
                    "source": "tds_website",
                    "title": metadata.get("title", filename),
                    "original_url": metadata.get("original_url", ""),
                    "downloaded_at": metadata.get("downloaded_at", ""),
                    "filename": filename,
                    "chunk_index": str(i)
                })
            
            # Mark this file as processed
            progress["tds_files_processed"].append(os.path.basename(md_file))
            save_progress(progress)
        
        except Exception as e:
            print(f"Error processing file {md_file}: {e}")
    
    print(f"Processed {len(all_documents)} TDS document chunks")
    return all_documents

def init_pinecone():
    """Initialize Pinecone with llama-text-embed-v2 model."""
    print("Initializing Pinecone...")
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index with llama-text-embed-v2: {PINECONE_INDEX_NAME}")
        pc.create_index_for_model(
            name=PINECONE_INDEX_NAME,
            cloud="aws",  # or "gcp" depending on your preference
            region="us-east-1",  # choose appropriate region
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {
                    "text": "text"  # Map the record field to be embedded
                }
            }
        )
    else:
        print(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")
    
    # Connect to the index
    index = pc.Index(PINECONE_INDEX_NAME)
    return index

def upload_to_pinecone(index, documents, progress=None):
    """Upload documents to Pinecone index using llama-text-embed-v2 with rate limiting."""
    if progress is None:
        progress = {"documents_uploaded": 0, "last_processed_document_id": None}
    
    total_docs = len(documents)
    documents_uploaded = progress["documents_uploaded"]
    
    # Skip already uploaded documents
    if documents_uploaded > 0:
        print(f"Resuming upload from document {documents_uploaded}/{total_docs}")
        documents = documents[documents_uploaded:]
    
    print(f"Uploading {len(documents)} documents to Pinecone...")
    progress["total_documents"] = total_docs
    
    batch_size = 50  # Reduced batch size to avoid hitting rate limits
    tokens_per_minute_limit = 250000  # Token rate limit as per error message
    
    # Estimate tokens based on rough approximation (4 chars ≈ 1 token)
    def estimate_tokens(text):
        return len(text) // 4
    
    # Initialize a token counter for rate limiting
    tokens_used_in_current_minute = 0
    minute_start_time = time.time()
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
        # Get the current batch
        batch = documents[i:i+batch_size]
        
        # Estimate tokens in this batch
        batch_token_estimate = sum(estimate_tokens(doc["text"]) for doc in batch)
        
        # Check if we need to wait to avoid rate limiting
        current_time = time.time()
        time_elapsed = current_time - minute_start_time
        
        if time_elapsed >= 60:
            # Reset the token counter for a new minute
            tokens_used_in_current_minute = 0
            minute_start_time = current_time
        elif tokens_used_in_current_minute + batch_token_estimate > tokens_per_minute_limit:
            # Wait until the next minute starts to avoid exceeding the rate limit
            wait_seconds = 60 - time_elapsed
            print(f"Rate limit approaching. Waiting {wait_seconds:.1f} seconds before next batch...")
            time.sleep(wait_seconds)
            tokens_used_in_current_minute = 0
            minute_start_time = time.time()
        
        # With the new approach, we can directly upsert records with the text field
        # and Pinecone will handle the embedding for us
        try:
            index.upsert_records(
                namespace="geton-data",
                records=batch
            )
            
            # Update token usage counter
            tokens_used_in_current_minute += batch_token_estimate
            
            # Update progress
            progress["documents_uploaded"] += len(batch)
            if batch:
                progress["last_processed_document_id"] = batch[-1]["id"]
            save_progress(progress)
            
            # Add a small delay between batches to avoid overloading the API
            time.sleep(1.5)
            
        except Exception as e:
            print(f"Error uploading batch {i//batch_size + 1}: {e}")
            
            if "RESOURCE_EXHAUSTED" in str(e) or "Too Many Requests" in str(e) or "429" in str(e):
                print("Rate limit exceeded. Waiting 60 seconds before retrying...")
                time.sleep(60)  # Wait a full minute before retrying
                try:
                    # Try again after waiting
                    index.upsert_records(
                        namespace="geton-data",
                        records=batch
                    )
                    tokens_used_in_current_minute += batch_token_estimate
                    
                    # Update progress on successful retry
                    progress["documents_uploaded"] += len(batch)
                    if batch:
                        progress["last_processed_document_id"] = batch[-1]["id"]
                    save_progress(progress)
                    
                except Exception as retry_err:
                    print(f"Failed retry for batch {i//batch_size + 1}: {retry_err}")
                    # If batch still fails, try uploading one by one
                    print("Trying to upload documents one by one...")
                    for doc in tqdm(batch, desc="Individual uploads"):
                        try:
                            # Wait a bit between individual uploads
                            time.sleep(1)
                            index.upsert_records(
                                namespace="geton-data",
                                records=[doc]
                            )
                            tokens_used_in_current_minute += estimate_tokens(doc["text"])
                            
                            # Update progress for each individual document
                            progress["documents_uploaded"] += 1
                            progress["last_processed_document_id"] = doc["id"]
                            save_progress(progress)
                            
                        except Exception as individual_err:
                            print(f"Failed to upload document {doc.get('id')}: {individual_err}")
            else:
                # For other errors, try uploading one by one
                print("Trying to upload documents one by one...")
                for doc in tqdm(batch, desc="Individual uploads"):
                    try:
                        time.sleep(1)  # Small delay between individual uploads
                        index.upsert_records(
                            namespace="geton-data",
                            records=[doc]
                        )
                        tokens_used_in_current_minute += estimate_tokens(doc["text"])
                        
                        # Update progress for each individual document
                        progress["documents_uploaded"] += 1
                        progress["last_processed_document_id"] = doc["id"]
                        save_progress(progress)
                        
                    except Exception as individual_err:
                        print(f"Failed to upload document {doc.get('id')}: {individual_err}")
    
    progress["completed"] = True
    save_progress(progress)
    print("Upload complete!")

def main():
    """Main function to process files and upload to Pinecone."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process files and upload to Pinecone")
    parser.add_argument("--resume", action="store_true", help="Resume from last run")
    parser.add_argument("--reset", action="store_true", help="Reset progress and start fresh")
    args = parser.parse_args()
    
    # Handle progress tracking
    if args.reset and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Progress reset. Starting fresh.")
    
    progress = load_progress()
    
    if args.resume and progress["completed"]:
        print("Previous run was already completed. Use --reset to start over.")
        return
    
    # Create directories if they don't exist
    os.makedirs(DISCOURSE_DIR, exist_ok=True)
    os.makedirs(DISCOURSE_SUMM_DIR, exist_ok=True)
    os.makedirs(TDS_PAGES_DIR, exist_ok=True)
    
    # Initialize Pinecone
    index = init_pinecone()
    
    # Process all data sources
    all_documents = []
    
    # Only process files if we don't have documents_uploaded or we're starting fresh
    if progress["documents_uploaded"] == 0 or not args.resume:
        # Process discourse JSON files (original content)
        discourse_documents = process_discourse_files(progress)
        
        # Process summarized discourse QA pairs
        discourse_qa_documents = process_discourse_summ_files(progress)
        
        # Process TDS markdown files
        tds_documents = process_tds_md_files(progress)
        
        # Combine all documents
        all_documents = discourse_documents + discourse_qa_documents + tds_documents
        print(f"Total documents to upload: {len(all_documents)}")
    else:
        print(f"Resuming upload from previous run ({progress['documents_uploaded']} documents already uploaded)")
    
    # Upload to Pinecone (will resume from where it left off)
    upload_to_pinecone(index, all_documents, progress)
    
    print("RAG processing completed successfully!")

if __name__ == "__main__":
    main()
