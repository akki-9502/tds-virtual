import os
import base64
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pinecone
from google import genai
from dotenv import load_dotenv
import tempfile
from PIL import Image
import io

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "geton-rag")
PINECONE_DIMENSION = 1024  # llama-text-embed-v2 dimension

# Initialize Pinecone with robust error handling
pc = None
index = None
try:
    print(f"Initializing Pinecone with API key: {PINECONE_API_KEY[:4]}... (masked)")
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    # List available indexes for debugging
    try:
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
        print(f"Available Pinecone indexes: {index_names}")
        
        if PINECONE_INDEX_NAME not in index_names:
            print(f"WARNING: Index '{PINECONE_INDEX_NAME}' not found in available indexes!")
    except Exception as list_err:
        print(f"Error listing indexes: {list_err}")
    
    # Connect to the index
    print(f"Connecting to index: {PINECONE_INDEX_NAME}")
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Test connection with a simple operation
    try:
        stats = index.describe_index_stats()
        print(f"Connected to index with stats: {stats}")
    except Exception as stats_err:
        print(f"Error getting index stats: {stats_err}")
        
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    print("WARNING: Continuing without Pinecone, search functionality will be limited")

# Initialize Google GenAI
try:
    print(f"Initializing Google GenAI with API key: {GOOGLE_API_KEY[:4]}... (masked)")
    genai_client = genai.Client(api_key=GOOGLE_API_KEY)
    print("Google GenAI initialized successfully")
except Exception as e:
    print(f"Error initializing Google GenAI: {e}")
    genai_client = None
    print("WARNING: Continuing without Google GenAI, generation functionality will be limited")

# Define models to use
EMBEDDING_MODEL = "llama-text-embed-v2"  # Pinecone's embedding model
GEMINI_MODEL = "gemini-2.0-flash"  # Gemini model for generation

# Create FastAPI app
app = FastAPI(title="geton RAG API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class GeminiResponse(BaseModel):
    answer: str

class Link(BaseModel):
    url: str
    text: str
    
    @classmethod
    def model_validator(cls, values):
        url = values.get('url', '')
        if not (url.startswith('http://') or url.startswith('https://')):
            raise ValueError('URL must start with http:// or https://')
        return values

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

def generate_link_text(metadata: Dict[str, Any], content: str) -> str:
    """Generate descriptive text for the link based on metadata and content."""
    # Try various metadata fields for descriptive text
    for field in ["topic_title", "title", "name", "description", "summary"]:
        if field in metadata and metadata[field] and isinstance(metadata[field], str):
            text = metadata[field].strip()
            if text:
                return text
                
    # If content is available and not too long, use it
    if content:
        # Clean up content for display
        clean_content = ' '.join(content.split())  # Normalize whitespace
        
        if len(clean_content) <= 100:
            return clean_content
        else:
            # Extract first sentence or first 100 chars
            first_sentence = clean_content.split('.')[0]
            if len(first_sentence) <= 100:
                return first_sentence + "."
            else:
                return clean_content[:97] + "..."
    
    return "Link to relevant content"

def search_pinecone(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search Pinecone for relevant documents and rerank the results."""
    # Validate Pinecone is initialized
    if not pc or not index:
        print("Pinecone client or index is not initialized. Cannot perform search.")
        return []
        
    try:
        print(f"Attempting to search Pinecone index: {PINECONE_INDEX_NAME}")
        print(f"Query: {query}")
        
        # Step 1: Initial search with integrated embedding
        search_response = index.search(
            namespace="geton-data",
            query={
                "inputs": {"text": query},
                "top_k": top_k * 2  # Get more results for reranking
            },
            fields=["text", "source", "url", "topic_title", "title", "original_url"],
        )
        
        print(f"Initial search response: {search_response}")
        
        # Extract hits from the response - handle the specific format we're seeing
        documents = []
        
        # Check if we have a response with 'result' and 'hits' keys
        if hasattr(search_response, 'result') and hasattr(search_response.result, 'hits'):
            hits = search_response.result.hits
            print(f"Found {len(hits)} hits in response.result.hits")
        elif isinstance(search_response, dict) and 'result' in search_response and 'hits' in search_response['result']:
            hits = search_response['result']['hits']
            print(f"Found {len(hits)} hits in search_response['result']['hits']")
        elif hasattr(search_response, 'matches'):
            hits = search_response.matches
            print(f"Found {len(hits)} hits in response.matches")
        else:
            print(f"Could not find hits in response format: {type(search_response)}")
            print(f"Response structure: {dir(search_response) if hasattr(search_response, '__dir__') else 'No dir available'}")
            return []
        
        # Process hits based on the format we received
        for hit in hits:
            # Extract ID
            hit_id = None
            if hasattr(hit, '_id'):
                hit_id = hit._id
            elif hasattr(hit, 'id'):
                hit_id = hit.id
            elif isinstance(hit, dict) and '_id' in hit:
                hit_id = hit['_id']
            elif isinstance(hit, dict) and 'id' in hit:
                hit_id = hit['id']
            
            # Extract score
            score = None
            if hasattr(hit, '_score'):
                score = hit._score
            elif hasattr(hit, 'score'):
                score = hit.score
            elif isinstance(hit, dict) and '_score' in hit:
                score = hit['_score']
            elif isinstance(hit, dict) and 'score' in hit:
                score = hit['score']
            
            # Extract text and metadata
            text = ""
            metadata = {}
            
            # Check for fields
            if hasattr(hit, 'fields'):
                fields = hit.fields
                if isinstance(fields, dict):
                    text = fields.get('text', '')
                    # Extract all other fields as metadata
                    metadata = {k: v for k, v in fields.items() if k != 'text'}
            elif isinstance(hit, dict) and 'fields' in hit:
                fields = hit['fields']
                text = fields.get('text', '')
                # Extract all other fields as metadata
                metadata = {k: v for k, v in fields.items() if k != 'text'}
            
            # If we have valid data, add to documents
            if hit_id and (text or metadata):
                doc = {
                    'id': hit_id,
                    'score': score if score is not None else 0.0,
                    'text': text,
                    'metadata': metadata
                }
                documents.append(doc)
                print(f"Extracted document: ID={hit_id}, Score={score}, Text length={len(text)}")
        
        print(f"Extracted {len(documents)} documents from search results")
        
        # Step 2: Rerank the results if we have documents and the reranker is available
        if documents and len(documents) > 1:
            try:
                print(f"Attempting to rerank {len(documents)} documents")
                
                # Format documents for reranking
                rerank_docs = [
                    {"id": doc["id"], "text": doc["text"]} 
                    for doc in documents if doc["text"]
                ]
                
                if rerank_docs and len(rerank_docs) > 1:
                    # Use Pinecone's rerank endpoint
                    rerank_result = pc.inference.rerank(
                        model="bge-reranker-v2-m3",
                        query=query,
                        documents=rerank_docs,
                        top_n=min(top_k, len(rerank_docs)),
                        return_documents=True
                    )
                    
                    print(f"Reranking successful. Top score: {rerank_result.data[0].score if rerank_result.data else 'N/A'}")
                    
                    # Extract reranked documents
                    reranked_documents = []
                    for item in rerank_result.data:
                        # Find the original document with all metadata
                        original_doc = next((doc for doc in documents if doc['id'] == item.document.id), None)
                        if original_doc:
                            # Create a new document with the reranked score
                            reranked_doc = original_doc.copy()
                            reranked_doc['score'] = item.score
                            reranked_documents.append(reranked_doc)
                    
                    return reranked_documents
                else:
                    print("Not enough valid documents for reranking")
            except Exception as rerank_err:
                print(f"Error during reranking: {rerank_err}. Falling back to initial results.")
                import traceback
                traceback.print_exc()
                # If reranking fails, return the initial search results
        
        print(f"Returning {len(documents)} documents from search")
        return documents
        
    except Exception as e:
        print(f"Error searching Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return []

def format_context(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format retrieved documents into context for the LLM and extract links."""
    context_parts = []
    links = []
    topic_links = set()  # Track unique topic links to avoid duplicates
    
    if not documents:
        print("No documents to format")
        return {"context": "", "links": []}
    
    print(f"Formatting {len(documents)} documents into context")
    
    # Sort documents by score in descending order
    sorted_docs = sorted(documents, key=lambda x: x.get('score', 0), reverse=True)
    
    for i, doc in enumerate(sorted_docs):
        # Format context without document numbering to avoid the model including it in responses
        context_text = doc.get("text", "")
        if not context_text:
            print(f"Warning: Document {i} has no text content")
            continue
            
        # Add document content to context without the "[Document X]:" prefix
        context_parts.append(f"{context_text}\n")
        
        # Extract metadata for links
        metadata = doc.get("metadata", {})
        
        # Try to extract URL from various metadata fields
        url = ""
        for field in ["url", "original_url", "link", "source_url", "href"]:
            if field in metadata and metadata[field] and isinstance(metadata[field], str):
                url = metadata[field]
                if url.startswith("http://") or url.startswith("https://"):
                    print(f"Found URL in metadata[{field}]: {url}")
                    break
        
        # If no URL in metadata, check if URL is directly in the document
        if not url:
            url = doc.get("url", "")
            
        # Try to build URL from topic_id if available
        if not url and "topic_id" in metadata:
            topic_id = metadata.get("topic_id")
            if topic_id:
                url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}"
                print(f"Built URL from topic_id: {url}")
        
        # Check for valid URL and content
        if url and context_text:
            link_text = generate_link_text(metadata, context_text)
            
            # Check if this is a discourse post URL with a post number
            topic_url = None
            if "discourse.onlinedegree.iitm.ac.in/t/" in url:
                # Extract the topic URL from the post URL
                post_parts = url.split("/")
                if len(post_parts) > 1 and post_parts[-1].isdigit():
                    # This is a post URL like .../t/topic-slug/topic-id/post-id
                    # Convert to topic URL .../t/topic-slug/topic-id
                    topic_url = "/".join(post_parts[:-1])
                    print(f"Extracted topic URL from post URL: {topic_url}")
                    
                    # Add topic URL to the set of unique topic links
                    if topic_url not in topic_links:
                        topic_links.add(topic_url)
                        topic_title = metadata.get("topic_title", "Discussion Topic")
                        links.append({
                            "url": topic_url,
                            "text": f"Topic: {topic_title}"
                        })
                        print(f"Added topic link: {topic_url}")
            
            # Add the original URL if not already present
            if not any(existing["url"] == url for existing in links):
                links.append({
                    "url": url,
                    "text": link_text
                })
                print(f"Extracted link: {url}")
    
    context = "\n\n".join(context_parts)
    print(f"Generated context with {len(context_parts)} document sections and {len(links)} links")
    
    return {
        "context": context,
        "links": links
    }

def process_image(base64_image: str) -> str:
    """Process base64 image and save to a temporary file."""
    try:
        # Remove potential data URL prefix
        if "," in base64_image:
            base64_image = base64_image.split(",", 1)[1]
            
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        
        # Verify image data is valid
        try:
            image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if needed (handles RGBA, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                image.save(temp_file.name, format="JPEG", quality=95)
                print(f"Image saved to temporary file: {temp_file.name}")
                return temp_file.name
        except Exception as img_err:
            print(f"Error processing image data: {img_err}")
            return None
    except Exception as e:
        print(f"Error processing base64 image: {e}")
        return None

def generate_answer(question: str, context: str, links: List[Dict[str, str]], image_path: Optional[str] = None) -> str:
    """Generate answer using Google Gemini with structured output for just the answer."""
    # Check if Gemini client is available
    if not genai_client:
        print("Google GenAI client is not initialized. Returning fallback answer.")
        return "I couldn't generate a specific answer due to a technical issue. Please try again later or reformulate your question."
        
    try:
        # Prepare prompt
        prompt = f"""You are an AI assistant for the IITM Online Degree. Answer questions based on the following context.
        
Context:
{context}

Question: {question}

Generate a helpful response. Include only highly relevant information from the context.
Focus ONLY on providing a clear and concise answer to the question based on the context provided.
Do not make up information not present in the context.
Do NOT include document references, numbers, or citations in your answer.
Do NOT include links or URLs in your response.
Do NOT mention "Document 1", "Document 2", etc. in your response.
Your Context cutoff date is 2025-04-15. any question after this date should not be answered.
Any question that is in the future should not be answered(anything after 2025 April 15). And return that this information is not available yet and does not know yet.
"""
        
        # Set up structured output configuration
        response_config = {
            "response_mime_type": "application/json",
            "response_schema": GeminiResponse,
        }
        
        # Generate content
        if image_path:
            try:
                # Load image parts
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                    
                # Create multimodal generation request with proper MIME type and structured output
                response = genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        {"role": "user", "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}}
                        ]}
                    ],
                    config=response_config
                )
                print("Successfully generated content with image")
            except Exception as img_err:
                print(f"Error generating content with image: {img_err}")
                # Fallback to text-only if image processing fails
                print("Falling back to text-only generation")
                response = genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=response_config
                )
        else:
            # Text-only generation with structured output
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=response_config
            )
        
        # Clean up temporary image file if it exists
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
            
        # Get the parsed response
        if hasattr(response, "parsed") and response.parsed:
            return response.parsed.answer
        
        # Fallback to text response if parsing fails
        return response.text.strip()
            
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"I encountered an error while generating an answer. Please try again with a different question."

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query and return a response."""
    print(f"\n----- New Query -----")
    print(f"Question: {request.question}")
    
    # Process image if provided
    image_path = None
    if request.image:
        print(f"Image provided: Processing image...")
        image_path = process_image(request.image)
        if not image_path:
            print("Failed to process image, continuing without image")
    
    # Search for relevant documents
    print(f"Searching Pinecone for relevant documents...")
    documents = search_pinecone(request.question)
    
    # Fallback if no documents found
    if not documents:
        print("No documents found from primary search. Trying fallback approach...")
        try:
            # Simplified fallback search using basic methods
            fallback_query = {
                "vector": {"values": [0.1] * PINECONE_DIMENSION},  # Simple dummy vector
                "top_k": 5
            }
            
            # Try direct namespace search
            fallback_response = index.search(
                namespace="geton-data",
                query=fallback_query
            )
            
            print(f"Fallback search response type: {type(fallback_response)}")
            
            # Process fallback results
            if fallback_response and hasattr(fallback_response, 'matches') and fallback_response.matches:
                documents = [
                    {
                        "id": match.id,
                        "score": 0.5,  # Arbitrary score since this is a fallback
                        "text": "Fallback document content. Please try a more specific question.",
                        "metadata": {"source": "fallback", "title": "Fallback Document"}
                    }
                    for match in fallback_response.matches[:3]
                ]
                print(f"Found {len(documents)} documents from fallback search")
        except Exception as fallback_err:
            print(f"Fallback search also failed: {fallback_err}")
    
    # Print returned documents for debugging
    print(f"\nFound {len(documents)} relevant documents:")
    for i, doc in enumerate(documents[:3]):  # Print top 3 for brevity
        print(f"\n--- Document {i+1} (Score: {doc.get('score', 'N/A')}) ---")
        print(f"ID: {doc.get('id', 'N/A')}")
        content = doc.get('text', '')
        print(f"Content: {content[:200]}..." if len(content) > 200 else f"Content: {content}")
        
        metadata = doc.get('metadata', {})
        if metadata:
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Metadata: {metadata}")
    
    if len(documents) > 3:
        print(f"... and {len(documents) - 3} more documents")
    
    # Format context and extract links
    formatted_data = format_context(documents)
    context = formatted_data["context"]
    links = formatted_data["links"]
    
    # Print links for debugging
    print(f"\nExtracted {len(links)} links:")
    for i, link in enumerate(links):
        print(f"{i+1}. {link['url']} - {link['text'][:50]}...")
    
    # If no documents found, provide a fallback answer
    if not documents:
        print("No relevant documents found. Providing a fallback answer.")
        return QueryResponse(
            answer="I couldn't find specific information about this in the course materials. Please try asking your question more specifically, or refer to the course syllabus for more information.",
            links=[]
        )
    
    # Generate answer with structured output - with retry logic
    print(f"\nGenerating answer with Gemini structured output...")
    
    max_retries = 3
    retries = 0
    valid_response = False
    answer = None
    
    while not valid_response and retries < max_retries:
        if retries > 0:
            print(f"\nRetrying answer generation (attempt {retries+1}/{max_retries})...")
            
        # Generate answer
        answer = generate_answer(request.question, context, links, image_path)
        print(f"\nAnswer from Gemini (attempt {retries+1}): {answer[:200]}...")
        
        # Validate the answer format
        try:
            # Check if the answer is a properly formatted string
            if isinstance(answer, str) and answer.strip() and not any(answer.strip().startswith(prefix) for prefix in ["{", "["]):
                # Check for JSON-like structures within the string
                if not (("{" in answer and "}" in answer and ":" in answer) or 
                       ("[" in answer and "]" in answer and ":" in answer)):
                    valid_response = True
                    print("Response format validated successfully.")
                else:
                    print("Response contains JSON-like content. Retrying...")
            else:
                print("Response is not a properly formatted string. Retrying...")
        except Exception as e:
            print(f"Error validating response format: {e}")
        
        retries += 1
        
        # If we're about to retry, sleep briefly to avoid API rate limits
        if not valid_response and retries < max_retries:
            import time
            time.sleep(1)
    
    # If we couldn't get a valid response after max retries, use the last response anyway
    if not valid_response:
        print("\nWarning: Could not get properly formatted response after max retries. Using last response.")
        
        # Try to clean up the response if it looks like JSON
        if isinstance(answer, str) and "{" in answer and "}" in answer:
            try:
                # Try to extract any answer field from potential JSON
                import re
                import json
                
                # Check for JSON objects
                json_matches = re.findall(r'{[^{}]*}', answer)
                for json_str in json_matches:
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict) and "answer" in parsed and isinstance(parsed["answer"], str):
                            answer = parsed["answer"]
                            print(f"Extracted answer from JSON: {answer[:100]}...")
                            break
                    except:
                        continue
            except Exception as e:
                print(f"Error cleaning response: {e}")
    
    # Convert links to proper Link objects for the response
    formatted_links = []
    for link in links[:3]:  # Use top 3 most relevant links
        if "url" in link and "text" in link:
            try:
                url = link["url"]
                text = link["text"]
                
                # Validate URL
                if url and isinstance(url, str) and (url.startswith("http://") or url.startswith("https://")):
                    formatted_links.append(Link(url=url, text=text))
            except Exception as e:
                print(f"Error formatting link: {e}")
    
    print(f"Returning response with {len(formatted_links)} formatted links")
    
    # Return response with the answer and links
    return QueryResponse(
        answer=answer,
        links=formatted_links
    )

@app.get("/")
async def root():
    """Root endpoint for simple API test."""
    return {"status": "ok", "message": "geton RAG API is running"}

@app.post("/debug")
async def debug(request: Request):
    """Debug endpoint to check what's being received."""
    body = await request.body()
    try:
        json_body = json.loads(body)
        return {
            "received": {
                "body": json_body,
                "content_type": request.headers.get("content-type")
            },
            "status": "ok"
        }
    except:
        return {
            "received": {
                "body": str(body),
                "content_type": request.headers.get("content-type")
            },
            "status": "error parsing JSON"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 