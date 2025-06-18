#!/usr/bin/env python3

import os
import json
import re
import argparse
from pathlib import Path
from google import genai
from google.genai import types
from typing import Dict, List, Any, Optional, Tuple
import requests
import html
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import shutil
from datetime import datetime, timedelta
from tqdm import tqdm
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("discourse_processing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add dotenv support for environment variables
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv package not found. Install using: pip install python-dotenv")
    logger.info("Falling back to system environment variables")

# Configure API key for Gemini
# First try to get it from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY environment variable not set")
    logger.info("Please set it in your .env file or as an environment variable:")
    logger.info("1. Create a .env file in the project root with: GEMINI_API_KEY=your-api-key")
    logger.info("2. Or set it using: export GEMINI_API_KEY='your-api-key'")
else:
    # Configure Gemini API with new SDK
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini client configured.")

# Set authentication cookie for raw endpoint directly in the code with triple quotes
# to handle special characters properly
RAW_COOKIE_STRING = """_t=L25QKXrqQv4KhrzK6TJxX5VLJgVtRRQU2CdEaHj%2BQBtfZUbTrdmYdbFXJeJhOLXnboL0q7ad0Xmg2eE7Noncx1guQJnNzeJxOr49fsuVh2vhWy%2ByWM%2BggqNucCA8nNr9nEoq%2B17Al5kmeOEMMnQxuonQRrvd1f1UFAMb16jDTezG44AJLWpnK%2FOBT9LO8yNC7ZStz720%2Fm5H7jEbC5MCLMXytTgUfp7oyWZXsGM5TwNTJjR6S1ymN%2FKF1tJko8gIIW9xlxRGeF8ZZzK0Tw1vySlUszYH8KuNzwMhxgRZRteZJ5V1o1zoSXEOma4Le91kN12yQw%3D%3D--d4Atgy0qK0sfjAIF--%2FiKtZjz1%2FHwYRlo2%2FUTiUQ%3D%3D; _forum_session=K69EP8S9qy02sy1sAeCupdED%2BNJsyRFFDW8WT%2BxeBEOQdJtDlRVhEoOu8Jseu1sPFfqQKcsHpopDm3g7AMAW%2F242fT5NXg29DRxU6Ma9lbQC%2BN2gVhGDlwsLVS6w0Fe2vmBLG15Yt67bH%2BiIdYdsW7w5PIxBOv6JRSagexo523Q6J9YsWHYsIVJpArLQ3aD%2B2P1t9%2Fnt7yjQh0xRKz9IIkatAvQhyPWK8sQ4DxQcTdZT9YLI06TaSk5qZqjbQK9JJWlDIN%2FRGrUr9kiS8HdTQJy63kXlraMK3bfHy9dFD6t1pvyBf1QOUGnk4ChaITUkmmcEE2vAov%2B1yeAx5Rc14zyruUfN6WUybslRGcdOGyreKEb1ANHgmMzu%2FTQf3ixYvjWloNMnjufV8GpMNsg%2BTovew3WWK2KtBNw%3D--0tpCrSsu3zZOgkfB--zbptZQQiGG%2BleDmk6alizw%3D%3D"""
logger.info("Using hardcoded authentication cookie for raw endpoint")

# Setup paths
DISCOURSE_JSON_DIR = Path("downloads/discourse_json")
DISCOURSE_SUMM_DIR = Path("downloads/discourse_summ")
DISCOURSE_MARKDOWN_DIR = Path("downloads/discourse_markdown")
TOPICS_OUTPUT_FILE = Path("downloads/topics.json")
QA_PAIRS_OUTPUT_FILE = Path("downloads/qa_pairs.json")

# Ensure output directories exist
DISCOURSE_SUMM_DIR.mkdir(parents=True, exist_ok=True)
DISCOURSE_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)


def clean_html_content(html_content: str) -> str:
    """
    Clean HTML content to extract clean text
    
    Args:
        html_content: HTML content to clean
        
    Returns:
        Clean text content
    """
    # Use BeautifulSoup for better HTML parsing
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        logger.error(f"Error cleaning HTML: {str(e)}")
        # Fallback to simpler cleaning if BeautifulSoup fails
        return re.sub(r'<[^>]+>', '', html.unescape(html_content))


# Global request timestamp tracking to implement rate limiting
last_request_time = 0

def get_raw_post_content(topic_id: int, post_number: int, discourse_url: str = "https://discourse.onlinedegree.iitm.ac.in") -> str:
    """
    Fetch the raw markdown content of a post directly from Discourse
    
    Args:
        topic_id: The topic ID
        post_number: The post number within the topic
        discourse_url: Base URL of the Discourse instance
    
    Returns:
        Raw markdown content of the post
    
    Raises:
        requests.exceptions.RequestException: If the request fails after retries.
    """
    global last_request_time
    
    # Implement rate limiting - wait at least 5 seconds between requests
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    
    if time_since_last_request < 2:
        sleep_time = 2 - time_since_last_request
        logger.debug(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    
    try:
        url = f"{discourse_url}/raw/{topic_id}/{post_number}"
        
        # Update last request time before making the request
        last_request_time = time.time()
        
        # Set up headers with cookie for authenticated request
        headers = {'Cookie': RAW_COOKIE_STRING}
        logger.debug(f"Making request to {url}")
        
        # Make request with authentication
        response = requests.get(url, headers=headers, timeout=30)
        
        # Log the response status for debugging
        logger.debug(f"Response status for {url}: {response.status_code}")
        
        if response.status_code == 200:
            return response.text
        elif response.status_code == 429:
            # If rate limited, wait longer and retry once
            logger.warning(f"Rate limited when fetching post number {post_number} in topic {topic_id}. Waiting 30 seconds...")
            time.sleep(30)
            
            # Update last request time before retrying
            last_request_time = time.time()
            
            retry_response = requests.get(url, headers=headers, timeout=30)
            if retry_response.status_code == 200:
                return retry_response.text
            else:
                logger.error(f"Still failed after retry for post number {post_number} in topic {topic_id}: {retry_response.status_code}")
                logger.error(f"Response body: {retry_response.text[:200]}...")  # Log first 200 chars
                retry_response.raise_for_status()
        else:
            logger.warning(f"Error fetching raw content for post number {post_number} in topic {topic_id}: {response.status_code}")
            logger.warning(f"Response body: {response.text[:200]}...")  # Log first 200 chars
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Exception fetching raw content for post number {post_number} in topic {topic_id}: {str(e)}")
        raise


def fetch_post_contents_sequential(topic_id: int, posts: List[Dict]) -> Dict[int, str]:
    """
    Fetch raw content for posts sequentially with rate limiting
    
    Args:
        topic_id: The topic ID
        posts: List of post dictionaries
        
    Returns:
        Dictionary mapping post IDs to their raw content
    """
    raw_posts = {}
    
    post_pbar = tqdm(posts, desc=f"Fetching posts for topic {topic_id}", leave=False, unit="post")
    for post in post_pbar:
        post_id = post.get("id")
        post_number = post.get("post_number")
        post_pbar.set_postfix_str(f"Post No: {post_number}")
        
        content = get_raw_post_content(topic_id, post_number)
        raw_posts[post_id] = content
        
    return raw_posts


def create_topic_raw_markdown(topic_data: Dict, raw_posts: Dict[int, str]) -> str:
    """
    Create a raw markdown file by directly concatenating raw post contents
    
    Args:
        topic_data: The topic data dictionary
        raw_posts: Dictionary mapping post IDs to raw content from the raw endpoint
        
    Returns:
        Raw markdown string representing all posts in the topic
    """
    topic_title = topic_data.get("title", "Untitled Topic")
    posts = topic_data.get("post_stream", {}).get("posts", [])
    
    # Start with topic header
    md = f"# {topic_title}\n\n"
    
    # Add each post with minimal formatting - just use the raw content directly
    for post in posts:
        post_id = post.get("id")
        username = post.get("username", "")
        post_number = post.get("post_number", "")
        created_at = post.get("created_at", "") # Keep it as a string
        
        # Add simple post header with creation date for context
        md += f"## Post #{post_number} by {username} (Created at: {created_at})\n\n"
        
        # Use raw content directly from the raw endpoint
        if post_id in raw_posts and raw_posts[post_id]:
            md += raw_posts[post_id]
        else:
            # Fallback to cleaned HTML if raw content not available (should be rare now)
            logger.warning(f"Raw content for post {post_id} in topic {topic_data.get('id')} not found, falling back to cooked content.")
            md += clean_html_content(post.get("cooked", ""))
        
        md += "\n\n---\n\n"  # Separator between posts
    
    return md


def analyze_with_gemini(topic_data: Dict, markdown_content: str) -> Dict:
    """
    Use Gemini API to analyze the topic, extract questions and answers, and create summary
    
    Args:
        topic_data: The full topic data from Discourse
        markdown_content: Raw markdown representation of the entire topic
        
    Returns:
        Dictionary with extracted questions, answers, and summary
    """
    if not GEMINI_API_KEY:
        logger.warning("Skipping Gemini analysis as API key is not set")
        return {
            "questions": [],
            "answers": [],
            "summary": "No summary generated (Gemini API key not set)"
        }

    # Extract basic topic metadata for context
    topic_title = topic_data.get("title", "")
    posts = topic_data.get("post_stream", {}).get("posts", [])
    tags = topic_data.get("tags", [])
    category_id = topic_data.get("category_id")
    is_solved = any(post.get("accepted_answer", False) for post in posts)
    
    # Prepare brief context information to help orient the model
    context_info = f"""
Topic Title: {topic_title}
Tags: {', '.join(tags)}
Category ID: {category_id}
Has accepted solution: {is_solved}
Number of posts: {len(posts)}
"""

    # Define the response schema for structured output
    class Answer(BaseModel):
        post_id: int
        text: str
        is_accepted: bool = False
        is_staff_response: bool = False

    class Question(BaseModel):
        id: int
        post_id: int
        text: str
        answers: List[Answer] = []
        is_answered: bool = False

    class TopicAnalysis(BaseModel):
        questions: List[Question] = []
        summary: str
        topic_type: str
        has_resolution: bool = False
        key_information: str

    # Improved prompt with more specific instructions
    prompt = f"""
You are an expert analyst specializing in extracting structured information from online forum discussions. Your task is to analyze this Discourse forum topic and extract high-quality question-answer pairs for a retrieval-augmented generation (RAG) system.

CONTEXT INFORMATION:
{context_info}

FULL TOPIC CONTENT:
Each post in the content below includes a "Created at" timestamp in its header. Use this timestamp to resolve any relative date and time references (e.g., "yesterday", "in 2 hours", "last week") within that specific post, converting them to absolute timestamps.

{markdown_content}

EXTRACTION TASKS:
1. Extract all questions being asked in the topic. Questions must be complete, detailed, and include necessary context. Do not create questions that don't exist in the content.
2. For each question, identify all posts that provide answers, mapping answers to their respective questions.
3. Generate a concise yet comprehensive summary of the entire topic (2-3 sentences).
4. The final output must contain absolute dates and times, resolved from any relative references in the source text.

IMPORTANT CONSIDERATIONS:
- When extracting text for questions and answers, make sure to replace all relative date and time references with calculated absolute timestamps. Use the "Created at" timestamp from the post header as the reference for each post's content.
- Be thorough in identifying questions - they may be explicit or implicit.
- Questions could be in any post, not just the first one.
- Include the full context in questions and answers to make them self-contained.
- If a question has no answers, mark it as unanswered.
- If a post doesn't have any clear questions or is purely informational, extract the key information.
- Ensure that accepted answers are highlighted.
- Pay special attention to staff/admin/moderator responses.
"""

    # Add retry logic for API call
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # API call with structured output
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": TopicAnalysis,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            
            # Extract the structured response
            result = response.parsed
            
            # Convert to dictionary
            result_dict = result.model_dump()
            
            # Add post metadata to questions and answers
            posts_dict = {post.get("id"): post for post in posts}
            
            for question in result_dict.get("questions", []):
                # Find the post that contains this question
                question_post_id = question.get("post_id")
                if question_post_id in posts_dict:
                    post = posts_dict[question_post_id]
                    question["post_number"] = post.get("post_number")
                    question["username"] = post.get("username")
                    question["created_at"] = post.get("created_at")
                
                # Add metadata to answers
                for answer in question.get("answers", []):
                    answer_post_id = answer.get("post_id")
                    if answer_post_id in posts_dict:
                        post = posts_dict[answer_post_id]
                        answer["post_number"] = post.get("post_number")
                        answer["username"] = post.get("username")
                        answer["created_at"] = post.get("created_at")
            
            return result_dict
        except Exception as e:
            logger.warning(f"API call attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"All {max_retries} retries failed for topic {topic_data.get('id')}.")
                raise  # Re-raise the exception if all retries failed


def process_topic_file(topic_file: Path) -> Optional[Dict]:
    """
    Process a single topic JSON file and extract the relevant information
    
    Args:
        topic_file: Path to the topic JSON file
    
    Returns:
        Dictionary with processed topic data or None if topic should be excluded
    
    Raises:
        Exception: Propagates exceptions from downstream functions like API calls or file processing.
    """
    with open(topic_file, 'r', encoding='utf-8') as f:
        topic_data = json.load(f)
    
    topic_id = topic_data.get("id")
    title = topic_data.get("title", "")
    posts = topic_data.get("post_stream", {}).get("posts", [])
    
    logger.info(f"Processing topic {topic_id}: {title}")
    
    # Skip topics with no posts
    if not posts:
        logger.info(f"Skipping topic {topic_id}: No posts found")
        return None
        
    # Special handling for topics with only one post
    if len(posts) == 1 and topic_data.get("reply_count", 0) == 0:
        # Check if this is a significant post that might deserve inclusion
        post_content = posts[0].get("cooked", "")
        content_length = len(clean_html_content(post_content))
        
        if content_length < 500:  # Arbitrary threshold
            logger.info(f"Skipping topic {topic_id}: Single short post with no replies")
            return None
        else:
            logger.info(f"Including topic {topic_id} despite single post due to significant content length: {content_length} chars")
    
    # Fetch raw content for each post sequentially with rate limiting
    raw_posts = fetch_post_contents_sequential(topic_id, posts)
    
    # Create raw markdown representation of the topic
    markdown_content = create_topic_raw_markdown(topic_data, raw_posts)
    
    # Save markdown to file
    markdown_file = DISCOURSE_MARKDOWN_DIR / f"topic_{topic_id}.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    logger.info(f"  Saved raw markdown content to {markdown_file}")
        
    # Use Gemini to analyze the topic with markdown content
    analysis = analyze_with_gemini(topic_data, markdown_content)
    
    # Add topic metadata to the result
    result = {
        "topic_id": topic_id,
        "title": title,
        "created_at": topic_data.get("created_at"),
        "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_data.get('slug')}/{topic_id}",
        "markdown_path": str(markdown_file),
        "tags": topic_data.get("tags", []),
        "category_id": topic_data.get("category_id"),
        "views": topic_data.get("views", 0),
        "posts_count": topic_data.get("posts_count", 0),
        "reply_count": topic_data.get("reply_count", 0),
        "like_count": topic_data.get("like_count", 0),
        "has_accepted_answer": any(post.get("accepted_answer", False) for post in posts),
        "analysis": analysis
    }
    
    # Always include topics with content for RAG - don't filter out
    # as previous version did - we want to be more inclusive for RAG
    if not analysis.get("questions"):
        result["content_only"] = True
        result["content_summary"] = analysis.get("summary", "")
        logger.info(f"Topic {topic_id}: No questions detected but keeping as content-only")
        
    # For topics with questions but no answers
    has_answered_questions = any(
        question.get("is_answered", False) 
        for question in analysis.get("questions", [])
    )
    
    if not has_answered_questions and analysis.get("questions"):
        result["unanswered_only"] = True
        logger.info(f"Topic {topic_id}: Contains {len(analysis.get('questions', []))} unanswered questions")
        
    return result


def extract_qa_pairs(processed_topics: List[Dict]) -> List[Dict]:
    """
    Extract question-answer pairs from processed topics
    
    Args:
        processed_topics: List of processed topic dictionaries
        
    Returns:
        List of question-answer pairs
    """
    qa_pairs = []
    
    for topic in processed_topics:
        topic_id = topic.get("topic_id")
        url = topic.get("url")
        title = topic.get("title")
        markdown_path = topic.get("markdown_path")
        
        for question in topic.get("analysis", {}).get("questions", []):
            # Include unanswered questions in the pairs too
            is_answered = question.get("is_answered", False)
            question_entry = {
                "topic_id": topic_id,
                "topic_title": title,
                "topic_url": url,
                "markdown_path": markdown_path,
                "question_id": question.get("post_id"),
                "question_text": question.get("text"),
                "is_answered": is_answered,
                "question_username": question.get("username"),
                "answers": []
            }
            
            if is_answered:
                for answer in question.get("answers", []):
                    question_entry["answers"].append({
                        "answer_id": answer.get("post_id"),
                        "answer_text": answer.get("text"),
                        "is_accepted": answer.get("is_accepted", False),
                        "is_staff_response": answer.get("is_staff_response", False),
                        "answer_username": answer.get("username")
                    })
            
            qa_pairs.append(question_entry)
    
    return qa_pairs


def save_progress(all_processed_topics: List[Dict], output_path: Path, qa_pairs_path: Path) -> None:
    """
    Save current progress to output files
    
    Args:
        all_processed_topics: List of processed topics
        output_path: Path to save topics summary
        qa_pairs_path: Path to save QA pairs
    """
    # Create topics summary with relevant information
    topics_summary = []
    
    # Track the last successfully processed topic ID for resuming
    last_processed_id = None
    if all_processed_topics:
        last_processed_id = all_processed_topics[-1].get("topic_id")
    
    for topic in all_processed_topics:
        topics_summary.append({
            "topic_id": topic.get("topic_id"),
            "title": topic.get("title"),
            "url": topic.get("url"),
            "markdown_path": topic.get("markdown_path"),
            "created_at": topic.get("created_at"),
            "tags": topic.get("tags", []),
            "category_id": topic.get("category_id"),
            "posts_count": topic.get("posts_count", 0),
            "has_accepted_answer": topic.get("has_accepted_answer", False),
            "question_count": len(topic.get("analysis", {}).get("questions", [])),
            "answered_question_count": len([
                q for q in topic.get("analysis", {}).get("questions", []) 
                if q.get("is_answered", False)
            ]),
            "summary": topic.get("analysis", {}).get("summary", ""),
            "topic_type": topic.get("analysis", {}).get("topic_type", "unknown"),
            "has_resolution": topic.get("analysis", {}).get("has_resolution", False),
            "key_information": topic.get("analysis", {}).get("key_information", "")
        })
    
    # Save the master topics file with all topics that have been processed successfully
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "topics": topics_summary,
            "last_processed_id": last_processed_id,
            "last_updated": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(topics_summary)} topic summaries to {output_path}")
    
    # Extract QA pairs for easier RAG usage
    qa_pairs = extract_qa_pairs(all_processed_topics)
    
    # Create topic-question-answer mapping
    topic_qa_mapping = {}
    for pair in qa_pairs:
        topic_id = pair["topic_id"]
        if topic_id not in topic_qa_mapping:
            topic_qa_mapping[topic_id] = {
                "topic": pair["topic_title"],
                "pairs": []
            }
        
        # Create answer IDs list
        answer_ids = [answer["answer_id"] for answer in pair.get("answers", [])]
        
        # Add to mapping if it has answers
        if pair["is_answered"]:
            topic_qa_mapping[topic_id]["pairs"].append({
                "question": pair["question_id"],
                "answers": answer_ids
            })
    
    # Save question-answer pairs
    with open(qa_pairs_path, 'w', encoding='utf-8') as f:
        json.dump({
            "qa_pairs": qa_pairs,
            "topic_qa_mapping": topic_qa_mapping,
            "last_processed_id": last_processed_id,
            "last_updated": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(qa_pairs)} question-answer pairs to {qa_pairs_path}")


def main():
    """
    Main function to process all topic files and generate output files
    """
    parser = argparse.ArgumentParser(description='Process Discourse JSON files for RAG system')
    parser.add_argument('--limit', type=int, help='Limit the number of files to process')
    parser.add_argument('--topic-id', type=int, help='Process only a specific topic ID')
    parser.add_argument('--skip-processed', action='store_true', help='Skip already processed topics')
    parser.add_argument('--save-interval', type=int, default=10, help='Number of topics to process before saving progress')
    parser.add_argument('--resume', action='store_true', help='Resume from last successfully processed topic')
    args = parser.parse_args()
    
    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY is not set. Exiting.")
        return

    topic_files = sorted(list(DISCOURSE_JSON_DIR.glob('topic_*.json')))
    total_files = len(topic_files)
    logger.info(f"Found {total_files} topic files to process")
    
    if args.topic_id:
        topic_file = DISCOURSE_JSON_DIR / f"topic_{args.topic_id}.json"
        if topic_file.exists():
            topic_files = [topic_file]
            logger.info(f"Processing only topic ID {args.topic_id}")
        else:
            logger.error(f"Topic file for ID {args.topic_id} not found")
            return
    
    # Resume from last processed topic if requested
    last_processed_id = None
    if args.resume and TOPICS_OUTPUT_FILE.exists():
        try:
            with open(TOPICS_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                topics_data = json.load(f)
                last_processed_id = topics_data.get("last_processed_id")
                
            if last_processed_id:
                logger.info(f"Resuming from last processed topic ID: {last_processed_id}")
                # Find the index of the last processed topic file
                for i, file_path in enumerate(topic_files):
                    current_id = int(re.search(r'topic_(\d+)\.json', file_path.name).group(1))
                    if current_id == last_processed_id:
                        # Start from the next topic
                        topic_files = topic_files[i+1:]
                        logger.info(f"Skipping {i+1} already processed topics. Remaining: {len(topic_files)}")
                        break
        except Exception as e:
            logger.warning(f"Error when trying to resume: {e}. Starting from the beginning.")
    
    if args.limit and args.limit < len(topic_files):
        topic_files = topic_files[:args.limit]
        logger.info(f"Processing limited to {len(topic_files)} files")
    
    # Load previously processed topics if resuming
    all_processed_topics = []
    if args.resume and TOPICS_OUTPUT_FILE.exists():
        try:
            with open(TOPICS_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                topics_data = json.load(f)
                # Load existing topics if available
                if "topics" in topics_data:
                    # We need to load the full topic data from the cleaned files
                    for topic_summary in topics_data.get("topics", []):
                        topic_id = topic_summary.get("topic_id")
                        clean_file = DISCOURSE_SUMM_DIR / f"topic_clean_{topic_id}.json"
                        if clean_file.exists():
                            with open(clean_file, 'r', encoding='utf-8') as cf:
                                topic_data = json.load(cf)
                                all_processed_topics.append(topic_data)
                    
                    logger.info(f"Loaded {len(all_processed_topics)} previously processed topics")
        except Exception as e:
            logger.warning(f"Error loading previously processed topics: {e}. Starting with empty list.")
    
    # Skip already processed topics if requested
    if args.skip_processed and not args.resume:
        # Check for the existence of the final cleaned JSON file
        processed_ids = {
            int(re.search(r'topic_clean_(\d+)\.json', f.name).group(1))
            for f in DISCOURSE_SUMM_DIR.glob('topic_clean_*.json')
        }
        if processed_ids:
            logger.info(f"Found {len(processed_ids)} already processed topics. Skipping them.")
            original_count = len(topic_files)
            topic_files = [
                f for f in topic_files 
                if int(re.search(r'topic_(\d+)\.json', f.name).group(1)) not in processed_ids
            ]
            logger.info(f"Skipped {original_count - len(topic_files)} topics. Remaining: {len(topic_files)}")
    
    # Track progress and errors
    start_time = time.time()
    processed_count = 0
    save_interval = args.save_interval
    consecutive_discourse_errors = 0
    consecutive_gemini_errors = 0
    MAX_CONSECUTIVE_ERRORS = 3
    
    # Use tqdm for a master progress bar
    topic_pbar = tqdm(topic_files, desc="Processing Topics", unit="topic")

    for topic_file in topic_pbar:
        topic_id = int(re.search(r'topic_(\d+)\.json', topic_file.name).group(1))
        topic_pbar.set_postfix_str(f"ID: {topic_id}")
        
        try:
            # Process the topic file
            try:
                topic_data = process_topic_file(topic_file)
                # Reset Discourse errors on success
                consecutive_discourse_errors = 0
            except requests.exceptions.RequestException as e:
                # Handle Discourse API errors separately
                consecutive_discourse_errors += 1
                logger.error(f"Discourse API error when processing topic {topic_id}: {e}")
                logger.warning(f"Consecutive Discourse errors: {consecutive_discourse_errors}/{MAX_CONSECUTIVE_ERRORS}")
                
                if consecutive_discourse_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical(f"Stopping script after {MAX_CONSECUTIVE_ERRORS} consecutive Discourse API errors.")
                    break
                continue
            except Exception as e:
                if "genai" in str(e).lower() or "gemini" in str(e).lower():
                    # Likely a Gemini API error
                    consecutive_gemini_errors += 1
                    logger.error(f"Possible Gemini API error when processing topic {topic_id}: {e}")
                    logger.warning(f"Consecutive Gemini errors: {consecutive_gemini_errors}/{MAX_CONSECUTIVE_ERRORS}")
                    
                    if consecutive_gemini_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.critical(f"Stopping script after {MAX_CONSECUTIVE_ERRORS} consecutive Gemini API errors.")
                        break
                else:
                    # Other error, track as Discourse for simplicity
                    consecutive_discourse_errors += 1
                    logger.error(f"Error when processing topic {topic_id}: {e}")
                    logger.warning(f"Consecutive errors: {consecutive_discourse_errors}/{MAX_CONSECUTIVE_ERRORS}")
                    
                    if consecutive_discourse_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.critical(f"Stopping script after {MAX_CONSECUTIVE_ERRORS} consecutive errors.")
                        break
                continue
            
            if topic_data:
                output_file = DISCOURSE_SUMM_DIR / f"topic_clean_{topic_id}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(topic_data, f, ensure_ascii=False, indent=2)
                logger.info(f"  Saved cleaned data to {output_file}")
                
                all_processed_topics.append(topic_data)
                processed_count += 1
                # Reset all error counters on success
                consecutive_discourse_errors = 0
                consecutive_gemini_errors = 0
            else:
                # This happens for intentionally skipped topics (e.g., no posts)
                logger.info(f"Topic {topic_id} was skipped as per processing rules.")

            # Save progress at specified intervals
            if processed_count > 0 and (processed_count % save_interval == 0 or topic_pbar.n == len(topic_pbar) - 1):
                logger.info(f"Saving progress after processing {processed_count} topics...")
                save_progress(all_processed_topics, TOPICS_OUTPUT_FILE, QA_PAIRS_OUTPUT_FILE)
        
        except Exception as e:
            logger.error(f"Unhandled exception when processing topic {topic_id}: {e}", exc_info=True)
            # Count as a general error
            consecutive_discourse_errors += 1
            if consecutive_discourse_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.critical(f"Stopping script after {MAX_CONSECUTIVE_ERRORS} consecutive errors.")
                break
    
    # Final save
    logger.info("Performing final save of all processed topics for this session.")
    save_progress(all_processed_topics, TOPICS_OUTPUT_FILE, QA_PAIRS_OUTPUT_FILE)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Processing complete. Processed {processed_count} topics successfully in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
