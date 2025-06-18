# GetOn RAG System

A smart retrieval system that helps students find answers from course materials and forums.

## What it does

This project grabs content from:
- The course Discourse forum (questions and answers)
- TDS website pages

Then it:
1. Processes everything
2. Creates vector embeddings 
3. Builds a search system that helps find relevant info
4. Powers a simple question-answering interface

## Getting started

1. Install the packages:
```
pip install -r requirements.txt
```

2. Set up your environment by creating a `.env` file with:
```
PINECONE_API=your_pinecone_key_here
GEMINI_API_KEY=your_gemini_key_here
```

## How to run

### Step 1: Download data
First, grab all the content:

```
python 1_discourse_download.py
python 2_tds_website_download.py
```

### Step 2: Process forum content
Process and analyze the forum content:

```
python 3_discourse_clean_summary.py
```

### Step 3: Index everything
Run the main RAG process:

```
python 4_RAG_Process.py
```

### Step 4: Start the API server
Run the API for the question-answering interface:

```
uvicorn rag_api:app --reload
```

## What's inside

### Scripts explained

- `1_discourse_download.py`: Downloads forum topics and posts from Discourse
- `2_tds_website_download.py`: Grabs pages from the TDS website as markdown
- `3_discourse_clean_summary.py`: Cleans and extracts Q&A pairs from forum content
- `4_RAG_Process.py`: Creates the search index with vector embeddings
- `rag_api.py`: Runs the API server that handles queries

### File structure

- `downloads/discourse_json/`: Raw forum data
- `downloads/tds_pages_md/`: Course website content
- `downloads/discourse_summ/`: Processed Q&A pairs
- `downloads/discourse_markdown/`: Markdown versions of forum content

## How the search works

When you ask a question:
1. Your question gets turned into a vector embedding
2. It finds similar content from the indexed materials
3. The most relevant chunks are sent to Gemini to generate a clear answer
4. You get back an answer with links to the original sources

## About the API

The API has these endpoints:
- `/query`: Send questions and get answers
- `/debug`: Test endpoint for troubleshooting

That's it! A simple search system that helps find answers from course materials.

## Deploying to Vercel

The project includes Vercel configuration files for easy deployment:

1. Fork or clone this repository
2. Connect it to your Vercel account
3. Add these environment variables in the Vercel dashboard:
   - `PINECONE_API` - Your Pinecone API key
   - `GEMINI_API_KEY` - Your Gemini API key
   - `PINECONE_INDEX_NAME` - The name of your Pinecone index

After deployment, the API will be available at your Vercel URL. 