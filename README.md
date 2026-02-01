# 🎬 dAIrectors: Semantic Footage Search Engine

A powerful AI-powered search tool designed for **filmmakers seeking inspiration** and **editors during post-production reviews**. It allows creative teams to instantly find specific referencing clips, emotional beats, or dialogue nuances using natural language queries (e.g., *"hesitant before answering"*, *"argument"*, *"grieving"*) instead of manually scrubbing through hours of footage.

## Features

- **Semantic Search**: Understands the *meaning* of your query, not just keyword matching (powered by `sentence-transformers`).
- **Context Awareness**: Analyzes dialogue in context (previous/next lines) to capture the flow of conversation.
- **Nuance Detection**: Specially tuned to detect hesitation ("um...", "uh...", pauses) and emotional cues.
- **Dual Modes**:
  - **Database Search**: Search instantly across your pre-indexed library of movies.
  - **Upload & Search**: Upload a video and `.srt` file to analyze it on the fly.
- **Smart Reranking**: Uses a Cross-Encoder model to deeply verify search candidates for high accuracy.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ripper-7/dAIrectors.git
   cd dAIrectors
   ```

2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Mac/Linux
   # .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the web interface using Streamlit:

```bash
streamlit run frontend.py
```

Opens locally at `http://localhost:8501`.

### Search Modes
- **Library Search**: Type a query like *"argument about money"* or *"hesitant reply"*. The system will rank clips by confidence.
- **Upload & Search**: Drag and drop an `.mp4` and its `.srt`. Click "Analyze". The video player will jump exactly to the relevant timestamp when you click a result.

## Project Structure

- `frontend.py`: The Streamlit User Interface. Handles user interactions, video playback, and displaying results.
- `meld_based.py`: The Logic Core. Handles:
  - Subtitle parsing (`.srt` cleaning & processing).
  - Neural embedding generation (Bi-Encoder `all-mpnet-base-v2`).
  - Search & Reranking (Cross-Encoder `ms-marco-MiniLM-L-6-v2`).
  - Caching logic to speed up startups.
- `requirements.txt`: List of Python libraries required.
- `cache/`: Stores generated embeddings and FAISS index (auto-generated).

## Tech Stack

- **Frontend**: Streamlit
- **Search Backend**: FAISS (Facebook AI Similarity Search)
- **AI Models**: 
  - Retrieval: `all-mpnet-base-v2`
  - Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Language**: Python 3.10+


## DEMO LINK:
https://drive.google.com/file/d/1ChfvODNOQIJSXxUUjt3ACfEbWYICvrYJ/view?usp=sharing
