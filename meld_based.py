# ======================================================
# 0. CRASH FIXES (MUST BE AT VERY TOP)
# ======================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

# ======================================================
# 1. Imports
# ======================================================
import re
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ======================================================
# 2. Timestamp conversion
# ======================================================
def time_to_seconds(t):
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# ======================================================
# 3. Parse SRT file (robust)
# ======================================================
    return clips

def parse_srt(file_path):
    clips = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        time_range = lines[1].strip()
        if "-->" not in time_range:
            continue

        start, end = [t.strip() for t in time_range.split("-->")]

        raw_text = " ".join(
            line.strip() for line in lines[2:]
            if line.strip() and not line.strip().isdigit()
        )

        if not raw_text:
            continue

        # Clean text for better semantic understanding
        # 1. Remove HTML tags like <i>...</i>
        clean_text = re.sub(r'<[^>]+>', '', raw_text)
        # 2. Remove hearing-impaired descriptors like (MUSIC), [APPLAUSE]
        clean_text = re.sub(r'[\(\[][^)]*[\)\]]', '', clean_text)
        # 3. Remove speaker labels like "JOHN:"
        clean_text = re.sub(r'^[A-Z\s]+:', '', clean_text)
        # 4. Remove leading hyphens
        clean_text = re.sub(r'^\s*-\s*', '', clean_text)
        # 5. Collapse whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        if not clean_text:
            continue

        clips.append({
            "movie": os.path.basename(file_path),
            "start_sec": time_to_seconds(start),
            "end_sec": time_to_seconds(end),
            "start_time": start,
            "end_time": end,
            "text": clean_text
        })
    return clips


# ======================================================
# 4. Load all subtitle files
# ======================================================
def load_all_subtitles():
    SUBTITLE_DIR = "Processed Movie Subtitles"
    raw_clips = []

    for file in os.listdir(SUBTITLE_DIR):
        if file.endswith(".srt"):
            path = os.path.join(SUBTITLE_DIR, file)
            raw_clips.extend(parse_srt(path))

    print(f"Raw subtitle clips loaded: {len(raw_clips)}")
    return raw_clips


# ======================================================
# 5. Build CONTEXTUAL clips
# ======================================================
def build_contextual_clips(clips, window=3):
    """
    Groups subtitles into larger context windows.
    Default window=3 means: [Prev3, Prev2, Prev1, CURRENT, Next1, Next2, Next3]
    (Total 7 lines of context roughly)
    """
    contextual = []

    for i in range(len(clips)):
        start_idx = max(0, i - window)
        end_idx = min(len(clips), i + window + 1)

        merged_text = " ".join(
            clips[j]["text"] for j in range(start_idx, end_idx)
        )

        contextual.append({
            "movie": clips[i]["movie"],
            "start_time": clips[start_idx]["start_time"],
            "end_time": clips[end_idx - 1]["end_time"],
            "start_sec": clips[start_idx]["start_sec"],
            "end_sec": clips[end_idx - 1]["end_sec"],
            "text": merged_text
        })

    return contextual


# ======================================================
# 6. Behavioral bonus
# ======================================================
def behavior_bonus(text):
    bonus = 0.0
    t = text.strip()
    tl = t.lower()
    words = t.split()

    if "..." in t:
        bonus += 0.06
    if t.endswith(("…", ".", ",")):
        bonus += 0.03
    if any(tl.startswith(w) for w in ["uh", "um", "well", "so ", "okay", "hmm"]):
        bonus += 0.05
    if t.count("?") >= 2:
        bonus += 0.05
    if any(p in tl for p in ["what do you mean", "are you saying", "how is that", "why would"]):
        bonus += 0.04
    if "..." in t and "?" in t:
        bonus += 0.04
    if len(words) <= 4 and "?" not in t:
        bonus += 0.04
    if any(c in t for c in ["—", "--"]):
        bonus += 0.04
    if "," in t and len(words) > 6:
        bonus += 0.02
    if t.endswith("?") and not tl.startswith(("why", "what", "how")):
        bonus += 0.03
    if any(tl.startswith(w) for w in ["i think", "maybe", "it depends", "sort of"]):
        bonus += 0.05
    if "!" in t and "?" in t:
        bonus += 0.04
    if t.isupper() and len(words) > 2:
        bonus += 0.03
    if any(w in tl for w in ["wait", "hold on", "let me think", "give me a second"]):
        bonus += 0.05
    if t.count(",") >= 2:
        bonus += 0.03
    if t.startswith(("...", "—", "-")):
        bonus += 0.05
    if len(words) <= 3 and t.endswith("."):
        bonus += 0.04
    if t.startswith(("but", "and", "no,")):
        bonus += 0.03
    if "/" in t or t.count("-") >= 2:
        bonus += 0.03
    if len(words) <= 5 and "..." in t:
        bonus += 0.05
    if len(words) >= 12 and t.count(",") >= 2:
        bonus += 0.04

    return min(bonus, 0.25)


# ======================================================
# 7. Confidence conversion
# ======================================================
def cosine_to_confidence(score):
    return max(0.0, min(float(score), 1.0)) * 100


# ======================================================
# 8. Initialize model and index (GLOBAL)
# ======================================================
# ======================================================
# 8. Initialize model and index (GLOBAL)
# ======================================================
CACHE_DIR = "cache"
CLIPS_FILE = os.path.join(CACHE_DIR, "clips_v3.pkl") # v3 forces rebuild for cleaning/context
INDEX_FILE = os.path.join(CACHE_DIR, "behavenet_v3.index")

def initialize_system():
    global model, cross_model, index, clips
    
    # 1. Load Bi-Encoder (Fast enough to load every time, ~2-5s)
    print("Loading Bi-Encoder (MPNet)...")
    model = SentenceTransformer("all-mpnet-base-v2")

    # 2. Load Cross-Encoder (For Reranking)
    print("Loading Cross-Encoder (MS-MARCO)...")
    try:
        cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    except Exception as e:
        print(f"⚠️ Warning: Could not load Cross-Encoder ({e}). Reranking will be disabled.")
        cross_model = None
    
    # 3. Check Cache
    if os.path.exists(CLIPS_FILE) and os.path.exists(INDEX_FILE):
        print("Found cached data. Loading...")
        try:
            with open(CLIPS_FILE, "rb") as f:
                clips = pickle.load(f)
            
            index = faiss.read_index(INDEX_FILE)
            print(f"✅ Loaded {len(clips)} clips and index with {index.ntotal} vectors from cache.")
            return
        except Exception as e:
            print(f"Error loading cache: {e}. Rebuilding...")
    
    # 4. Generate if no cache
    print("No cache found (or cache outdated). Generating embeddings with clean data & larger context...")
    
    # Ensure cache dir exists
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    raw_clips = load_all_subtitles()
    clips = build_contextual_clips(raw_clips, window=3) # Increased window to 3
    print(f"Contextual clips created: {len(clips)}")
    
    # Generate embeddings
    texts = [clip["text"] for clip in clips]
    embeddings = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    
    # Save to cache
    print("Saving data to cache...")
    with open(CLIPS_FILE, "wb") as f:
        pickle.dump(clips, f)
    
    faiss.write_index(index, INDEX_FILE)
    print("✅ System initialized and cached!")

# Run initialization
initialize_system()


# ======================================================
# 9. Search function (EXPORTABLE)
# ======================================================
def search(query, top_k=50, min_confidence=40):
    """
    Semantic search function with Cross-Encoder Reranking and Fallback
    """
    # 1. Retrieve Candidates (Bi-Encoder)
    candidate_k = top_k * 5 
    query_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_emb, candidate_k)

    results = []
    
    # Check if we can use Cross-Encoder
    use_rerank = True
    if cross_model is None:
        use_rerank = False
    
    # Prepare candidates
    candidates = []
    for i, idx in enumerate(indices[0]):
        clip = clips[idx]
        candidates.append(clip)
        
    final_results = []
    
    if use_rerank and candidates:
        try:
            # Rerank
            model_inputs = [[query, clip["text"]] for clip in candidates]
            cross_scores = cross_model.predict(model_inputs)
            
            for i, clip in enumerate(candidates):
                score = cross_scores[i]
                # Shifted sigmoid
                adj_score = score + 1.5
                confidence = (1 / (1 + np.exp(-adj_score))) * 100
                bonus = behavior_bonus(clip["text"]) * 10
                confidence += bonus
                
                if confidence < min_confidence: continue
                
                final_results.append({
                    "movie": clip["movie"],
                    "start_time": clip["start_time"],
                    "end_time": clip["end_time"],
                    "start_sec": int(clip["start_sec"]),
                    "end_sec": clip["end_sec"],
                    "confidence": round(confidence, 2),
                    "text": clip["text"]
                })
        except Exception as e:
            print(f"Reranking failed ({e}), falling back.")
            final_results = [] # Force fallback
            
    # FAILSAFE / FALLBACK
    if not final_results:
        print("⚠️ Reranking yielded no results (or disabled). Using Bi-Encoder.")
        for i, idx in enumerate(indices[0]):
            if i >= top_k: break # Only take top K for fallback
            raw_score = float(scores[0][i])
            clip = clips[idx]
            conf = cosine_to_confidence(raw_score)
            if conf < min_confidence: continue
            
            final_results.append({
                "movie": clip["movie"],
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "start_sec": int(clip["start_sec"]),
                "end_sec": clip["end_sec"],
                "confidence": round(conf, 2),
                "text": clip["text"]
            })

    # Sort by new confidence
    final_results = sorted(final_results, key=lambda x: x["confidence"], reverse=True)
    
    # 6. DEDUPLICATION (Remove overlapping clips)
    unique_results = []
    seen_intervals = {} # movie -> list of (start, end)
    
    for res in final_results:
        movie = res["movie"]
        start = res["start_sec"]
        end = res["end_sec"]
        
        # Check against existing results for this movie
        is_duplicate = False
        if movie in seen_intervals:
            for (s, e) in seen_intervals[movie]:
                # Overlap calculation
                # If intervals overlap significantly or are too close (within 15s)
                # Simple check: timestamp distance
                if abs(start - s) < 15: # 15 second buffer
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_results.append(res)
            if movie not in seen_intervals:
                seen_intervals[movie] = []
            seen_intervals[movie].append((start, end))
            
    return unique_results[:top_k]


# ======================================================
# 10. Get dataset info (EXPORTABLE)
# ======================================================
def get_dataset_info():
    """Get dataset statistics"""
    return {
        "total_clips": len(clips),
        "unique_movies": len(set(c['movie'] for c in clips)),
        "model_name": "all-mpnet-base-v2",
        "embedding_dim": index.d
    }


# ======================================================
# 11. Interactive CLI (for testing)
# ======================================================
if __name__ == "__main__":
    print("\n🎬 Semantic Footage Search Engine (CLI Mode)")
    print("Type a natural language query (or 'exit' to quit)\n")

    while True:
        user_query = input("🔎 Enter your query: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("\nExiting search. Goodbye 👋")
            break

        print("\n" + "=" * 80)
        print(f"🔎 QUERY: {user_query.upper()}")
        print("=" * 80)

        results = search(user_query)

        if not results:
            print("No strong matches found.\n")
            continue

        top_5 = results[:5]

        for i, r in enumerate(top_5, 1):
            print(f"\n{i}. 🎬 MOVIE       : {r['movie']}")
            print(f"   ⏱️  TIMESTAMPS  : {r['start_time']} → {r['end_time']}")
            print(f"   📊 CONFIDENCE  : {r['confidence']}%")
            print(f"   💬 DIALOGUE    : {r['text']}")

        print("\n")