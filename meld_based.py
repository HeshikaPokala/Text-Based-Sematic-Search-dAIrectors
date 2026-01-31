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
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ======================================================
# 2. Timestamp conversion
# ======================================================
def time_to_seconds(t):
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# ======================================================
# 3. Load MELD data
# ======================================================
df = pd.read_csv("train/train_sent_emo.csv")

df = df[
    [
        "Dialogue_ID",
        "Utterance_ID",
        "Utterance",
        "StartTime",
        "EndTime"
    ]
]

df["start_sec"] = df["StartTime"].apply(time_to_seconds)
df["end_sec"] = df["EndTime"].apply(time_to_seconds)

print("Total dialogues:", df["Dialogue_ID"].nunique())
print("Total utterances:", len(df))


# ======================================================
# 4. Limit number of dialogues (SAFE FOR HACKATHON)
# ======================================================
NUM_DIALOGUES = None   # change to None for all dialogues

if NUM_DIALOGUES:
    selected_dialogues = df["Dialogue_ID"].unique()[:NUM_DIALOGUES]
    df = df[df["Dialogue_ID"].isin(selected_dialogues)].reset_index(drop=True)

print("Dialogues used:", df["Dialogue_ID"].nunique())
print("Utterances used:", len(df))


# ======================================================
# 5. Create single-utterance clips (NO WINDOW)
# ======================================================
clips = []

for _, row in df.iterrows():
    clips.append({
        "dialogue_id": int(row["Dialogue_ID"]),
        "utterance_ids": [int(row["Utterance_ID"])],
        "start_time": float(row["start_sec"]),
        "end_time": float(row["end_sec"]),
        "text": row["Utterance"]
    })

print("Total clips created:", len(clips))


# ======================================================
# 6. Load embedding model
# ======================================================
model = SentenceTransformer("all-mpnet-base-v2")


# ======================================================
# 7. Generate embeddings (SAFE SETTINGS)
# ======================================================
texts = [clip["text"] for clip in clips]

embeddings = model.encode(
    texts,
    batch_size=8,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embedding shape:", embeddings.shape)


# ======================================================
# 8. Build FAISS index (Cosine Similarity)
# ======================================================
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))

print("Vectors indexed:", index.ntotal)


# ======================================================
# 9. Cosine similarity → confidence
# ======================================================
def cosine_to_confidence(score):
    """
    Convert cosine similarity (0–1) to percentage confidence
    """
    return max(0.0, min(float(score), 1.0)) * 100


# ======================================================
# 10. Search function
# ======================================================
def search(query, top_k=5, min_confidence=0):
    query_emb = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(query_emb, top_k)

    raw_scores = scores[0].tolist()

    results = []
    for i, idx in enumerate(indices[0]):
        clip = clips[idx]
        confidence = cosine_to_confidence(raw_scores[i])

        if confidence < min_confidence:
            continue

        results.append({
            "dialogue_id": clip["dialogue_id"],
            "utterance_ids": clip["utterance_ids"],
            "start_time": clip["start_time"],
            "end_time": clip["end_time"],
            "confidence": round(confidence, 2),
            "raw_score": round(raw_scores[i], 3),
            "text": clip["text"]
        })

    return results


# ======================================================
# 11. Demo queries
# ======================================================
queries = [
    "hesitant reaction before answering",
    "awkward pause",
    "confused reply",
    "avoids answering the question"
]

for q in queries:
    print(f"\nQUERY: {q}")
    results = search(q, top_k=5, min_confidence=35)

    if not results:
        print("No strong matches found.\n")
        continue

    for r in results:
        print(
            f"- Dialogue {r['dialogue_id']} | "
            f"Utterances {r['utterance_ids']} | "
            f"Time [{r['start_time']:.2f}s - {r['end_time']:.2f}s] | "
            f"Confidence {r['confidence']}%"
        )
        print(f"  Text: {r['text']}\n")
