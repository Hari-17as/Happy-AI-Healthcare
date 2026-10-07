import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_FILE = r"C:\Happy\Medical_Knowledge\_rag_database\medical_index.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


print("=" * 70)
print("HAPPY RAG SEARCH TEST")
print("=" * 70)

print("\nLoading medical database...")
with open(INDEX_FILE, "rb") as f:
    database = pickle.load(f)

chunks = database["chunks"]
embeddings = database["embeddings"]

print(f"Loaded {len(chunks):,} medical chunks.")

print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("\nHappy RAG is ready.")
print("Type a medical question.")
print("Type 'exit' to quit.")
print()


while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    # Create embedding for the question
    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    # Calculate similarity
    scores = np.dot(embeddings, query_embedding)

    # Top 5 results
    top_indices = np.argsort(scores)[::-1][:5]

    print("\n" + "=" * 70)
    print("TOP MEDICAL SOURCES")
    print("=" * 70)

    for rank, index in enumerate(top_indices, 1):

        item = chunks[index]
        score = float(scores[index])

        print(f"\n[{rank}] Similarity: {score:.3f}")
        print(f"Book    : {item['book']}")
        print(f"Subject : {item['subject']}")
        print(f"PDF Page: {item['page']}")
        print("-" * 70)

        # Show a short preview
        preview = item["text"].replace("\n", " ")
        print(preview[:700])

    print()