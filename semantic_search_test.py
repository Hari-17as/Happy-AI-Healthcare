from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np


# =========================
# MEDICAL TEXT
# =========================

KNOWLEDGE_FOLDERS = [
    Path(r"C:\Happy\Medical_Knowledge\01_Anatomy\extracted_text"),
    Path(r"C:\Happy\Medical_Knowledge\02_Embryology\extracted_text"),
]


documents = []
sources = []


print("Loading medical books...")


for folder in KNOWLEDGE_FOLDERS:

    for text_file in folder.glob("*.txt"):

        print("Loading:", text_file.name)

        text = text_file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        chunks = text.split("\n\n--- PAGE ")

        for chunk in chunks:

            if not chunk.strip():
                continue

            # Ignore very short sections
            if len(chunk.strip()) < 200:
                continue

        # Ignore figure captions
        if chunk.lstrip().upper().startswith("FIGURE"):
            continue

        # Ignore bibliography/reference sections
        lower_chunk = chunk.lower()

        reference_words = [
            "references",
            "bibliography",
            "selected references",
            "suggested readings"
        ]

        if any(
            word in lower_chunk[:500]
            for word in reference_words
        ):
            continue

        documents.append(chunk)
        sources.append(text_file.name)

print(
    f"Loaded {len(documents)} useful medical sections."
)


# =========================
# EMBEDDING MODEL
# =========================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# CREATE EMBEDDINGS
# =========================

print("Creating medical embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Semantic search system ready.")


# =========================
# SEARCH
# =========================

while True:

    question = input(
        "\nAsk a medical question "
        "(type exit to stop): "
    )

    if question.lower() == "exit":
        break


    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]


    scores = np.dot(
        embeddings,
        question_embedding
    )


    best_indexes = np.argsort(
        scores
    )[-5:][::-1]


    print("\nTop medical matches:\n")


    for index in best_indexes:

        print(
            "SOURCE:",
            sources[index]
        )

        print(
            "SCORE:",
            round(
                float(scores[index]),
                3
            )
        )

        # Remove page marker from display
        text = documents[index].strip()

        lines = text.splitlines()

        if lines and "---" in lines[0]:
            text = "\n".join(lines[1:]).strip()

        print(
            text[:700]
        )

        print(
            "\n"
            + "-" * 60
            + "\n"
        )