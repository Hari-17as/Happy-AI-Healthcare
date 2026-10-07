from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_FOLDER = Path(r"C:\Happy\Medical_Knowledge\01_Anatomy\extracted_text")

documents = []
sources = []

for text_file in TEXT_FOLDER.glob("*.txt"):
    text = text_file.read_text(encoding="utf-8", errors="ignore")

    # Split into smaller sections
    chunks = text.split("\n\n--- PAGE ")

    for chunk in chunks:
        if chunk.strip():
            documents.append(chunk)
            sources.append(text_file.name)

print(f"Loaded {len(documents)} anatomy sections.")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

matrix = vectorizer.fit_transform(documents)

print("Anatomy search system is ready.")
print("Type 'exit' to stop.\n")

while True:
    question = input("Search anatomy: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    question_vector = vectorizer.transform([question])
    scores = matrix @ question_vector.T
    scores = scores.toarray().flatten()

    best_indexes = scores.argsort()[-5:][::-1]

    print("\nRelevant information:\n")

    for index in best_indexes:
        if scores[index] > 0:
            print("=" * 60)
            print(f"Source: {sources[index]}")
            print(documents[index][:1500])
            print()