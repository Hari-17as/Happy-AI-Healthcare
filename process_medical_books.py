import os
import re
import json
import pickle
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# HAPPY MEDICAL KNOWLEDGE PROCESSOR
# ============================================================

BASE_DIR = Path(r"C:\Happy\Medical_Knowledge")
OUTPUT_DIR = BASE_DIR / "_rag_database"

TEXT_DIR = OUTPUT_DIR / "text"
INDEX_FILE = OUTPUT_DIR / "medical_index.pkl"
META_FILE = OUTPUT_DIR / "medical_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunk settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# ============================================================
# FOLDERS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove common PDF junk
    junk_patterns = [
        r"www\.[^\s]+",
        r"https?://[^\s]+",
        r"ISBN[\s:0-9Xx\-]+",
        r"©\s*\d{4}",
    ]

    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove repeated blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


# ============================================================
# CHECK WHETHER A PAGE LOOKS LIKE JUNK
# ============================================================

def is_junk_page(text):
    if not text:
        return True

    stripped = text.strip()

    # Very short pages
    if len(stripped) < 150:
        return True

    lower = stripped.lower()

    junk_words = [
        "table of contents",
        "contents",
        "index",
        "bibliography",
        "references",
        "acknowledgments",
        "acknowledgements",
        "copyright",
        "all rights reserved",
        "activation code",
        "online access",
    ]

    # Don't automatically reject every page containing these words.
    # Only reject when the page is dominated by them.
    for word in junk_words:
        if lower.startswith(word) and len(stripped) < 1200:
            return True

    return False


# ============================================================
# CHUNK TEXT
# ============================================================

def make_chunks(text, book_name, subject, page_number):
    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)

        # Try to end at a sentence/paragraph boundary
        if end < text_length:
            possible_end = text.rfind(". ", start, end)

            if possible_end > start + 500:
                end = possible_end + 1
            else:
                possible_end = text.rfind("\n", start, end)

                if possible_end > start + 500:
                    end = possible_end

        chunk_text = text[start:end].strip()

        if len(chunk_text) >= 100:
            chunks.append({
                "text": chunk_text,
                "book": book_name,
                "subject": subject,
                "page": page_number
            })

        next_start = end - CHUNK_OVERLAP

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(pdf_path):

    print()
    print("=" * 70)
    print(f"Processing: {pdf_path.name}")
    print("=" * 70)

    # Subject = folder immediately above PDF
    subject = pdf_path.parent.name

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"ERROR opening PDF: {e}")
        return []

    print(f"Pages: {len(reader.pages)}")

    all_chunks = []

    for page_index, page in enumerate(reader.pages):

        page_number = page_index + 1

        try:
            raw_text = page.extract_text() or ""
        except Exception as e:
            print(f"  Page {page_number}: extraction error: {e}")
            continue

        text = clean_text(raw_text)

        if is_junk_page(text):
            print(f"  Page {page_number}: skipped")
            continue

        chunks = make_chunks(
            text=text,
            book_name=pdf_path.name,
            subject=subject,
            page_number=page_number
        )

        all_chunks.extend(chunks)

        print(
            f"  Page {page_number}: "
            f"{len(text):,} chars -> "
            f"{len(chunks)} chunks"
        )

    print(f"Chunks from book: {len(all_chunks)}")

    return all_chunks


# ============================================================
# FIND ALL PDFs
# ============================================================

def find_pdfs():

    pdfs = []

    for path in BASE_DIR.rglob("*.pdf"):

        # Never process our output directory
        if OUTPUT_DIR in path.parents:
            continue

        pdfs.append(path)

    return sorted(pdfs)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HAPPY MEDICAL KNOWLEDGE PROCESSOR")
    print("=" * 70)
    print()
    print(f"Medical library: {BASE_DIR}")
    print()

    pdfs = find_pdfs()

    print(f"PDF files found: {len(pdfs)}")
    print()

    if not pdfs:
        print("No PDF files found.")
        return

    all_chunks = []

    # --------------------------------------------------------
    # Extract all books
    # --------------------------------------------------------

    for pdf in pdfs:

        chunks = process_pdf(pdf)

        all_chunks.extend(chunks)

    print()
    print("=" * 70)
    print("TEXT EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Total chunks: {len(all_chunks):,}")

    if not all_chunks:
        print("No usable text was extracted.")
        return

    # --------------------------------------------------------
    # Save extracted chunks
    # --------------------------------------------------------

    chunks_file = TEXT_DIR / "medical_chunks.json"

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved chunks: {chunks_file}")

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING EMBEDDING MODEL")
    print("=" * 70)

    print(f"Model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING MEDICAL EMBEDDINGS")
    print("=" * 70)

    texts = [item["text"] for item in all_chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # Build searchable database
    # --------------------------------------------------------

    database = {
        "model": EMBEDDING_MODEL,
        "chunks": all_chunks,
        "embeddings": embeddings
    }

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(database, f)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    books = {}

    for item in all_chunks:

        book = item["book"]
        subject = item["subject"]

        if subject not in books:
            books[subject] = {}

        if book not in books[subject]:
            books[subject][book] = 0

        books[subject][book] += 1

    metadata = {
        "total_pdfs": len(pdfs),
        "total_chunks": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL,
        "subjects": books
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HAPPY MEDICAL KNOWLEDGE DATABASE COMPLETE")
    print("=" * 70)

    print()
    print(f"PDFs processed : {len(pdfs)}")
    print(f"Chunks created : {len(all_chunks):,}")
    print()
    print(f"Database:")
    print(INDEX_FILE)
    print()
    print("Happy is ready for the RAG testing stage.")
    print()


if __name__ == "__main__":
    main()