from pypdf import PdfReader
from pathlib import Path

input_folder = Path(r"C:\Happy\Medical_Knowledge\01_Anatomy")
output_folder = Path(r"C:\Happy\Medical_Knowledge\01_Anatomy\extracted_text")

output_folder.mkdir(exist_ok=True)

for pdf_file in input_folder.glob("*.pdf"):
    print(f"\nReading: {pdf_file.name}")

    reader = PdfReader(str(pdf_file))
    all_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            all_text.append(
                f"\n\n--- PAGE {page_number} ---\n\n{text}"
            )
        except Exception as error:
            print(f"Could not read page {page_number}: {error}")

    output_file = output_folder / (pdf_file.stem + ".txt")
    output_file.write_text(
        "".join(all_text),
        encoding="utf-8"
    )

    print(f"Saved: {output_file.name}")

print("\nFinished! Anatomy text extraction is complete.")