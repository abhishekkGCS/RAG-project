import os
import json
import glob
from typing import List, Dict, Any

from ingestion.loader import load_pdf
from ingestion.cleaner import clean_pages
from ingestion.chunker import chunk_clinical_document


def run_ingestion_pipeline(
    raw_dir: str = "document",
    output_file: str = os.path.join("data", "processed", "chunks.json")
) -> List[Dict[str, Any]]:
    """
    Executes the end-to-end ingestion pipeline:
    1. Finds all PDFs in raw_dir
    2. Extracts page-level text
    3. Cleans clinical artifacts
    4. Generates section-aware chunks
    5. Saves structured chunks to JSON
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Find all PDFs in the directory
    pdf_paths = glob.glob(os.path.join(raw_dir, "*.pdf"))
    
    # Fallback to check Documents folder if data/raw is empty
    if not pdf_paths and os.path.exists("Documents"):
        pdf_paths = glob.glob(os.path.join("Documents", "*.pdf"))

    if not pdf_paths:
        print(f"[!] No PDFs found in {raw_dir} or Documents/")
        return []

    print(f"[*] Found {len(pdf_paths)} PDF(s) to process.")
    
    total_pages = 0
    all_chunks = []

    for path in pdf_paths:
        print(f"\n---> Processing: {os.path.basename(path)}")
        
        # 1. Load
        raw_pages = load_pdf(path)
        total_pages += len(raw_pages)
        print(f"     Extracted: {len(raw_pages)} pages")

        # 2. Clean
        cleaned = clean_pages(raw_pages)
        print(f"     Cleaned:   {len(cleaned)} pages")

        # 3. Chunk
        doc_chunks = chunk_clinical_document(cleaned)
        print(f"     Chunks:    {len(doc_chunks)} chunks created")

        all_chunks.extend(doc_chunks)

    # 4. Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"Ingestion Pipeline Complete!")
    print(f"Total PDFs Processed:  {len(pdf_paths)}")
    print(f"Total Pages Extracted: {total_pages}")
    print(f"Total Chunks Created:  {len(all_chunks)}")
    print(f"Saved artifacts to:    {output_file}")
    print("=" * 50)

    return all_chunks


if __name__ == "__main__":
    run_ingestion_pipeline()