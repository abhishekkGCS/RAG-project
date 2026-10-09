import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Standard FDA Structured Product Labeling (SPL) Sections
FDA_SECTIONS = [
    "BOXED WARNING",
    "HIGHLIGHTS OF PRESCRIBING INFORMATION",
    "INDICATIONS AND USAGE",
    "DOSAGE AND ADMINISTRATION",
    "DOSAGE FORMS AND STRENGTHS",
    "CONTRAINDICATIONS",
    "WARNINGS AND PRECAUTIONS",
    "ADVERSE REACTIONS",
    "DRUG INTERACTIONS",
    "USE IN SPECIFIC POPULATIONS",
    "OVERDOSAGE",
    "DESCRIPTION",
    "CLINICAL PHARMACOLOGY",
    "NONCLINICAL TOXICOLOGY",
    "CLINICAL STUDIES",
    "HOW SUPPLIED/STORAGE AND HANDLING",
    "PATIENT COUNSELING INFORMATION",
]

# Compile a regex to detect section headers (e.g., "4 CONTRAINDICATIONS" or "CONTRAINDICATIONS")
SECTION_PATTERN = re.compile(
    r'^(?:\d+[\.\s]+)?(' + '|'.join(re.escape(sec) for sec in FDA_SECTIONS) + r')\b',
    re.IGNORECASE | re.MULTILINE
)


def extract_drug_name(filename: str) -> str:
    """
    Extracts the primary active pharmaceutical ingredient (API) from filename.
    Example: 'WARFARIN SODIUM tablet.pdf' -> 'Warfarin'
    """
    clean_name = filename.replace(".pdf", "").replace("-", " ")
    first_token = clean_name.split()[0].capitalize()
    return first_token


def chunk_clinical_document(
    cleaned_pages: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Splits clinical pages into section-aware chunks with rich metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    current_section = "GENERAL"

    for page in cleaned_pages:
        page_num = page["page_number"]
        page_text = page["text"]
        source_file = page["source"]
        drug_name = extract_drug_name(source_file)

        # Split page into lines to track section headers
        lines = page_text.split("\n")
        page_buffer = []

        for line in lines:
            match = SECTION_PATTERN.search(line.strip())
            if match:
                # Flush existing buffer under previous section
                if page_buffer:
                    sub_text = "\n".join(page_buffer).strip()
                    if sub_text:
                        split_texts = text_splitter.split_text(sub_text)
                        for txt in split_texts:
                            chunks.append({
                                "text": txt,
                                "metadata": {
                                    "source": source_file,
                                    "drug": drug_name,
                                    "page": page_num,
                                    "section": current_section
                                }
                            })
                    page_buffer = []

                # Update the active section (normalized to uppercase)
                matched_section = match.group(1).upper()
                for standard_sec in FDA_SECTIONS:
                    if standard_sec in matched_section:
                        current_section = standard_sec
                        break

            page_buffer.append(line)

        # Flush any remaining text on the page
        if page_buffer:
            sub_text = "\n".join(page_buffer).strip()
            if sub_text:
                split_texts = text_splitter.split_text(sub_text)
                for txt in split_texts:
                    chunks.append({
                        "text": txt,
                        "metadata": {
                            "source": source_file,
                            "drug": drug_name,
                            "page": page_num,
                            "section": current_section
                        }
                    })

    return chunks


if __name__ == "__main__":
    import os
    from ingestion.loader import load_pdf
    from ingestion.cleaner import clean_pages
    pdf_path = os.path.join("Documents", "WARFARIN SODIUM tablet.pdf")
    if not os.path.exists(pdf_path):
        pdf_path = os.path.join("data", "raw", "warfarin.pdf")
    raw = load_pdf(pdf_path)
    cleaned = clean_pages(raw)
    all_chunks = chunk_clinical_document(cleaned)
    print(f"Total Chunks Generated: {len(all_chunks)}")
    
    # Inspect a few sample chunks
    for i in [0, len(all_chunks) // 2, len(all_chunks) - 1]:
        chunk = all_chunks[i]
        print(f"\n--- Chunk #{i+1} ---")
        print(f"Drug:    {chunk['metadata']['drug']}")
        print(f"Section: {chunk['metadata']['section']}")
        print(f"Page:    {chunk['metadata']['page']}")
        print(f"Text Snippet: {chunk['text'][:180]}...")