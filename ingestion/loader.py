import os
from typing import List, Dict, Any
from pypdf import PdfReader


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF document.
    
    Args:
        file_path: Relative or absolute path to the PDF.
        
    Returns:
        List of dicts containing page_number (1-indexed), raw text, and source filename.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    filename = os.path.basename(file_path)
    reader = PdfReader(file_path)
    pages_data = []

    for index, page in enumerate(reader.pages):
        # extract_text can return None on scanned or image-only pages
        raw_text = page.extract_text() or ""
        
        pages_data.append({
            "page_number": index + 1,  # 1-indexed for human-readable clinical citations
            "text": raw_text,
            "source": filename
        })

    return pages_data


if __name__ == "__main__":
    # Quick sanity test
    test_path = os.path.join("data", "raw", "warfarin.pdf")
    pages = load_pdf(test_path)
    print(f"Loaded {len(pages)} pages from {test_path}")
    if pages:
        print(f"--- Sample Page 1 (First 300 chars) ---\n{pages[0]['text'][:300]}")