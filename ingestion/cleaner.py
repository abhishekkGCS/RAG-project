import re
from typing import List, Dict, Any


def clean_text(raw_text: str) -> str:
    """
    Cleans raw PDF text while strictly preserving clinical entities,
    casing, and dosage measurements.
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Normalize unicode characters (like smart quotes, non-breaking spaces)
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Fix words broken across line wraps by a hyphen
    # Example: "hyper-\ntension" -> "hypertension"
    # Matches a lowercase/uppercase letter, hyphen, newline, and letter
    text = re.sub(r'([A-Za-z])-\n([A-Za-z])', r'\1\2', text)

    # 3. Remove standalone separator lines (e.g., "----------", "______", "======")
    text = re.sub(r'^[ \t]*[-_=]{3,}[ \t]*$', '', text, flags=re.MULTILINE)

    # 4. Remove isolated page number lines (e.g., lines containing only "Page 3 of 31" or "3")
    text = re.sub(r'^[ \t]*(?:Page\s+)?\d+(?:\s+of\s+\d+)?[ \t]*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # 5. Collapse excessive horizontal spaces and tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # 6. Collapse 3+ newlines into a standard paragraph break (\n\n)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def clean_pages(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes the output of load_pdf and returns a list of pages with cleaned text,
    filtering out completely empty pages.
    """
    cleaned_pages = []
    for page in pages_data:
        cleaned_str = clean_text(page.get("text", ""))
        
        # Keep page if it has substantive content (more than 10 chars)
        if len(cleaned_str) > 10:
            cleaned_pages.append({
                "page_number": page["page_number"],
                "text": cleaned_str,
                "source": page["source"]
            })
            
    return cleaned_pages


if __name__ == "__main__":
    import os
    from ingestion.loader import load_pdf

    # Test path - adjust filename to match your file
    pdf_path = os.path.join("Documents", "WARFARIN SODIUM tablet.pdf")
    if not os.path.exists(pdf_path):
        pdf_path = os.path.join("data", "raw", "warfarin.pdf")

    raw_pages = load_pdf(pdf_path)
    cleaned = clean_pages(raw_pages)
    
    print(f"Original pages: {len(raw_pages)} | Non-empty cleaned pages: {len(cleaned)}")
    print("\n--- Cleaned Page 1 (First 350 chars) ---")
    print(cleaned[0]["text"][:350])