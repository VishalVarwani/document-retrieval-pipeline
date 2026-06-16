# textbased.py

import pdfplumber
from pathlib import Path
from detect_text_structure import detect_text_structure
from extract_text_blocks import extract_text_blocks, mark_continuations
import json
from detect_table_regions import (
    detect_table_regions,
    filter_real_tables,
    block_overlaps_table
)
MIN_CHAR_COUNT = 30
MIN_WORD_COUNT = 5
def clean_text_structure(text_structure):
    """
    Keeps useful text structure info.
    Removes heavy debug data like full words/coordinates.
    """

    if not text_structure:
        return None

    return {
        "text_structure": text_structure.get("text_structure"),
        "reason": text_structure.get("reason"),
        "mixed_types": text_structure.get("mixed_types", []),
        "scores": text_structure.get("scores", {}),
        "region_summary": text_structure.get("region_summary", {})
    }


def clean_page_result(page_result):
    """
    Keeps only final ingestion output.
    Removes repeated debug-heavy signals.
    """

    return {
        "page_number": page_result.get("page_number"),
        "page_type": page_result.get("page_type"),
        "has_meaningful_text": page_result.get("has_meaningful_text"),
        "has_some_text": page_result.get("has_some_text"),
        "has_images": page_result.get("has_images"),
        "char_count": page_result.get("char_count"),
        "word_count": page_result.get("word_count"),
        "image_count": page_result.get("image_count"),
        

        "text_structure": clean_text_structure(
            page_result.get("text_structure")
        ),
        

        "text_blocks": page_result.get("text_blocks", []),
        "tables": page_result.get("tables", [])
    }


def save_results_to_json(results, output_path="pdf_ingestion_output.json"):
    """
    Saves clean JSON output.
    Full block text is kept.
    Heavy debug word-level data is removed.
    """

    clean_results = [
        clean_page_result(page_result)
        for page_result in results
    ]

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(clean_results, file, indent=4, ensure_ascii=False)

    print(f"\nClean JSON output saved to: {output_path}")
def mark_table_blocks(text_blocks, tables):
    """
    Marks text blocks as table only if they overlap
    an accepted real table region.
    """

    for block in text_blocks:
        for table in tables:
            table_bbox = table.get("bbox")

            if table_bbox and block_overlaps_table(block, table_bbox):
                block["role"] = "table"
                break

    return text_blocks
def classify_page(page):
    """
    First detects page content type:
    - text_based
    - image_based
    - mixed
    - low_text_page
    - blank_or_unknown

    If the page has enough text, it also detects internal text structure:
    - paragraph_based
    - table_based
    - structured_list
    - form_like
    - mixed_text_structure
    - unknown_text_structure
    """

    text = page.extract_text() or ""
    clean_text = text.strip()

    words = page.extract_words() or []
    chars = page.chars or []
    images = page.images or []

    char_count = len(clean_text)
    word_count = len(words)
    pdf_char_object_count = len(chars)
    image_count = len(images)

    has_meaningful_text = (
        char_count >= MIN_CHAR_COUNT and
        word_count >= MIN_WORD_COUNT
    )

    has_some_text = (
        char_count > 0 or
        word_count > 0 or
        pdf_char_object_count > 0
    )
    
    has_images = image_count > 0

    if has_meaningful_text and has_images:
        page_type = "mixed"

    elif has_meaningful_text:
        page_type = "text_based"

    elif has_some_text and has_images:
        page_type = "image_with_low_text"

    elif has_some_text and not has_meaningful_text:
        page_type = "low_text_page"

    elif not has_meaningful_text and has_images:
        page_type = "image_based"

    else:
        page_type = "blank_or_unknown"

    text_structure_result = None

    if page_type in ["text_based", "mixed", "low_text_page"]:
        text_structure_result = detect_text_structure(page)
    text_blocks = []

    if page_type in ["text_based", "mixed", "low_text_page", "image_with_low_text"]:
        structure_name = None

        if text_structure_result:
            structure_name = text_structure_result.get("text_structure")

        text_blocks = extract_text_blocks(page, page_text_structure=structure_name)
    table_candidates = detect_table_regions(page)

    tables = filter_real_tables(
        table_candidates,
        page_text_structure=text_structure_result
    )

    text_blocks = mark_table_blocks(text_blocks, tables)
    

    return {
        "page_number": page.page_number,
        "page_type": page_type,
        "has_meaningful_text": has_meaningful_text,
        "has_some_text": has_some_text,
        "has_images": has_images,
        "char_count": char_count,
        "word_count": word_count,
        "pdf_char_object_count": pdf_char_object_count,
        "image_count": image_count,
        "text_structure": text_structure_result,
        "text_blocks": text_blocks,
        "tables": tables
    }


def analyze_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_result = classify_page(page)
            results.append(page_result)
    results = mark_continuations(results)

    return results


def print_results(results):
    print("\nPDF Page Detection Results")
    print("-" * 100)

    for result in results:
        page_number = result["page_number"]
        page_type = result["page_type"]
        chars = result["char_count"]
        words = result["word_count"]
        images = result["image_count"]
        tables = result.get("tables", [])
        table_count = len(tables)

        if result["text_structure"]:
            structure_data = result["text_structure"]
            structure = structure_data["text_structure"]
            scores = structure_data.get("scores", {})
            mixed_types = structure_data.get("mixed_types", [])
            text_blocks = result.get("text_blocks", [])
            block_count = len(text_blocks)
        else:
            structure = "not_checked"
            scores = {}
            mixed_types = []

        print(
            f"Page {page_number}: "
            f"page_type={page_type} | "
            f"text_structure={structure} | "
            f"mixed_types={mixed_types} | "
            f"scores={scores} | "
            f"chars={chars} | "
            f"words={words} | "
            f"images={images} | "
            f"text_blocks={block_count} | "
            f"tables={table_count} | "
        )
        for block in text_blocks:
            preview = block["text"].replace("\n", " ")[:120]

            print(
    f"   Block {block['block_id']} | "
    f"role={block['role']} | "
    f"continuation={block['is_continuation']} | "
    f"lines={block['line_count']} | "
    f"words={block['word_count']} | "
    f"text=\"{preview}...\""
            )

if __name__ == "__main__":
    pdf_file = "9789240117808-eng-1-30.pdf"  

    results = analyze_pdf(pdf_file)
    print_results(results)
    save_results_to_json(results)