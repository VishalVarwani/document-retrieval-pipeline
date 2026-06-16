# llm_cleaner_groq.py

import os
import json
from groq import Groq


MODEL_NAME = "llama-3.3-70b-versatile"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"LLM cleaned output saved to: {path}")


def get_groq_client():
    """
    Creates Groq client from environment variable.
    """

    api_key = ""  # Replace with your actual API key or set it as an environment variable

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    return Groq(api_key=api_key)


def copy_block_without_llm(block):
    """
    Copies normal blocks directly without using LLM.

    Used for:
    - heading
    - subheading
    - paragraph
    - footer
    - unknown
    """

    return {
        "block_id": block.get("block_id"),
        "role": block.get("role", ""),
        "text": block.get("text", ""),
        "is_continuation": block.get("is_continuation", False),
        "continues_from_previous_page": block.get("continues_from_previous_page", False),
        "continues_after_visual": block.get("continues_after_visual", False)
    }


def should_clean_block_with_llm(block):
    """
    Only list-like and table-like text blocks are sent to LLM.
    """

    role = str(block.get("role", "")).lower().strip()

    llm_roles = {
        "list",
        "structured_list",
        "list_structure",
        "table"
    }

    return role in llm_roles


def build_block_prompt(block):
    """
    Builds prompt for cleaning only one list/table text block.
    """

    return f"""
You are a strict PDF block cleaner.

Your job is ONLY to clean this one extracted text block.
Return valid JSON only.

Rules:
- Do not invent information.
- Do not translate.
- Keep the original document language.
- Keep original block_id.
- Keep original role exactly as given.
- Keep all continuation flags exactly as given.
- Do not remove content.
- Do not add new content.
- Do not summarize.
- Do not change numbers, units, symbols, article numbers, table values, or codes.
- Preserve symbols such as "-", "*", "**", "<", ">", "%", commas, decimal commas, and decimal points.
- Only clean obvious spacing problems or broken line breaks inside the same block.
- If unsure, keep text unchanged and add a warning.

Return exactly this JSON structure:

{{
  "block_id": null,
  "role": "",
  "text": "",
  "is_continuation": false,
  "continues_from_previous_page": false,
  "continues_after_visual": false,
  "warning": null
}}

Input block JSON:
{json.dumps(block, ensure_ascii=False)}
"""


def clean_block_with_llm(client, block):
    prompt = build_block_prompt(block)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON-only PDF block cleaner."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


def get_table_source_rows(table):
    """
    Fallback source if LLM table cleaning fails.
    """

    raw_table = table.get("raw_table")
    rough_matrix = table.get("rough_matrix")

    if raw_table:
        return raw_table

    if rough_matrix:
        return rough_matrix

    return []


def build_table_prompt(table):
    """
    Builds prompt for cleaning only one detected table.
    """

    return f"""
You are a strict PDF table cleaner.

Your job is ONLY to clean this one detected table.
Return valid JSON only.

Rules:
- Do not invent headers.
- Do not invent missing values.
- Do not remove rows.
- Do not remove columns.
- Do not summarize.
- Do not translate.
- Keep original table_id.
- Preserve all numbers, symbols, units, commas, decimal commas, decimal points, %, -, *, **, <, >.
- Convert null cells to empty strings "".
- Preserve empty cells as empty strings "".
- Clean only obvious spacing problems inside cells.
- Do not shift values between columns.
- If raw_table exists and contains rows, use raw_table as the primary source.
- If raw_table is missing or empty, use rough_matrix as fallback.
- If both raw_table and rough_matrix exist, prefer raw_table but use rough_matrix to understand broken spacing.
- If headers are unclear, multi-row, or merged, keep "headers": [] and put header-like rows inside "rows".
- If the first row is clearly a simple header row, place it in "headers".
- If the table is complex, keep the data and add a note instead of simplifying it.

Return exactly this JSON structure:

{{
  "table_id": null,
  "headers": [],
  "rows": [],
  "notes": null
}}

Input table JSON:
{json.dumps(table, ensure_ascii=False)}
"""


def clean_table_with_llm(client, table):
    prompt = build_table_prompt(table)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON-only PDF table cleaner."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


def clean_page_selectively(client, page):
    """
    Creates one final cleaned page.

    Logic:
    - Normal text blocks are copied directly.
    - Only list/table blocks are cleaned with LLM.
    - Detected tables are cleaned with LLM.
    """

    page_number = page.get("page_number")

    raw_blocks = page.get("text_blocks", [])
    raw_tables = page.get("tables", [])

    clean_blocks = []
    clean_tables = []
    warnings = []

    for block in raw_blocks:
        try:
            if should_clean_block_with_llm(block):
                cleaned_block = clean_block_with_llm(client, block)

                warning = cleaned_block.get("warning")
                if warning:
                    warnings.append(
                        f"Page {page_number}, block {block.get('block_id')}: {warning}"
                    )

                cleaned_block.pop("warning", None)

                clean_blocks.append({
                    "block_id": cleaned_block.get("block_id", block.get("block_id")),
                    "role": cleaned_block.get("role", block.get("role", "")),
                    "text": cleaned_block.get("text", block.get("text", "")),
                    "is_continuation": cleaned_block.get(
                        "is_continuation",
                        block.get("is_continuation", False)
                    ),
                    "continues_from_previous_page": cleaned_block.get(
                        "continues_from_previous_page",
                        block.get("continues_from_previous_page", False)
                    ),
                    "continues_after_visual": cleaned_block.get(
                        "continues_after_visual",
                        block.get("continues_after_visual", False)
                    )
                })

            else:
                clean_blocks.append(copy_block_without_llm(block))

        except Exception as error:
            warnings.append(
                f"LLM block cleaning failed on page {page_number}, "
                f"block {block.get('block_id')}: {error}"
            )
            clean_blocks.append(copy_block_without_llm(block))

    for table in raw_tables:
        try:
            cleaned_table = clean_table_with_llm(client, table)

            clean_tables.append({
                "table_id": cleaned_table.get("table_id", table.get("table_id")),
                "headers": cleaned_table.get("headers", []),
                "rows": cleaned_table.get("rows", []),
                "notes": cleaned_table.get("notes")
            })

        except Exception as error:
            warnings.append(
                f"LLM table cleaning failed on page {page_number}, "
                f"table {table.get('table_id')}: {error}"
            )

            clean_tables.append({
                "table_id": table.get("table_id"),
                "headers": [],
                "rows": get_table_source_rows(table),
                "notes": "Table kept from raw extraction because LLM table cleaning failed."
            })

    return {
        "page_number": page.get("page_number"),
        "page_type": page.get("page_type"),
        "text_structure": page.get("text_structure"),
        "clean_blocks": clean_blocks,
        "clean_tables": clean_tables,
        "warnings": warnings
    }


def clean_pdf_output_with_groq(
    input_path="pdf_ingestion_output.json",
    output_path="llm_cleaned_output.json"
):
    """
    Main function used by Streamlit.

    Keeps old working interface:
    - input: pdf_ingestion_output.json
    - output: llm_cleaned_output.json

    But internally saves cost:
    - paragraphs/headings/footer/unknown copied directly
    - only list/table blocks and detected tables go to LLM
    """

    client = get_groq_client()

    pages = load_json(input_path)
    cleaned_pages = []

    for page in pages:
        page_number = page.get("page_number")
        print(f"Selective cleaning page {page_number}...")

        try:
            cleaned_page = clean_page_selectively(client, page)
            cleaned_pages.append(cleaned_page)

        except Exception as error:
            cleaned_pages.append({
                "page_number": page_number,
                "page_type": page.get("page_type"),
                "text_structure": page.get("text_structure"),
                "clean_blocks": [
                    copy_block_without_llm(block)
                    for block in page.get("text_blocks", [])
                ],
                "clean_tables": [],
                "warnings": [
                    f"Page-level selective cleaning failed: {error}"
                ]
            })

    save_json(cleaned_pages, output_path)

    return cleaned_pages


if __name__ == "__main__":
    clean_pdf_output_with_groq()