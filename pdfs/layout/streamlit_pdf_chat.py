# streamlit_pdf_chat.py

import os
import json
import tempfile
import sys
from pathlib import Path
import re
import hashlib

import pandas as pd
import streamlit as st
from groq import Groq


CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


try:
    from textbased import analyze_pdf, save_results_to_json
    from llm_cleaner_groq import clean_pdf_output_with_groq

except Exception as error:
    st.error("Import failed.")
    st.exception(error)
    st.stop()


MODEL_NAME = "llama-3.3-70b-versatile"

INGESTION_OUTPUT_PATH = "pdf_ingestion_output.json"
LLM_CLEANED_OUTPUT_PATH = "llm_cleaned_output.json"


# ============================================================
# Basic helpers
# ============================================================

def get_groq_client():
    """
    Creates Groq client using environment variable.
    Do not hardcode API keys in code.
    """

    api_key = " "  # Replace with your actual API key or set it as an environment variable

    if not api_key:
        st.error("GROQ_API_KEY is missing. Set it in your terminal first.")
        st.stop()

    return Groq(api_key=api_key)


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_file_hash(file_bytes):
    """
    Creates a stable hash for the uploaded PDF.
    Same PDF = same hash.
    """

    return hashlib.md5(file_bytes).hexdigest()


def truncate_text(text, max_chars=2500):
    """
    Prevents very large blocks/tables from exceeding Groq context length.
    """

    if text is None:
        return ""

    text = str(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[truncated]"


def normalize_pages(data):
    """
    Makes sure loaded JSON is always a list of page dictionaries.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(data.get("pages"), list):
            return data["pages"]

        if isinstance(data.get("data"), list):
            return data["data"]

    return []


def get_blocks(page):
    """
    Supports both final cleaned JSON and raw parser JSON.
    Chat normally uses clean_blocks.
    """

    return page.get("clean_blocks", page.get("text_blocks", []))


def get_tables(page):
    """
    Supports both final cleaned JSON and raw parser JSON.
    Chat normally uses clean_tables.
    """

    return page.get("clean_tables", page.get("tables", []))


def get_or_create_llm_cleaned_output(force_create=False):
    """
    Keeps the old stable two-JSON flow.

    - pdf_ingestion_output.json = raw parser output
    - llm_cleaned_output.json = chat-ready output

    force_create=True is used after a new PDF upload.
    """

    cleaned_output_path = Path(LLM_CLEANED_OUTPUT_PATH)

    if cleaned_output_path.exists() and not force_create:
        return normalize_pages(load_json_file(cleaned_output_path))

    clean_pdf_output_with_groq(
        input_path=INGESTION_OUTPUT_PATH,
        output_path=LLM_CLEANED_OUTPUT_PATH
    )

    return normalize_pages(load_json_file(cleaned_output_path))


# ============================================================
# Context builders
# ============================================================

def build_pdf_context(pages):
    """
    Builds full context from cleaned JSON.
    Kept for compatibility/debugging.
    Chat does not send this full context directly.
    """

    context_parts = []

    for page in pages:
        page_number = page.get("page_number")
        page_type = page.get("page_type")
        text_structure = page.get("text_structure")

        context_parts.append(
            f"\n--- PAGE {page_number} | "
            f"page_type={page_type} | "
            f"text_structure={text_structure} ---"
        )

        for block in get_blocks(page):
            block_id = block.get("block_id")
            role = block.get("role")
            text = block.get("text", "")

            context_parts.append(
                f"[page={page_number} block={block_id} role={role}]\n{text}"
            )

        for table in get_tables(page):
            table_id = table.get("table_id")
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            notes = table.get("notes")

            context_parts.append(
                f"[page={page_number} table={table_id}]\n"
                f"headers={json.dumps(headers, ensure_ascii=False)}\n"
                f"rows={json.dumps(rows, ensure_ascii=False)}\n"
                f"notes={notes}"
            )

        warnings = page.get("warnings", [])

        if warnings:
            context_parts.append(
                f"[page={page_number} warnings]\n"
                f"{json.dumps(warnings, ensure_ascii=False)}"
            )

    return "\n\n".join(context_parts)


def page_to_search_text(page):
    """
    Converts one page into searchable text.
    """

    parts = []

    for block in get_blocks(page):
        parts.append(block.get("text", ""))

    for table in get_tables(page):
        parts.append(json.dumps(table, ensure_ascii=False))

    return " ".join(parts).lower()


def extract_question_page_numbers(question):
    """
    Detects explicit page references from user questions.
    This is an obvious universal rule, not PDF-specific hardcoding.
    """

    question_lower = question.lower()

    patterns = [
        r"\bpage\s+(\d+)\b",
        r"\bpage\s+no\.?\s+(\d+)\b",
        r"\bpage\s+number\s+(\d+)\b",
        r"\bcontents?\s+of\s+page\s+(\d+)\b",
        r"\bwhat\s+is\s+on\s+page\s+(\d+)\b",
        r"\bshow\s+me\s+page\s+(\d+)\b",
        r"\bpg\.?\s*(\d+)\b",
        r"\bp\.?\s*(\d+)\b",
        r"\bpágina\s+(\d+)\b",
        r"\bpagina\s+(\d+)\b"
    ]

    page_numbers = []

    for pattern in patterns:
        matches = re.findall(pattern, question_lower)
        for match in matches:
            page_numbers.append(int(match))

    return sorted(list(set(page_numbers)))


def extract_question_table_number(question):
    """
    Detects explicit document table references.
    Example: table 2, Table 3, tabla 1.
    """

    question_lower = question.lower()

    table_match = re.search(
        r"\btable\s+(\d+)\b|\btabla\s+(\d+)\b",
        question_lower
    )

    if not table_match:
        return None

    for value in table_match.groups():
        if value:
            return int(value)

    return None


def build_document_index(pages, max_preview_blocks=8):
    """
    Builds a compact index of the document for the LLM query planner.

    The planner reads this small index to understand user intent and decide
    which pages are likely relevant. This avoids hardcoding individual phrases.
    """

    document_index = []

    for page in pages:
        page_number = page.get("page_number")
        page_type = page.get("page_type")
        text_structure = page.get("text_structure")

        headings = []
        captions = []
        important_lines = []
        table_previews = []
        page_preview = []

        blocks = get_blocks(page)
        tables = get_tables(page)

        for block in blocks:
            role = str(block.get("role", "")).lower().strip()
            text = block.get("text", "")

            if not text:
                continue

            clean_text = " ".join(str(text).split())
            lower_text = clean_text.lower()

            if role in ["heading", "subheading"]:
                headings.append(clean_text[:300])

            if re.match(r"^(table|tabla)\s+\d+", lower_text):
                captions.append(clean_text[:300])

            if role in ["unknown", "paragraph", "table", "list", "structured_list"]:
                if len(important_lines) < max_preview_blocks:
                    important_lines.append({
                        "role": role,
                        "text": clean_text[:350]
                    })

            if role == "table":
                table_previews.append(clean_text[:350])

            if len(page_preview) < max_preview_blocks:
                page_preview.append(clean_text[:250])

        for table in tables:
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            notes = table.get("notes")

            table_previews.append(
                json.dumps(
                    {
                        "headers": headers,
                        "sample_rows": rows[:3],
                        "notes": notes
                    },
                    ensure_ascii=False
                )[:700]
            )

        document_index.append({
            "page_number": page_number,
            "page_type": page_type,
            "text_structure": text_structure,
            "headings": headings[:6],
            "captions": captions[:6],
            "has_tables": bool(tables or table_previews),
            "table_count": len(tables),
            "table_previews": table_previews[:4],
            "important_lines": important_lines[:8],
            "page_preview": page_preview[:8]
        })

    return document_index


# ============================================================
# Retrieval: deterministic rules + LLM query planner
# ============================================================

def select_relevant_pages_keyword_fallback(pages, question, max_pages=4):
    """
    Fallback retrieval when LLM planner fails.

    This is intentionally generic:
    - explicit page numbers
    - explicit table numbers
    - keyword scoring
    """

    question_lower = question.lower().strip()

    page_numbers = extract_question_page_numbers(question)

    if page_numbers:
        selected = [
            page for page in pages
            if page.get("page_number") in page_numbers
        ]

        if selected:
            return selected

        return []

    table_number = extract_question_table_number(question)

    if table_number:
        table_phrase = f"table {table_number}"
        tabla_phrase = f"tabla {table_number}"

        selected = []

        for page in pages:
            page_text = page_to_search_text(page)

            if table_phrase in page_text or tabla_phrase in page_text:
                selected.append(page)

        if selected:
            return selected[:max_pages]

    question_words = [
        word for word in re.findall(r"\w+", question_lower)
        if len(word) >= 3
    ]

    scored_pages = []

    for page in pages:
        page_text = page_to_search_text(page)
        score = 0

        for word in question_words:
            if word in page_text:
                score += 1

        if "table" in question_lower or "tabla" in question_lower:
            if get_tables(page):
                score += 5

            for block in get_blocks(page):
                if block.get("role") == "table":
                    score += 2

        if score > 0:
            scored_pages.append((score, page))

    scored_pages.sort(key=lambda item: item[0], reverse=True)

    selected_pages = [
        page for score, page in scored_pages[:max_pages]
    ]

    if selected_pages:
        return selected_pages

    return pages[:2]


def plan_user_question(client, question, document_index):
    """
    LLM query planner.

    It does not answer the user.
    It decides what content should be retrieved.
    """

    system_prompt = """
You are a query planner for a PDF chat system.

You do NOT answer the user.
You ONLY decide what PDF content should be retrieved.

You are given a compact document index. The index contains page numbers,
headings, captions, table previews, important lines, and short page previews.

Return valid JSON only.

Possible intents:
- document_overview
- document_structure
- page_lookup
- table_lookup
- section_lookup
- factual_question
- list_request
- comparison
- unknown

Universal interpretation rules:
- If the user asks what the PDF/document/report is about, use document_overview.
- If the user asks for contents, chapters, sections, outline, index, or structure, use document_structure.
- If the user mentions an explicit page number, use page_lookup.
- If the user mentions an explicit table number, use table_lookup.
- If the user asks about a named topic/section, use section_lookup or factual_question.
- If the user asks for all items, main points, key findings, important data, or insights, retrieve pages with summary/introduction/overview/headings and relevant table pages.
- Use the document index to choose likely target_pages.
- Do not invent page numbers.
- If unsure, return useful search_terms from the user's question and the document index.

Return exactly this JSON:

{
  "intent": "",
  "target_pages": [],
  "target_table_number": null,
  "search_terms": [],
  "needs_broad_context": false,
  "reason": ""
}
"""

    user_prompt = f"""
DOCUMENT INDEX:
{json.dumps(document_index, ensure_ascii=False)}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


def retrieve_pages_from_plan(pages, plan, question, max_pages=4):
    """
    Retrieves pages based on the LLM query plan.
    Deterministic rules still override obvious cases.
    """

    explicit_pages = extract_question_page_numbers(question)

    if explicit_pages:
        selected = [
            page for page in pages
            if page.get("page_number") in explicit_pages
        ]

        if selected:
            return selected[:max_pages]

        return []

    explicit_table_number = extract_question_table_number(question)

    if explicit_table_number:
        target_table_number = explicit_table_number
    else:
        target_table_number = plan.get("target_table_number")

    target_pages = plan.get("target_pages", [])

    if target_pages:
        selected = [
            page for page in pages
            if page.get("page_number") in target_pages
        ]

        if selected:
            return selected[:max_pages]

    search_terms = plan.get("search_terms", [])
    intent = plan.get("intent", "")

    scored_pages = []

    for page in pages:
        page_text = page_to_search_text(page)
        score = 0

        for term in search_terms:
            term = str(term).lower().strip()
            if term and term in page_text:
                score += 4

        if target_table_number:
            table_phrase = f"table {target_table_number}"
            tabla_phrase = f"tabla {target_table_number}"

            if table_phrase in page_text or tabla_phrase in page_text:
                score += 12

        # Universal intent-based boosts.
        # These are not PDF-specific; they represent common document structure signals.
        if intent == "document_structure":
            structure_signals = [
                "contents",
                "table of contents",
                "chapter",
                "section",
                "executive summary",
                "introduction"
            ]

            for signal in structure_signals:
                if signal in page_text:
                    score += 3

        if intent == "document_overview":
            overview_signals = [
                "summary",
                "executive summary",
                "introduction",
                "overview",
                "background",
                "key findings",
                "conclusion"
            ]

            for signal in overview_signals:
                if signal in page_text:
                    score += 3

        if intent == "table_lookup":
            if get_tables(page):
                score += 5

            for block in get_blocks(page):
                if block.get("role") == "table":
                    score += 3

        if score > 0:
            scored_pages.append((score, page))

    scored_pages.sort(key=lambda item: item[0], reverse=True)

    selected = [
        page for score, page in scored_pages[:max_pages]
    ]

    if selected:
        return selected

    return select_relevant_pages_keyword_fallback(
        pages=pages,
        question=question,
        max_pages=max_pages
    )


def build_question_context(pages, question, client, max_context_chars=18000):
    """
    Builds context using:
    1. compact document index
    2. LLM query planner
    3. page retrieval from plan
    4. controlled context size
    """

    document_index = build_document_index(pages)

    try:
        plan = plan_user_question(
            client=client,
            question=question,
            document_index=document_index
        )

        selected_pages = retrieve_pages_from_plan(
            pages=pages,
            plan=plan,
            question=question
        )

    except Exception as error:
        plan = {
            "intent": "fallback_keyword_retrieval",
            "target_pages": [],
            "target_table_number": None,
            "search_terms": [],
            "needs_broad_context": False,
            "reason": f"Planner failed, used keyword fallback: {error}"
        }

        selected_pages = select_relevant_pages_keyword_fallback(
            pages=pages,
            question=question
        )

    if not selected_pages:
        available_pages = [
            page.get("page_number")
            for page in pages
            if page.get("page_number") is not None
        ]

        return (
            f"No matching page found. Available pages: {available_pages}\n\n"
            f"QUERY PLAN:\n{json.dumps(plan, ensure_ascii=False)}"
        )

    context_parts = []
    current_length = 0

    plan_text = f"QUERY PLAN:\n{json.dumps(plan, ensure_ascii=False)}"
    context_parts.append(plan_text)
    current_length += len(plan_text)

    for page in selected_pages:
        page_number = page.get("page_number")
        page_type = page.get("page_type")
        text_structure = page.get("text_structure")

        page_header = (
            f"\n--- PAGE {page_number} | "
            f"page_type={page_type} | text_structure={text_structure} ---"
        )

        if current_length + len(page_header) > max_context_chars:
            context_parts.append("\n[Context truncated before adding more pages.]")
            return "\n\n".join(context_parts)

        context_parts.append(page_header)
        current_length += len(page_header)

        for block in get_blocks(page):
            block_id = block.get("block_id")
            role = block.get("role")
            text = truncate_text(block.get("text", ""), max_chars=2500)

            block_text = f"[page={page_number} block={block_id} role={role}]\n{text}"

            if current_length + len(block_text) > max_context_chars:
                context_parts.append("\n[Context truncated because it was too large.]")
                return "\n\n".join(context_parts)

            context_parts.append(block_text)
            current_length += len(block_text)

        for table in get_tables(page):
            table_id = table.get("table_id")
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            notes = table.get("notes")

            limited_rows = rows[:30]

            table_text = (
                f"[page={page_number} table={table_id}]\n"
                f"headers={json.dumps(headers, ensure_ascii=False)}\n"
                f"rows={json.dumps(limited_rows, ensure_ascii=False)}\n"
                f"notes={notes}"
            )

            if len(rows) > 30:
                table_text += f"\n[Only first 30 rows shown out of {len(rows)} rows.]"

            if current_length + len(table_text) > max_context_chars:
                context_parts.append("\n[Context truncated because it was too large.]")
                return "\n\n".join(context_parts)

            context_parts.append(table_text)
            current_length += len(table_text)

    return "\n\n".join(context_parts)


# ============================================================
# Chat answering
# ============================================================

def answer_direct_question(question, pages):
    """
    Handles simple deterministic questions without calling the final answer LLM.
    """

    question_lower = question.lower()

    if (
        "how many pages" in question_lower
        or "number of pages" in question_lower
        or "total pages" in question_lower
    ):
        return f"There are {len(pages)} pages in the PDF."

    return None


def ask_pdf_llm(client, question, pages):
    """
    Answers user question from selected cleaned PDF context.
    Uses LLM query planner before final answering.
    """

    direct_answer = answer_direct_question(question, pages)

    if direct_answer:
        return direct_answer

    pdf_context = build_question_context(
        pages=pages,
        question=question,
        client=client
    )

    system_prompt = """
You are a strict PDF question-answering assistant for an extracted PDF ingestion system.

You must answer ONLY using the provided cleaned PDF context.
You must not use outside knowledge.
You must not invent information.

==================================================
CORE BEHAVIOUR
==================================================

- The provided context is the only source of truth.
- If information is present anywhere in the provided context, use it.
- Do not say information is unavailable if it exists in the provided context.
- Preserve original document wording, names, numbers, units, table labels, and page references.
- Mention the page number when answering page-specific questions or when it helps verification.
- Be concise unless the user asks for detail.
- Use the query plan only to understand what was retrieved. Do not expose the query plan unless the user asks.

==================================================
USER INTENT
==================================================

Understand the user's natural meaning from the question and the retrieved context.

Examples:
- If the user asks what the PDF is about, provide a document-level overview.
- If the user asks for contents/chapters/sections/structure, provide the document structure if present.
- If the user asks for a page, use the page section in the context.
- If the user asks for a table by document number, look for captions such as "Table 1.", "Table 2.", etc.
- If the user asks a factual question, answer directly from the retrieved context.

==================================================
PAGE QUESTIONS
==================================================

If the user asks about a page:
- Use the matching PAGE section in the context.
- Summarize or list content from that page.
- Include headings, paragraphs, tables, captions, footers, and notes if relevant.
- Do not say the page is unavailable unless the context explicitly says no matching page was found.

If the context contains:
"No matching page found. Available pages: [...]"

Then answer:
"The requested page is not available. Available pages are: [...]"

==================================================
TABLE QUESTIONS
==================================================

Important distinction:
- "table_id" is an internal parser ID.
- "Table 1", "Table 2", "Table 3" in the document text are document table numbers from captions.
- Do not confuse internal table_id with document table number.

For table answers:
- If clean_tables contains rows, use those rows.
- If clean_tables is empty or rows are empty, use clean_blocks with role="table" as fallback.
- If table-like rows exist as role="table" blocks, reconstruct the best possible table from those blocks.
- Return a markdown table when possible.
- Do not drop columns.
- Do not drop rows.
- Preserve numbers exactly.
- Preserve symbols such as %, -, *, **, <, >, commas, decimals, and units.
- Do not summarize a table unless the user asks for a summary.
- If a table caption is present, include it before the table.
- If table structure is partially available, say that clearly instead of saying the table does not exist.

==================================================
BLOCK ROLE RULES
==================================================

The context may contain blocks with roles:
- heading
- subheading
- paragraph
- list
- structured_list
- table
- footer
- unknown

Use all roles when relevant.
Do not ignore "unknown" blocks because captions or important labels can be stored as unknown.
Do not ignore "table" blocks because table rows may be stored as clean_blocks instead of clean_tables.
Do not say footer is noise.
Do not change detected roles.

==================================================
ANSWER FORMAT
==================================================

- If the user asks for a table: use markdown table format.
- If the user asks for a list: use bullet points.
- If the user asks for document contents/structure: organize by sections/chapters/headings if available.
- If the user asks for page contents: organize by headings, paragraphs, tables, notes.
- If the user asks a direct factual question: answer directly and mention the source page if useful.
- If the answer is not in the provided context, say:
  "This information is not available in the provided extracted PDF context."

==================================================
FAILURE HANDLING
==================================================

Before saying information is unavailable, check:
1. clean_tables
2. clean_blocks with role="table"
3. clean_blocks with role="unknown" for captions/labels
4. clean_blocks with role="paragraph", "heading", "subheading", "list", or "structured_list"

Only say unavailable if it is absent from all provided context.
"""

    user_prompt = f"""
You are given selected cleaned PDF context below.
This context was selected from extracted JSON using a query-planning step.

Use ONLY this context to answer.

Important:
- PAGE headers show source page numbers.
- Blocks may contain headings, paragraphs, captions, footers, lists, and table text.
- Tables may appear in clean_tables OR as clean_blocks with role="table".
- Captions such as "Table 1.", "Table 2.", etc. may appear as role="unknown" or heading-like text.
- Internal table_id is not the same as the document table number.
- If the requested information is present in any block or table below, use it.
- If the context says no matching page was found, report that directly.

SELECTED PDF CONTEXT
====================

{pdf_context}

====================
USER QUESTION
====================

{question}

====================
ANSWER
====================
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# Automatic overview
# ============================================================

def build_overview_context(pages, max_pages=12, max_context_chars=20000):
    """
    Builds compact context for automatic PDF overview.
    This is for normal users, not backend debugging.
    """

    context_parts = []
    current_length = 0

    selected_pages = pages[:max_pages]

    for page in selected_pages:
        page_number = page.get("page_number")
        page_header = f"\n--- PAGE {page_number} ---"

        if current_length + len(page_header) > max_context_chars:
            break

        context_parts.append(page_header)
        current_length += len(page_header)

        for block in get_blocks(page):
            role = block.get("role")
            text = block.get("text", "")

            if not text:
                continue

            if role in ["heading", "subheading", "paragraph", "unknown", "table", "list", "structured_list"]:
                block_text = f"[{role}] {truncate_text(text, max_chars=900)}"

                if current_length + len(block_text) > max_context_chars:
                    break

                context_parts.append(block_text)
                current_length += len(block_text)

        for table in get_tables(page):
            table_id = table.get("table_id")
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            notes = table.get("notes")

            table_text = (
                f"[table_id={table_id}]\n"
                f"headers={json.dumps(headers, ensure_ascii=False)}\n"
                f"rows={json.dumps(rows[:8], ensure_ascii=False)}\n"
                f"notes={notes}"
            )

            if current_length + len(table_text) > max_context_chars:
                break

            context_parts.append(table_text)
            current_length += len(table_text)

        warnings = page.get("warnings", [])

        if warnings:
            warning_text = f"[warnings] {json.dumps(warnings, ensure_ascii=False)}"

            if current_length + len(warning_text) <= max_context_chars:
                context_parts.append(warning_text)
                current_length += len(warning_text)

    return "\n\n".join(context_parts)


def generate_pdf_overview(client, pages):
    """
    Generates automatic PDF overview after upload/load.
    """

    overview_context = build_overview_context(pages)

    system_prompt = """
You are a PDF overview assistant.

Use ONLY the provided extracted PDF context.
Do not use outside knowledge.
Do not invent missing facts.

Write for a normal reader, not a developer.

Do not focus on backend details like block IDs, parser IDs, or page numbers unless needed.
Do not mention page numbers by default.
Do not mention internal table_id unless the user asks.

Create a useful overview with these sections:

1. Quick summary
2. Main topics covered
3. Key findings
4. Important numbers
5. Important tables
6. Why this document matters
7. Things to pay attention to
8. Data quality note

Rules:
- Keep the language simple and clear.
- Use bullet points where useful.
- For important tables, describe what the table is about, not its internal table_id.
- If table extraction looks incomplete, mention that in normal language.
- If something is not available in the extracted context, say so.
"""

    user_prompt = f"""
Extracted PDF context:
{overview_context}

Generate the automatic PDF overview.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# Display helpers
# ============================================================

def show_cleaned_tables(pages):
    """
    Displays cleaned tables from llm_cleaned_output.json.
    """

    table_found = False

    for page in pages:
        page_number = page.get("page_number")

        for table in get_tables(page):
            table_id = table.get("table_id")
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            notes = table.get("notes")

            if not rows:
                continue

            table_found = True

            st.markdown(f"### Page {page_number} - Table {table_id}")

            try:
                if headers and rows and len(headers) == len(rows[0]):
                    df = pd.DataFrame(rows, columns=headers)
                else:
                    df = pd.DataFrame(rows)

                st.dataframe(df, use_container_width=True)

            except Exception:
                st.write({
                    "headers": headers,
                    "rows": rows,
                    "notes": notes
                })

            if notes:
                st.caption(notes)

    if not table_found:
        st.info("No cleaned tables available.")


def show_cleaned_blocks(pages):
    """
    Displays cleaned blocks page by page.
    """

    for page in pages:
        page_number = page.get("page_number")

        st.markdown(f"### Page {page_number}")

        for block in get_blocks(page):
            role = block.get("role")
            block_id = block.get("block_id")
            text = block.get("text", "")

            st.markdown(f"**Block {block_id} | role={role}**")
            st.text(text)


# ============================================================
# Main Streamlit app
# ============================================================

def main():
    st.set_page_config(
        page_title="PDF Ingestion Chat",
        layout="wide"
    )

    st.title("PDF Ingestion Chat")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if "pages" not in st.session_state:
        st.session_state.pages = None

    if "pdf_context" not in st.session_state:
        st.session_state.pdf_context = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_file_hash" not in st.session_state:
        st.session_state.uploaded_file_hash = None

    if "pdf_overview" not in st.session_state:
        st.session_state.pdf_overview = None

    # Load existing cleaned output if available
    if (
        st.session_state.pages is None
        and Path(LLM_CLEANED_OUTPUT_PATH).exists()
    ):
        cleaned_pages = get_or_create_llm_cleaned_output(force_create=False)

        st.session_state.pages = cleaned_pages
        st.session_state.pdf_context = build_pdf_context(cleaned_pages)

        client = get_groq_client()
        st.session_state.pdf_overview = generate_pdf_overview(client, cleaned_pages)

    # New PDF upload
    if uploaded_pdf is not None:
        file_bytes = uploaded_pdf.getvalue()
        current_file_hash = get_file_hash(file_bytes)

        is_new_file = current_file_hash != st.session_state.uploaded_file_hash

        if is_new_file:
            st.session_state.uploaded_file_hash = current_file_hash

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_bytes)
                temp_pdf_path = temp_file.name

            with st.spinner("Analyzing PDF with parser..."):
                pages = analyze_pdf(temp_pdf_path)
                save_results_to_json(pages, INGESTION_OUTPUT_PATH)

            with st.spinner("Cleaning extracted PDF with LLM..."):
                cleaned_pages = get_or_create_llm_cleaned_output(force_create=True)

                st.session_state.pages = cleaned_pages
                st.session_state.pdf_context = build_pdf_context(cleaned_pages)
                st.session_state.messages = []

                client = get_groq_client()
                st.session_state.pdf_overview = generate_pdf_overview(client, cleaned_pages)

            Path(temp_pdf_path).unlink(missing_ok=True)

            st.success("PDF analyzed and cleaned successfully.")

        else:
            st.info("PDF already processed. Using existing cleaned output.")

    if st.session_state.pages:
        if st.session_state.pdf_overview:
            with st.expander("PDF Overview", expanded=True):
                st.markdown(st.session_state.pdf_overview)

        with st.expander("View cleaned JSON"):
            st.json(st.session_state.pages)

        with st.expander("View cleaned text blocks"):
            show_cleaned_blocks(st.session_state.pages)

        with st.expander("View cleaned tables"):
            show_cleaned_tables(st.session_state.pages)

        st.divider()

        client = get_groq_client()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("Ask anything from the PDF")

        if user_question:
            st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Reading cleaned PDF context..."):
                    answer = ask_pdf_llm(
                        client=client,
                        question=user_question,
                        pages=st.session_state.pages
                    )

                    st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


if __name__ == "__main__":
    main()
