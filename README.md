# Document Retrieval Pipeline

A Python-based document ingestion pipeline for converting PDF files into structured JSON before retrieval, search, or chat-based document analysis.

The project focuses on understanding the internal structure of PDF documents instead of treating each page as plain raw text. It detects page type, text layout, text blocks, and table regions so the final output can be used more reliably in downstream applications such as document search, PDF chat, or RAG pipelines.

## Purpose

PDF documents often contain mixed content such as paragraphs, headings, tables, lists, forms, scanned pages, and low-quality layouts. A simple text extraction step is usually not enough because it can merge unrelated sections, break tables, or lose page structure.

This project solves that problem by creating a structured ingestion layer that separates document content into meaningful parts before any retrieval or LLM-based processing is applied.

## Key Features

* Detects whether a page is text-based, image-based, mixed, low-text, or blank.
* Identifies page structure such as paragraphs, lists, tables, forms, or mixed layouts.
* Extracts text into clean blocks instead of one long unstructured string.
* Classifies blocks as headings, paragraphs, lists, footers, or unknown sections.
* Detects real table regions separately from normal text.
* Prevents paragraphs and lists from being wrongly treated as tables.
* Tracks continuation between blocks and pages.
* Generates structured JSON output for downstream document retrieval or analysis.

## Project Structure

```text
INGESTION/
├── deepseval/
├── pdfs_layout/
├── detect_table_regions.py
├── detect_text_structure.py
├── extract_text_blocks.py
├── llm_cleaned_org.py
├── streamlit_pdf_chat.py
├── pdf_ingestion_output.json
├── llm_cleaned_output.json
├── .gitignore
└── README.md
```

## Main Components

### `detect_text_structure.py`

Detects the layout structure of each page.

It checks whether the page is mainly paragraph-based, list-based, table-based, form-like, mixed, or unknown. This helps the pipeline decide how the page should be processed before extraction.

### `extract_text_blocks.py`

Extracts meaningful text blocks from PDF pages.

It groups words into lines, groups lines into blocks, and then classifies each block based on its role in the document. This makes the extracted content easier to search, clean, and understand.

### `detect_table_regions.py`

Detects real table areas inside PDF pages.

This file helps separate tables from normal paragraphs or lists. It reduces false table detection and improves the quality of the final structured output.

### `llm_cleaned_org.py`

Cleans and organizes the extracted ingestion output.

It prepares the structured JSON in a cleaner format so it can be used more easily for retrieval, document understanding, or later LLM-based workflows.

### `streamlit_pdf_chat.py`

Provides a Streamlit interface for interacting with processed PDF content.

This can be used to test document retrieval, inspect extracted content, and build a simple PDF chat interface on top of the ingestion output.

## Output Files

### `pdf_ingestion_output.json`

Raw structured output generated from the ingestion pipeline.

This file contains extracted page-level information such as page type, text structure, text blocks, table regions, and layout metadata.

### `llm_cleaned_output.json`

Cleaned and organized version of the ingestion output.

This file is prepared for easier use in downstream retrieval or document understanding workflows.

## Why PDF Files Are Not Included

PDF files are ignored in this repository because they are input/test files. They can be large, private, copyrighted, or project-specific.

The repository only stores the pipeline code and structured output examples. Original PDF files should remain local and should not be committed to GitHub.

PDF files are ignored using the `.gitignore` rule:

```gitignore
*.pdf
```

## Installation

Install the required Python packages:

```bash
pip install pdfplumber pandas streamlit
```

Depending on your local setup, you may also need additional packages used by your scripts.

## How to Run

Run the ingestion scripts depending on the step you want to test:

```bash
python detect_text_structure.py
```

```bash
python extract_text_blocks.py
```

```bash
python detect_table_regions.py
```

```bash
python llm_cleaned_org.py
```

To run the Streamlit interface:

```bash
streamlit run streamlit_pdf_chat.py
```

## Example Workflow

1. Add a PDF file locally.
2. Run the ingestion scripts.
3. Detect the page type and page structure.
4. Extract clean text blocks.
5. Detect table regions separately.
6. Save the structured result into JSON.
7. Use the JSON output for retrieval, search, or PDF chat.

## Current Focus

The current focus of this project is reliable PDF ingestion.

Before using LLMs or retrieval systems, the document must first be understood correctly. This project focuses on building that foundation by extracting clean structure, separating tables from text, and preparing high-quality JSON output.

## Repository Status

Initial version of the document ingestion and retrieval pipeline.
