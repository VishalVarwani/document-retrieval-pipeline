# extract_text_blocks.py

from statistics import median
import re


LINE_TOP_TOLERANCE = 4
MIN_BLOCK_GAP_MULTIPLIER = 1.8


def extract_text_blocks(page, page_text_structure=None):
    """
    Extracts text from a PDF page as visual text blocks.

    Process:
    words -> lines -> blocks -> classify blocks
    """

    words = page.extract_words() or []

    if not words:
        return []

    lines = group_words_into_lines(words)
    blocks = group_lines_into_blocks(lines)

    blocks = classify_text_blocks(
    blocks,
    page.height,
    page_text_structure=page_text_structure,
)
    blocks = mark_internal_visual_continuations(blocks)


    return blocks


def group_words_into_lines(words):
    """
    Groups words that are on the same visual line.
    """

    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    lines = []

    for word in sorted_words:
        added_to_line = False

        for line in lines:
            if abs(word["top"] - line["top"]) <= LINE_TOP_TOLERANCE:
                line["words"].append(word)
                line["top"] = min(line["top"], word["top"])
                line["bottom"] = max(line["bottom"], word["bottom"])
                line["x0"] = min(line["x0"], word["x0"])
                line["x1"] = max(line["x1"], word["x1"])
                added_to_line = True
                break

        if not added_to_line:
            lines.append({
                "words": [word],
                "top": word["top"],
                "bottom": word["bottom"],
                "x0": word["x0"],
                "x1": word["x1"]
            })

    final_lines = []

    for line in lines:
        line_words = sorted(line["words"], key=lambda w: w["x0"])
        text = " ".join(word["text"] for word in line_words)

        final_lines.append({
            "text": text,
            "top": line["top"],
            "bottom": line["bottom"],
            "x0": line["x0"],
            "x1": line["x1"],
            "word_count": len(line_words)
        })

    return sorted(final_lines, key=lambda l: l["top"])


def group_lines_into_blocks(lines):
    """
    Groups nearby lines into text blocks.
    A large vertical gap starts a new block.
    """

    if not lines:
        return []

    line_gaps = []

    for i in range(1, len(lines)):
        gap = lines[i]["top"] - lines[i - 1]["bottom"]

        if gap > 0:
            line_gaps.append(gap)

    normal_gap = median(line_gaps) if line_gaps else 10
    block_gap_threshold = normal_gap * MIN_BLOCK_GAP_MULTIPLIER

    blocks = []
    current_block_lines = [lines[0]]

    for i in range(1, len(lines)):
        previous_line = lines[i - 1]
        current_line = lines[i]

        gap = current_line["top"] - previous_line["bottom"]

        if gap > block_gap_threshold:
            blocks.append(create_block(current_block_lines, len(blocks) + 1))
            current_block_lines = [current_line]
        else:
            current_block_lines.append(current_line)

    blocks.append(create_block(current_block_lines, len(blocks) + 1))

    return blocks


def create_block(lines, block_id):
    """
    Creates one text block from grouped lines.
    """

    text = "\n".join(line["text"] for line in lines)

    return {
        "block_id": block_id,
        "type": "text_block",
        "role": "unknown",
        "is_continuation": False,
        "continues_from_previous_page": False,
        "continues_after_visual": False,
        "text": text,
        "top": min(line["top"] for line in lines),
        "bottom": max(line["bottom"] for line in lines),
        "x0": min(line["x0"] for line in lines),
        "x1": max(line["x1"] for line in lines),
        "line_count": len(lines),
        "word_count": sum(line["word_count"] for line in lines)
    }
def mark_internal_visual_continuations(blocks):
    """
    Marks paragraph blocks that continue after a large visual gap
    inside the same page.
    """

    previous_paragraph = None

    for block in blocks:
        if block["role"] != "paragraph":
            continue

        if previous_paragraph is not None:
            vertical_gap = block["top"] - previous_paragraph["bottom"]

            rule_1 = previous_paragraph_ended_incomplete(
                previous_paragraph["text"]
            )

            rule_2 = starts_with_lowercase_word(
                block["text"]
            )

            large_gap = vertical_gap >= 80

            if large_gap and (rule_1 or rule_2):
                block["continues_after_visual"] = True

        previous_paragraph = block

    return blocks

def classify_text_blocks(blocks, page_height, page_text_structure=None):
    """
    Classifies each text block using universal roles:

    1. footer
    2. table
    3. list
    4. heading
    5. paragraph
    6. unknown
    """

    for index, block in enumerate(blocks):
        text = block["text"].strip()
        sentence_count = count_sentence_endings(text)

        next_block = blocks[index + 1] if index + 1 < len(blocks) else None

        if is_footer_block(block, page_height):
            block["role"] = "footer"

        # elif is_table_block(block):
        #     block["role"] = "table"

        elif is_list_block(block):
            block["role"] = "list"

        elif page_text_structure == "structured_list":
            block["role"] = "list"

        elif is_subheading_block(block):
            block["role"] = "subheading"

        elif is_heading_block(block, next_block):
            block["role"] = "heading"

        elif is_paragraph_block(block, sentence_count):
            block["role"] = "paragraph"

        else:
            block["role"] = "unknown"

    return blocks


def is_footer_block(block, page_height):
    """
    Footer is usually near the bottom and often contains page number,
    company info, version, date, revision, or contact text.
    """

    text = block["text"].strip()

    near_bottom = block["top"] >= page_height * 0.85

    footer_keywords = [
        "page", "página", "pagina",
        "revision", "revisión",
        "tel", "calle", "stand",
        "date", "fecha"
    ]

    has_footer_keyword = any(
        keyword in text.lower()
        for keyword in footer_keywords
    )

    standalone_page_number = (
        text.isdigit()
        and block["word_count"] == 1
        and block["line_count"] == 1
    )

    short_footer = block["word_count"] <= 25 and block["line_count"] <= 5

    return near_bottom and (
        standalone_page_number
        or has_footer_keyword
        or short_footer
    )

def is_table_block(block):
    """
    Detects table-like text blocks.

    Universal idea:
    Tables often have multiple lines where each line has many separated values.
    """

    lines = [line.strip() for line in block["text"].split("\n") if line.strip()]

    if len(lines) < 3:
        return False

    multi_value_lines = 0

    for line in lines:
        parts = line.split()

        if len(parts) >= 4:
            multi_value_lines += 1

    ratio = multi_value_lines / len(lines)

    return ratio >= 0.7 and block["line_count"] >= 3


def is_list_block(block):
    """
    Detects list-like text blocks.

    Universal examples:
    1. Introduction ........ 6
    1.1. Objective ........ 8
    Tabla 1. Something .... 12
    Anexo 1. Something
    - item
    • item
    """

    lines = [line.strip() for line in block["text"].split("\n") if line.strip()]

    if len(lines) < 2:
        return False

    list_score = 0
    list_like_lines = 0
    dotted_leader_lines = 0
    page_number_end_lines = 0

    for line in lines:
        starts_like_numbered = re.match(r"^\d+(\.\d+)*\.?\s+", line)
        starts_like_named_list = re.match(
            r"^(Tabla|Table|Anexo|Annex|Figure|Figura)\s+\d+",
            line,
            re.IGNORECASE
        )
        starts_like_bullet = re.match(r"^[\-\*•]\s+", line)

        has_dotted_leader = "....." in line
        ends_with_number = re.search(r"\s+\d+$", line)

        if starts_like_numbered or starts_like_named_list or starts_like_bullet:
            list_like_lines += 1

        if has_dotted_leader:
            dotted_leader_lines += 1

        if ends_with_number:
            page_number_end_lines += 1

    if list_like_lines >= 3:
        list_score += 2

    if dotted_leader_lines >= 3:
        list_score += 2

    if page_number_end_lines >= 3:
        list_score += 1

    return list_score >= 3
def is_subheading_block(block):
    """
    Detects numbered subheadings.

    Examples:
    3.1. Metodología de Muestreo
    3.1.1. Medición de caudales:
    5.2.3. Índice de calidad del agua
    """

    text = block["text"].strip()

    if block["line_count"] > 2:
        return False

    if block["word_count"] > 15:
        return False

    return bool(re.match(r"^\d+\.\d+(\.\d+)*\.?\s+", text))

def is_heading_block(block, next_block):
    """
    Heading is usually short and can be:
    - main numbered heading: 1. OBJETIVOS
    - uppercase title: INTRODUCCIÓN
    - short block followed by larger paragraph
    """

    text = block["text"].strip()

    short_text = block["word_count"] <= 10
    few_lines = block["line_count"] <= 2

    next_is_larger = (
        next_block is not None and
        next_block["word_count"] >= 20
    )

    main_numbered_heading = bool(
        re.match(r"^\d+\.\s+", text)
    )

    uppercase_heading = (
        text.isupper()
        and block["word_count"] <= 10
    )

    return few_lines and short_text and (
        main_numbered_heading
        or uppercase_heading
        or next_is_larger
    )


def is_paragraph_block(block, sentence_count):
    """
    Paragraph usually has many words, multiple lines, or sentence endings.
    """

    many_words = block["word_count"] >= 30
    multiple_lines = block["line_count"] >= 3
    has_sentences = sentence_count >= 2

    return many_words or multiple_lines or has_sentences


def count_sentence_endings(text):
    """
    Counts simple sentence-ending punctuation.
    """

    return text.count(".") + text.count("?") + text.count("!")

def get_main_blocks(blocks):
    """
    Returns meaningful content blocks only.
    Ignores footer and standalone page numbers.
    """

    return [
        block for block in blocks
        if block["role"] != "footer"
        and not is_page_number_block(block)
    ]
def mark_list_continuations(page_results):
    """
    Marks list blocks that continue from the previous page.

    Logic:
    - previous page last main block must be list
    - current page starts/continues with list-like blocks
    - keep marking list-like blocks until the pattern stops
    """

    previous_last_main_block = None

    for page_result in page_results:
        blocks = page_result.get("text_blocks", [])
        main_blocks = get_main_blocks(blocks)

        if main_blocks and previous_last_main_block is not None:
            previous_was_list = previous_last_main_block["role"] == "list"

            if previous_was_list:
                list_continuation_started = False

                for block in main_blocks:
                    if looks_like_list_continuation_block(block):
                        block["role"] = "list"
                        block["continues_from_previous_page"] = True
                        list_continuation_started = True

                    else:
                        if list_continuation_started:
                            break

        if main_blocks:
            previous_last_main_block = main_blocks[-1]

    return page_results
def looks_like_list_continuation_block(block):
    """
    Detects whether a block looks like a continued list item.

    This is stricter than normal subheading detection.
    It checks for TOC/list-style patterns.
    """

    text = block["text"].strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        return False

    list_like_lines = 0
    dotted_leader_lines = 0
    page_number_end_lines = 0

    for line in lines:
        starts_like_numbered = re.match(r"^\d+(\.\d+)*\.?\s+", line)

        starts_like_named_list = re.match(
            r"^(Tabla|Table|Anexo|Annex|Figure|Figura)\s+\d+",
            line,
            re.IGNORECASE
        )

        has_dotted_leader = "....." in line
        ends_with_number = re.search(r"\s+\d+$", line)

        if starts_like_numbered or starts_like_named_list:
            list_like_lines += 1

        if has_dotted_leader:
            dotted_leader_lines += 1

        if ends_with_number:
            page_number_end_lines += 1

    # Case 1: one-line continued TOC entry
    if (
        len(lines) == 1
        and list_like_lines == 1
        and (dotted_leader_lines == 1 or page_number_end_lines == 1)
    ):
        return True

    # Case 2: multi-line continued list block
    if list_like_lines >= 2:
        return True

    if dotted_leader_lines >= 2 and page_number_end_lines >= 2:
        return True

    return False

    return page_results
def is_page_number_block(block):
    """
    Detects standalone page-number blocks.
    Example: 1, 2, 3, 4
    """

    text = block["text"].strip()

    return (
        text.isdigit()
        and block["word_count"] == 1
        and block["line_count"] == 1
    )
def mark_continuations(page_results):
    """
    Marks first paragraph block on a page as continuation using 2 rules:

    Rule 1:
    Previous page's last paragraph ends incomplete after removing citations/numbers.

    Rule 2:
    Current page's first paragraph starts with a lowercase word.
    """

    previous_last_main_block = None

    for page_result in page_results:
        blocks = page_result.get("text_blocks", [])

        main_blocks = [
    block for block in blocks
    if block["role"] != "footer"
    and not is_page_number_block(block)
]


        if main_blocks:
            first_main_block = main_blocks[0]

            if (
                previous_last_main_block is not None
                and previous_last_main_block["role"] == "paragraph"
                and first_main_block["role"] == "paragraph"
            ):
                rule_1 = previous_paragraph_ended_incomplete(
                    previous_last_main_block["text"]
                )

                rule_2 = starts_with_lowercase_word(
                    first_main_block["text"]
                )

                if rule_1 or rule_2:
                    first_main_block["is_continuation"] = True

            previous_last_main_block = main_blocks[-1]
    page_results = mark_list_continuations(page_results)

    return page_results
def previous_paragraph_ended_incomplete(text):
    """
    Returns True if previous paragraph looks incomplete.

    Before checking punctuation, remove:
    - [12]
    - (12)
    - standalone number
    - [Smith, 2020]
    - (Smith et al., 2020)
    """

    cleaned_text = clean_trailing_citations_and_numbers(text)

    if not cleaned_text:
        return False

    return not cleaned_text.endswith((".", "?", "!", ":"))


def clean_trailing_citations_and_numbers(text):
    """
    Removes citation-like endings and trailing standalone numbers.
    """

    cleaned = text.strip()

    patterns = [
        r"\s*\[[^\]]+\]\s*$",          # [12], [Smith, 2020]
        r"\s*\([^\)]+\)\s*$",          # (12), (Smith et al., 2020)
        r"\s+Page\s+\d+\s*$",          # Page 12
        r"\s+\d+\s*$"                  # standalone trailing number
    ]

    changed = True

    while changed:
        changed = False

        for pattern in patterns:
            new_cleaned = re.sub(pattern, "", cleaned).strip()

            if new_cleaned != cleaned:
                cleaned = new_cleaned
                changed = True

    return cleaned


def starts_with_lowercase_word(text):
    """
    Returns True if first real word starts with lowercase.
    """

    cleaned = text.strip()

    if not cleaned:
        return False

    match = re.search(r"[A-Za-zÄÖÜäöüß]", cleaned)

    if not match:
        return False

    first_letter = match.group(0)

    return first_letter.islower()