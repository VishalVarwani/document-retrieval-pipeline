# detect_text_structure.py

from collections import Counter
import re


TOP_REGION_RATIO = 0.15
BOTTOM_REGION_RATIO = 0.10


def detect_text_structure(page):
    """
    Detects the main text structure of a PDF page.

    It separates:
    - top region
    - body region
    - bottom region

    Main structure is decided mostly from the body region.
    Full page is still preserved through signals.
    """

    full_features = get_page_features(page)

    top_region, body_region, bottom_region = split_page_regions(page)

    top_features = get_page_features(top_region)
    body_features = get_page_features(body_region)
    bottom_features = get_page_features(bottom_region)

    if body_features["word_count"] == 0 and full_features["word_count"] == 0:
        return {
            "text_structure": "unknown_text_structure",
            "reason": "No words found on page",
            "mixed_types": [],
            "scores": {},
            "region_summary": {},
            "signals": full_features
        }

    # Main decision should come from body, not header/footer
    scores = calculate_structure_scores(body_features)

    strong_matches = {
        name: score for name, score in scores.items()
        if score >= 3
    }

    mixed_types = []

    if len(strong_matches) > 1:
        text_structure = "mixed_text_structure"
        mixed_types = list(strong_matches.keys())
        reason = (
            "Body region contains multiple strong text structures: "
            + ", ".join(mixed_types)
        )
    else:
        text_structure = max(scores, key=scores.get)
        best_score = scores[text_structure]

        if best_score < 2:
            text_structure = "unknown_text_structure"
            reason = "No strong text structure pattern found in body region"
        else:
            reason = f"Strongest body-region pattern: {text_structure}"

    region_summary = {
        "top_region": summarize_region(top_features),
        "body_region": summarize_region(body_features),
        "bottom_region": summarize_region(bottom_features),
    }

    return {
        "text_structure": text_structure,
        "reason": reason,
        "mixed_types": mixed_types,
        "scores": scores,
        "region_summary": region_summary,
        "signals": {
            "full_page": full_features,
            "top_region": top_features,
            "body_region": body_features,
            "bottom_region": bottom_features
        }
    }


def split_page_regions(page):
    """
    Splits page into top, body, and bottom regions.
    We do not delete anything.
    We only separate regions so header/footer do not dominate the main structure.
    """

    page_width = page.width
    page_height = page.height

    top_end = page_height * TOP_REGION_RATIO
    bottom_start = page_height * (1 - BOTTOM_REGION_RATIO)

    top_region = page.crop((0, 0, page_width, top_end))
    body_region = page.crop((0, top_end, page_width, bottom_start))
    bottom_region = page.crop((0, bottom_start, page_width, page_height))

    return top_region, body_region, bottom_region


def get_page_features(page_area):
    """
    Extracts useful layout features from a page or cropped page region.
    """

    text = page_area.extract_text() or ""
    words = page_area.extract_words() or []
    lines = page_area.lines or []
    rects = page_area.rects or []
    tables = page_area.extract_tables() or []

    return {
        "text": text,
        "words": words,
        "word_count": len(words),
        "char_count": len(text.strip()),
        "line_count": len(lines),
        "rect_count": len(rects),
        "table_count": len(tables)
    }


def calculate_structure_scores(features):
    """
    Calculates scores only from the selected region.
    Usually we pass body-region features here.
    """

    text = features["text"]
    words = features["words"]

    paragraph_score = get_paragraph_score(text)
    table_score = get_table_score(
        words=words,
        table_count=features["table_count"],
        line_count=features["line_count"],
        rect_count=features["rect_count"]
    )
    list_score = get_structured_list_score(text)
    form_score = get_form_score(
        text=text,
        rect_count=features["rect_count"]
    )
    if table_score >= 3 and features["table_count"] > 0:
        list_score = 0

    return {
        "paragraph_based": paragraph_score,
        "table_based": table_score,
        "structured_list": list_score,
        "form_like": form_score
    }


def summarize_region(features):
    """
    Gives a simple readable summary of each region.
    """

    scores = calculate_structure_scores(features)

    if features["word_count"] == 0:
        likely_structure = "empty_or_non_text"
    else:
        best_structure = max(scores, key=scores.get)
        best_score = scores[best_structure]

        if best_score == 0:
            likely_structure = "unknown"
        else:
            likely_structure = best_structure

    return {
        "likely_structure": likely_structure,
        "word_count": features["word_count"],
        "char_count": features["char_count"],
        "line_count": features["line_count"],
        "rect_count": features["rect_count"],
        "table_count": features["table_count"],
        "scores": scores
    }


def get_paragraph_score(text):
    """
    Paragraph pages usually have full sentences,
    longer lines, and many words.
    """

    score = 0

    text_words = text.split()
    text_lines = [line.strip() for line in text.split("\n") if line.strip()]

    sentence_count = text.count(".") + text.count("?") + text.count("!")
    long_lines = [line for line in text_lines if len(line.split()) >= 8]

    if sentence_count >= 3:
        score += 1

    if len(long_lines) >= 3:
        score += 1

    if len(text_words) >= 80:
        score += 1

    return score


def get_table_score(words, table_count, line_count, rect_count):
    """
    Table pages usually have detected tables,
    many table lines/rectangles, or several aligned columns.
    """

    score = 0

    if table_count > 0:
        score += 2

    if line_count >= 12:
        score += 1

    if rect_count >= 8:
        score += 1

    column_count = estimate_column_count(words)

    if column_count >= 3:
        score += 1

    return score


def get_structured_list_score(text):
    """
    Detects real structured lists using line-level patterns.
    """

    score = 0

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if len(lines) < 3:
        return score

    numbered_lines = 0
    bullet_lines = 0
    lettered_lines = 0
    short_lines = 0

    for line in lines:
        if re.match(r"^\d+[\.\)]?\s+\S+", line):
            numbered_lines += 1

        if re.match(r"^[\-\*•]\s+\S+", line):
            bullet_lines += 1

        if re.match(r"^[A-Za-z][\.\)]\s+\S+", line):
            lettered_lines += 1

        if len(line.split()) <= 10:
            short_lines += 1

    list_marker_lines = numbered_lines + bullet_lines + lettered_lines
    list_marker_ratio = list_marker_lines / len(lines)
    short_line_ratio = short_lines / len(lines)

    if list_marker_lines >= 3:
        score += 2

    if list_marker_ratio >= 0.5:
        score += 1

    if list_marker_lines >= 3 and short_line_ratio >= 0.6:
        score += 1

    return score


def get_form_score(text, rect_count):
    """
    Form-like pages usually contain many label-value fields.
    """

    score = 0

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    colon_lines = [line for line in lines if ":" in line]

    if len(colon_lines) >= 4:
        score += 2

    if rect_count >= 5:
        score += 1

    short_colon_lines = [
        line for line in colon_lines
        if len(line.split()) <= 8
    ]

    if len(short_colon_lines) >= 3:
        score += 1

    return score


def estimate_column_count(words, tolerance=8):
    """
    Estimates whether words form multiple vertical columns.

    We ignore single left-margin alignment because normal paragraphs
    also start from the same left side.
    """

    x_positions = []

    for word in words:
        x0 = word.get("x0")

        if x0 is not None:
            rounded_x = round(x0 / tolerance) * tolerance
            x_positions.append(rounded_x)

    counts = Counter(x_positions)

    repeated_positions = [
        position for position, count in counts.items()
        if count >= 5
    ]

    if len(repeated_positions) <= 1:
        return 0

    return len(repeated_positions)