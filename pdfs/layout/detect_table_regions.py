# detect_table_regions.py

def detect_table_regions(page):
    """
    Detects table regions on a PDF page using pdfplumber.

    Output:
    - table_id
    - bbox
    - rows
    - columns
    - confidence
    - detection_signals

    This does NOT extract final business data.
    It only detects where tables are.
    """

    tables = page.find_tables() or []

    detected_tables = []

    for index, table in enumerate(tables, start=1):
        bbox = table.bbox
        extracted_table = table.extract() or []

        row_count = len(extracted_table)
        column_count = estimate_column_count_from_table(extracted_table)

        region_lines = count_objects_inside_bbox(page.lines or [], bbox)
        region_rects = count_objects_inside_bbox(page.rects or [], bbox)

        confidence = calculate_table_confidence(
            row_count=row_count,
            column_count=column_count,
            line_count=region_lines,
            rect_count=region_rects
        )

        detected_tables.append({
            "table_id": index,
            "bbox": list(bbox),
            "rows": row_count,
            "columns": column_count,
            "confidence": confidence,
            "detection_signals": {
                "found_by_pdfplumber": True,
                "line_count_inside_region": region_lines,
                "rect_count_inside_region": region_rects
            }
        })

    return detected_tables


def estimate_column_count_from_table(table_data):
    """
    Estimates number of columns from extracted table rows.
    Uses the longest row because some rows may have missing cells.
    """

    if not table_data:
        return 0

    return max(len(row) for row in table_data if row is not None)


def calculate_table_confidence(row_count, column_count, line_count, rect_count):
    """
    Gives confidence based on table strength.

    Strong table:
    - multiple rows
    - multiple columns
    - lines or rectangles support the region
    """

    score = 0

    if row_count >= 2:
        score += 1

    if column_count >= 2:
        score += 1

    if row_count >= 3 and column_count >= 3:
        score += 1

    if line_count >= 4:
        score += 1

    if rect_count >= 2:
        score += 1

    if score >= 4:
        return "high"

    if score >= 2:
        return "medium"

    return "low"


def count_objects_inside_bbox(objects, bbox):
    """
    Counts lines/rectangles that are inside a table bounding box.
    """

    count = 0

    for obj in objects:
        obj_bbox = get_object_bbox(obj)

        if obj_bbox and bbox_contains_object(bbox, obj_bbox):
            count += 1

    return count


def get_object_bbox(obj):
    """
    Converts pdfplumber line/rect object into bbox format.
    """

    x0 = obj.get("x0")
    x1 = obj.get("x1")
    top = obj.get("top")
    bottom = obj.get("bottom")

    if x0 is None or x1 is None or top is None or bottom is None:
        return None

    return (x0, top, x1, bottom)


def bbox_contains_object(parent_bbox, child_bbox):
    """
    Checks whether child bbox is mostly inside parent bbox.
    """

    parent_x0, parent_top, parent_x1, parent_bottom = parent_bbox
    child_x0, child_top, child_x1, child_bottom = child_bbox

    return (
        child_x0 >= parent_x0
        and child_x1 <= parent_x1
        and child_top >= parent_top
        and child_bottom <= parent_bottom
    )


def block_overlaps_table(block, table_bbox, overlap_threshold=0.5):
    """
    Checks if a text block overlaps a table region.

    Later we can use this to mark:
    block["role"] = "table"
    only when the block is actually inside a detected table region.
    """

    block_bbox = (
        block["x0"],
        block["top"],
        block["x1"],
        block["bottom"]
    )

    overlap_area = calculate_overlap_area(block_bbox, table_bbox)
    block_area = calculate_bbox_area(block_bbox)

    if block_area == 0:
        return False

    overlap_ratio = overlap_area / block_area

    return overlap_ratio >= overlap_threshold


def calculate_overlap_area(bbox1, bbox2):
    """
    Calculates overlap area between two bounding boxes.
    """

    x0_1, top_1, x1_1, bottom_1 = bbox1
    x0_2, top_2, x1_2, bottom_2 = bbox2

    overlap_x0 = max(x0_1, x0_2)
    overlap_top = max(top_1, top_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_bottom = min(bottom_1, bottom_2)

    if overlap_x1 <= overlap_x0 or overlap_bottom <= overlap_top:
        return 0

    return (overlap_x1 - overlap_x0) * (overlap_bottom - overlap_top)


def calculate_bbox_area(bbox):
    """
    Calculates bbox area.
    """

    x0, top, x1, bottom = bbox

    width = max(0, x1 - x0)
    height = max(0, bottom - top)

    return width * height

def filter_real_tables(tables, page_text_structure=None, min_table_height=40):
    """
    Keeps only real content tables.

    A table is accepted only when:
    - page structure allows table detection
    - confidence is high
    - rows >= 2
    - columns >= 2
    - table height is meaningful

    This removes fake header/footer table detections.
    """

    if not page_structure_allows_tables(page_text_structure):
        return []

    real_tables = []

    for table in tables:
        bbox = table.get("bbox", [])
        rows = table.get("rows", 0)
        columns = table.get("columns", 0)
        confidence = table.get("confidence")

        if len(bbox) != 4:
            continue

        x0, top, x1, bottom = bbox
        table_height = bottom - top

        is_real_table = (
            confidence == "high"
            and rows >= 2
            and columns >= 2
            and table_height >= min_table_height
        )

        if is_real_table:
            real_tables.append(table)

    return real_tables


def page_structure_allows_tables(page_text_structure):
    """
    Allows table detection only when page-level structure says:
    - table_based
    - mixed_text_structure with table_based inside mixed_types
    """

    if not page_text_structure:
        return False

    structure = page_text_structure.get("text_structure")
    mixed_types = page_text_structure.get("mixed_types", [])

    if structure == "table_based":
        return True

    if structure == "mixed_text_structure" and "table_based" in mixed_types:
        return True

    return False