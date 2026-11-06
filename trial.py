"""Build citable source documents from the Komatsu product guide PDF.

Most spec tables in the PDF are rotated 90 degrees, so plain text extraction
returns them reversed. This script rebuilds each table from character positions
and writes one document per machine model, plus one per remaining page.

    pip install pdfplumber
    python trial.py                 # build source_documents.json
    python trial.py "PC220 weight"  # search it; results cite the PDF page
"""
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

import pdfplumber

HERE = Path(__file__).resolve().parent
PDF_PATH = HERE / "komatsu-southern-africa-product-line-ups-booklet-2025.pdf"
OUTPUT_PATH = HERE / "source_documents.json"

LINE_TOLERANCE = 2.5  # characters this close across the line belong to one line
SEGMENT_GAP = 5.0  # a wider gap along the line starts a new cell
WORD_GAP = 1.0  # a wider gap along the line is a space
SPAN_TOLERANCE = 3.0  # a cell this close to midway between two columns spans both
MAX_LABEL_LENGTH = 60  # longer text in the label column is a footnote


class Cell(NamedTuple):
    pos: float  # centre along the reading direction
    text: str
    bold: bool


def is_rotated(char):
    return abs(char["matrix"][1]) > abs(char["matrix"][0])


def is_footnote(char):
    """Footnotes are grey (CMYK 0,0,0,0.7); table text is solid black, titles white."""
    color = char.get("non_stroking_color")
    return bool(color) and len(color) == 4 and not any(color[:3]) and 0 < color[3] < 1


def read_lines(chars, rotated):
    """Group chars into lines, each a list of Cells in reading order."""
    if rotated:
        # Rotated text reads bottom-to-top, with lines running left to right.
        cross = lambda c: (c["x0"] + c["x1"]) / 2
        start, end = (lambda c: -c["bottom"]), (lambda c: -c["top"])
    else:
        cross = lambda c: (c["top"] + c["bottom"]) / 2
        start, end = (lambda c: c["x0"]), (lambda c: c["x1"])

    groups = []
    for char in sorted(chars, key=cross):
        if groups and cross(char) - cross(groups[-1][-1]) <= LINE_TOLERANCE:
            groups[-1].append(char)
        else:
            groups.append([char])

    lines = []
    for group in groups:
        segments = []
        for char in sorted(group, key=start):
            if segments and start(char) - end(segments[-1][-1]) <= SEGMENT_GAP:
                segments[-1].append(char)
            else:
                segments.append([char])

        cells = []
        for segment in segments:
            text = ""
            for previous, char in zip([None] + segment, segment):
                if previous and start(char) - end(previous) > WORD_GAP:
                    text += " "
                text += char["text"]
            text = text.strip()
            if text:
                pos = (start(segment[0]) + end(segment[-1])) / 2
                bold = any("Bold" in char["fontname"] for char in segment)
                cells.append(Cell(pos, text, bold))
        if cells:
            lines.append(cells)
    return lines


def parse_spec_table(lines):
    """Map a rotated table's cells to model columns using the 'Model' header row.

    Returns (title, {model: {spec: value}}, notes), or None if there is no header.
    """
    header_index = next(
        (i for i, cells in enumerate(lines)
         if cells[0].text.lower().startswith("model") and len(cells) > 1),
        None,
    )
    if header_index is None:
        return None

    header_label, *header_cells = lines[header_index]
    columns = [(cell.pos, re.sub(r"\s+-", "-", cell.text)) for cell in header_cells]
    half_width = (columns[1][0] - columns[0][0]) / 2 if len(columns) > 1 else 40.0
    first_column_edge = columns[0][0] - half_width

    title = " ".join(cell.text for cells in lines[:header_index] for cell in cells)
    rows = lines[header_index + 1:]

    # Repeated names mean variants, named on the next row (e.g. Std/High lift).
    names = [name for _, name in columns]
    if len(set(names)) < len(names):
        variants = rows.pop(0)
        columns = [
            (pos, f"{name} ({variant.text})")
            for (pos, name), variant in zip(columns, variants)
        ]

    def target_models(pos):
        """Return the column at pos, or both columns when a merged cell spans them."""
        ranked = sorted(columns, key=lambda column: abs(column[0] - pos))
        spans_two = (
            len(ranked) > 1
            and abs(abs(ranked[0][0] - pos) - abs(ranked[1][0] - pos)) <= SPAN_TOLERANCE
        )
        return [column[1] for column in ranked[:2 if spans_two else 1]]

    def split_row(cells):
        labels = [cell for cell in cells if cell.pos < first_column_edge]
        values = [cell for cell in cells if cell.pos >= first_column_edge]
        return labels, values

    last_value_row = max(i for i, cells in enumerate(rows) if split_row(cells)[1])

    models = {name: {} for _, name in columns}
    notes = [header_label.text] if header_label.text != "Model" else []
    section = ""  # bold heading that applies to the rows below it
    pending = ""  # plain label-only row that qualifies only the next row
    label = ""
    base_label = ""  # last full label, which "(...)" and "@..." rows qualify

    def extend_label(extra):
        nonlocal label
        extended = f"{label} {extra}"
        for specs in models.values():
            if label in specs:
                specs[extended] = specs.pop(label)
        label = extended

    for index, cells in enumerate(rows):
        labels, values = split_row(cells)
        is_bold = any(cell.bold for cell in labels)
        # Three label cells are group, name and unit (e.g. Hoist motor, Rating, kW).
        if len(labels) >= 3:
            section = labels[0].text
            labels = labels[1:]
            is_bold = False
        label_text = " ".join(cell.text for cell in labels)

        if (
            index > last_value_row
            or label_text.startswith("*")
            or label_text.lower().startswith("note")
            or len(label_text) > MAX_LABEL_LENGTH
        ):
            notes.append(" ".join(cell.text for cell in cells))
            continue

        if not values:
            closes_previous = label_text.count(")") > label_text.count("(")
            if label_text.startswith("(") or closes_previous:
                extend_label(label_text)  # the previous label wrapped onto this row
            elif is_bold:
                if pending:
                    extend_label(pending)
                    pending = ""
                section = label_text
            else:
                pending = f"{pending} {label_text}".strip()
            continue

        # "or" separates alternative engines; it is not a spec of its own.
        wraps_previous_row = not label_text or label_text == "or"
        if not wraps_previous_row:
            if is_bold:
                section = ""
            if label_text[0] in "(@":
                label = f"{base_label} {label_text}"
            else:
                label = " / ".join(part for part in (section, pending, label_text) if part)
                base_label = label
            pending = ""

        for value in values:
            for model in target_models(value.pos):
                specs = models[model]
                if label in specs:
                    separator = " " if wraps_previous_row else " | "
                    specs[label] += separator + value.text
                else:
                    specs[label] = value.text

    return title, models, notes


def load_source_documents(pdf_path=PDF_PATH):
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            rotated_chars = [c for c in page.chars if is_rotated(c)]
            rotated = read_lines([c for c in rotated_chars if not is_footnote(c)], rotated=True)
            footnotes = read_lines([c for c in rotated_chars if is_footnote(c)], rotated=True)
            upright = read_lines([c for c in page.chars if not is_rotated(c)], rotated=False)
            base = {"source": pdf_path.name, "page": number}

            table = parse_spec_table(rotated)
            if table is None:
                page_lines = [
                    " | ".join(cell.text for cell in cells)
                    for cells in rotated + footnotes + upright
                ]
                page_lines = [line for line in page_lines if not re.fullmatch(r"Page\s*\d+", line)]
                if page_lines:
                    documents.append(
                        {**base, "id": f"p{number}", "type": "page", "text": "\n".join(page_lines)}
                    )
                continue

            title, models, notes = table
            notes += [" ".join(cell.text for cell in cells) for cells in footnotes]
            for index, (model, specs) in enumerate(models.items(), start=1):
                lines = [f"{title} - {model}"]
                lines += [f"{spec.rstrip(':')}: {value}" for spec, value in specs.items()]
                lines += notes
                documents.append({
                    **base,
                    "id": f"p{number}-{index}",
                    "type": "model_spec",
                    "category": title,
                    "model": model,
                    "specs": specs,
                    "text": "\n".join(lines),
                })
    return documents


def search(documents, query, limit=5):
    terms = query.lower().split()
    scored = []
    for document in documents:
        text = document["text"].lower()
        score = sum(text.count(term) for term in terms)
        matched = sum(term in text for term in terms)
        if score:
            scored.append((matched, score, document))
    scored.sort(key=lambda item: item[:2], reverse=True)
    return [document for _, _, document in scored[:limit]]


def main():
    documents = load_source_documents()
    OUTPUT_PATH.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    models = sum(document["type"] == "model_spec" for document in documents)
    print(f"{len(documents)} documents ({models} model specs) -> {OUTPUT_PATH.name}")

    query = " ".join(sys.argv[1:])
    if query:
        for document in search(documents, query):
            print(f"\n[{document['source']}, page {document['page']}]\n{document['text']}")


if __name__ == "__main__":
    main()
