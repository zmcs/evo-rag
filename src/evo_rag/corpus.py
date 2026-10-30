"""Split the EU AI Act PDF into one document per article/annex.

Input:  data/raw/ai_act.pdf          (the official PDF, Regulation (EU) 2024/1689)
Output: data/processed/corpus.jsonl  (one JSON record per article/annex)
"""

import json
import re
from pathlib import Path

import pymupdf

PDF_PATH = Path("data/raw/ai_act.pdf")
OUT_PATH = Path("data/processed/corpus.jsonl")

ARTICLE_RE = re.compile(r"^Article\s+(\d+)$")
ANNEX_RE = re.compile(r"^ANNEX\s+([IVXLC]+)$")

# Page headers/footers repeated on every page of the Official Journal PDF.
PAGE_NOISE_RE = re.compile(
    r"^(EN|\d{4}/\d{4}|OJ L, \d{1,2}\.\d{1,2}\.\d{4}|ELI: http\S+|\d+/\d+)$"
)
# Structural headings between articles ("CHAPTER III", "SECTION 2"); not article text.
CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLC]+$")
SECTION_RE = re.compile(r"^SECTION\s+\d+$")
# Footnotes citing other legal acts, e.g. '(57)' followed by 'Directive (EU) 2016/943 ...'
FOOTNOTE_NUM_RE = re.compile(r"^\(\d+\)$")
FOOTNOTE_TEXT_RE = re.compile(r"^(Regulation|Directive|Commission|Council|Decision|OJ)\b")


SUPERSCRIPT_FLAG = 1  # PyMuPDF marks superscript text spans with this bit


def span_line_text(spans: list[dict]) -> str:
    """Join a line's text spans, writing exponents as '^': '10' + superscript '25' -> '10^25'."""
    text = ""
    for span in spans:
        piece = span["text"]
        is_superscript = span["flags"] & SUPERSCRIPT_FLAG
        if is_superscript and piece.strip() and text[-1:].isdigit():
            piece = "^" + piece.strip()
        text += piece
    return text


def pdf_to_lines(path: Path = PDF_PATH) -> list[str]:
    """Extract non-empty text lines from the PDF, dropping page headers/footers."""
    lines = []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            for block in page.get_text("dict")["blocks"]:
                for raw_line in block.get("lines", []):
                    line = span_line_text(raw_line["spans"]).replace("\xa0", " ").strip()
                    if line and not PAGE_NOISE_RE.match(line):
                        lines.append(line)
    return lines


def drop_structural_headings(lines: list[str]) -> list[str]:
    """Remove 'CHAPTER X' / 'SECTION N' lines and the title line(s) that follow them."""
    out, i = [], 0
    while i < len(lines):
        if CHAPTER_RE.match(lines[i]):
            i += 1
            # Chapter titles are in capitals and may wrap over several lines.
            while (
                i < len(lines)
                and lines[i].isupper()
                and not ANNEX_RE.match(lines[i])
                and not SECTION_RE.match(lines[i])
            ):
                i += 1
            continue
        if SECTION_RE.match(lines[i]):
            i += 2  # 'SECTION 2' + its one-line title
            continue
        if (
            FOOTNOTE_NUM_RE.match(lines[i])
            and i + 1 < len(lines)
            and FOOTNOTE_TEXT_RE.match(lines[i + 1])
        ):
            # Footnote: '(57)' then a citation that may wrap; it ends with '.'
            i += 1
            while i < len(lines) and not lines[i].endswith("."):
                i += 1
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def split_into_documents(lines: list[str]) -> list[dict]:
    """Start a new document at each article/annex heading."""
    documents, current, seen_ids = [], None, set()

    for line in lines:
        article = ARTICLE_RE.match(line)
        annex = ANNEX_RE.match(line)
        new_id = None
        if article:
            new_id, doc_type = f"art_{article.group(1)}", "article"
        elif annex:
            new_id, doc_type = f"annex_{annex.group(1)}", "annex"

        # A heading we've already seen is a cross-reference that wrapped onto its own line.
        if new_id and new_id not in seen_ids:
            if current:
                documents.append(current)
            seen_ids.add(new_id)
            current = {"doc_id": new_id, "type": doc_type, "title": None, "lines": []}
            continue

        if current is None:  # skip the preamble and recitals before Article 1
            continue
        if line.startswith("Done at Brussels"):  # signature block after the last article
            current["closed"] = True
        if current.get("closed"):
            continue
        if current["title"] is None:
            current["title"] = line.rstrip("`")  # the PDF has a stray backtick in one title
        elif not current["lines"] and line[0].islower():
            current["title"] += " " + line  # a long title wrapped onto a second line
        else:
            current["lines"].append(line)

    if current:
        documents.append(current)

    return [
        {"doc_id": d["doc_id"], "type": d["type"], "title": d["title"], "text": "\n".join(d["lines"])}
        for d in documents
    ]


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(f"Put the AI Act PDF at {PDF_PATH} first.")

    lines = drop_structural_headings(pdf_to_lines())
    docs = split_into_documents(lines)
    if not docs:
        raise SystemExit(f"No articles found in {PDF_PATH}.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    n_articles = sum(d["type"] == "article" for d in docs)
    n_annexes = sum(d["type"] == "annex" for d in docs)
    print(f"Saved {len(docs)} documents ({n_articles} articles, {n_annexes} annexes) to {OUT_PATH}")
    print("First IDs:", [d["doc_id"] for d in docs[:8]])


if __name__ == "__main__":
    main()
