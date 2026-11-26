"""Build the static demo page (docs/index.html) from saved test results. No API calls.

The page shows every held-out test question with the gold answer and the answers of the three
frozen pipelines, their verdicts, the articles they retrieved, and the planner's sub-queries.
A timeline at the bottom explains the four stages of the project; pipeline names show how each one works.
GitHub Pages can serve it straight from the docs/ folder.

  uv run python -m evo_rag.build_demo
"""

import glob
import json
import os
from pathlib import Path

RESULTS_DIR = Path("data/results")
QA_PATH = Path("data/processed/qa.jsonl")
OUT_PATH = Path("docs/index.html")
LABELS = {"baseline": "Baseline", "cheap": "Low-cost tuned", "best": "Best: evolved"}
DESCRIPTIONS = {
    "baseline": "Searches by meaning and reads 5 short passages",
    "cheap": "Same cost as the baseline, with a search model trained on the AI Act",
    "best": "Plans several searches, mixes meaning and keyword search, reads 13 long passages",
}
# Shown when hovering (or tapping) a pipeline name.
DETAILS = {
    "baseline": {
        "origin": "Picked by hand as a sensible starting point.",
        "points": [
            "The Act is cut into passages of about 800 characters, each labelled with its article.",
            "Search by meaning: a small model (bge-small) turns text into numbers that capture meaning, "
            "and the 5 passages closest to the question are picked. It finds \u201cpenalties\u201d for "
            "\u201cfines\u201d, but can miss exact terms like \u201cArticle 99\u201d.",
            "Gemini answers from those 5 passages with the standard instruction.",
        ],
    },
    "cheap": {
        "origin": "Picked by hand to test fine-tuning.",
        "points": [
            "Same small search model (bge-small), retrained on the AI Act with LoRA so it knows terms like "
            "\u201cdeployer\u201d or \u201cconformity assessment\u201d.",
            "Shorter passages (600 characters), still 5 read.",
            "The \u201ccombine\u201d instruction: connect passages that refer to each other, answer the "
            "parts that are supported, say which part is missing.",
        ],
    },
    "best": {
        "origin": "Found by evolutionary search (NSGA-II), then checked with real answers.",
        "points": [
            "Query planning: Gemini first writes up to 3 extra searches, e.g. one for a rule and one for its fine.",
            "Hybrid search: 37% meaning, 63% keyword matching, because legal text is full of exact terms and numbers.",
            "Reads 13 long passages (1,600 characters), so whole articles usually arrive intact.",
            "The \u201ccombine\u201d instruction, as in Low-cost tuned. About 4\u00d7 the cost: one extra planning call and more text read.",
        ],
    },
}


def doc_label(doc_id: str) -> str:
    kind, number = doc_id.split("_", 1)
    return f"Art. {number}" if kind == "art" else f"Annex {number}"


def verdict(record: dict, qtype: str) -> str:
    if qtype == "unanswerable":
        return "declined" if record["abstained"] else "explained"
    if record["abstained"]:
        return "refused"
    return {1.0: "correct", 0.5: "partial"}.get(record["correctness"], "incorrect")


def main() -> None:
    files = glob.glob(str(RESULTS_DIR / "final_test_*.json"))
    if not files:
        raise SystemExit("No final_test_*.json found: run `python -m evo_rag.final_eval --test --yes` first.")
    final = json.loads(Path(max(files, key=os.path.getmtime)).read_text())
    evals = [json.loads(line) for line in (RESULTS_DIR / "evals.jsonl").read_text().splitlines()]
    qa = {q["qid"]: q for q in map(json.loads, QA_PATH.read_text(encoding="utf-8").splitlines())}

    pipelines, answers = [], {}
    for entry in final:
        same = [e for e in evals if e["split"] == "test" and e["config"] == entry["config"]]
        if not same:
            raise SystemExit(f"No logged test evaluation for pipeline {entry['name']}.")
        details = [json.loads(line) for line in
                   (RESULTS_DIR / "details" / f"{same[-1]['eval_id']}.jsonl").read_text().splitlines()]
        m = entry["metrics"]
        explained = sum(1 for d in details if d["type"] == "unanswerable" and not d["abstained"])
        n_unans = sum(1 for d in details if d["type"] == "unanswerable")
        by_type = {t: m.get(f"correctness_{t}") for t in ("single_hop", "multi_hop", "numerical", "unanswerable")}
        if n_unans and by_type["unanswerable"] is not None:
            by_type["unanswerable_lenient"] = by_type["unanswerable"] + explained / n_unans
        pipelines.append({
            "key": entry["name"], "label": LABELS.get(entry["name"], entry["name"]),
            "description": DESCRIPTIONS.get(entry["name"], ""),
            "details": DETAILS.get(entry["name"]),
            "strict": m["correctness"], "lenient": m["correctness"] + explained / len(details),
            "byType": by_type, "faith": m["faithfulness"],
            "cost": m["cost_usd_per_query"] * 1000,
        })
        for d in details:
            answers.setdefault(d["qid"], {})[entry["name"]] = {
                "text": d["answer"], "verdict": verdict(d, d["type"]), "reason": d["reason"],
                "docs": list(dict.fromkeys(d["retrieved_docs"])),
                "queries": d.get("queries", [])[1:],
            }

    questions = []
    for qid, by_pipeline in answers.items():
        q = qa[qid]
        questions.append({
            "qid": qid, "type": q["type"], "question": q["question"], "gold": q["answer"],
            "goldDocs": q["doc_ids"],
            "evidence": [{"doc": doc_label(d), "quote": quote} for d, quote in q["evidence"]],
            "answers": by_pipeline,
        })
    order = {"single_hop": 0, "multi_hop": 1, "numerical": 2, "unanswerable": 3}
    questions.sort(key=lambda q: (order.get(q["type"], 9), q["qid"]))

    data = {"pipelines": pipelines, "questions": questions,
            "docLabels": {d: doc_label(d) for q in questions for a in q["answers"].values() for d in a["docs"]}
            | {d: doc_label(d) for q in questions for d in q["goldDocs"]}}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(questions)} questions, {len(pipelines)} pipelines, "
          f"{OUT_PATH.stat().st_size / 1024:.0f} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>evo-rag: evolved RAG for the EU AI Act</title>
<meta name="description" content="Compare answers from three RAG pipelines on held-out questions about the EU AI Act.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>
:root {
  color-scheme: light;
  --paper: #f5f6f8; --sheet: #ffffff; --ink: #18223a; --ink-2: #4a5468; --ink-3: #6b7486;
  --rule: #d9dde5; --blue: #2848a8; --blue-soft: #e6ebf7;
  --ok: #2f7d4f; --ok-bg: #e5f2ea; --mid: #9a6a12; --mid-bg: #f8efd9;
  --bad: #b03a3a; --bad-bg: #f8e4e4;
  --serif: "Source Serif 4", Georgia, serif; --sans: "IBM Plex Sans", system-ui, sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 var(--sans); }
a { color: var(--blue); }
:focus-visible { outline: 3px solid var(--blue); outline-offset: 2px; border-radius: 4px; }
button { font: inherit; color: inherit; }

.page { max-width: 1180px; margin: 0 auto; padding: 0 24px; }

/* ---------- "How it was built" timeline ---------- */
.timeline { list-style: none; margin: 18px 0 0; padding: 0; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 18px; position: relative; }
.timeline::before { content: ""; position: absolute; left: 18px; right: calc((100% - 54px) / 4 - 18px); top: 17px; height: 2px; background: var(--rule); }
.tl-step { all: unset; box-sizing: border-box; cursor: pointer; display: block; width: 100%; padding: 0 10px 12px 0; position: relative; border-radius: 8px; }
.tl-step .n { width: 36px; height: 36px; border-radius: 50%; background: var(--paper); border: 2px solid var(--blue); color: var(--blue); font: 600 17px/1 var(--serif); display: grid; place-items: center; margin-bottom: 12px; position: relative; transition: background .15s, color .15s; box-shadow: 0 0 0 5px var(--paper); }
.tl-step strong { display: block; font: 600 17px/1.3 var(--serif); }
.tl-step .brief { display: block; font-size: 14px; color: var(--ink-2); line-height: 1.45; margin-top: 4px; }
.tl-step:hover .n, .tl-step[aria-selected="true"] .n { background: var(--blue); color: #fff; }
.tl-step[aria-selected="true"] strong { color: var(--blue); }
.tl-step[aria-selected="true"]::after { content: ""; position: absolute; left: 12px; bottom: -14px; width: 14px; height: 14px; background: var(--sheet); border-left: 1px solid var(--rule); border-top: 1px solid var(--rule); transform: rotate(45deg); z-index: 2; }
.tl-panel { position: relative; margin-top: 6px; background: var(--sheet); border: 1px solid var(--rule); border-radius: 12px; padding: 26px 30px; }
.tl-panel .step { max-width: 78ch; }
.tl-close { position: absolute; top: 12px; right: 12px; border: 0; background: none; color: var(--ink-3); font-size: 24px; line-height: 1; padding: 4px 10px; border-radius: 6px; cursor: pointer; }
@media (prefers-reduced-motion: no-preference) { .tl-panel .step:not([hidden]) { animation: settle .2s ease-out; } }
.step h3 { font: 600 24px/1.2 var(--serif); margin: 0 0 4px; }
.step .sub { color: var(--ink-3); font-size: 15px; margin: 0 0 16px; }
.step h4 { font-size: 15px; font-weight: 600; margin: 20px 0 6px; }
.step p, .step li { font-size: 15px; color: var(--ink-2); }
.step p { margin: 0 0 10px; }
.step ul { margin: 0 0 10px; padding-left: 18px; } .step li { margin-bottom: 5px; }
.step li b { color: var(--ink); font-weight: 600; }
.step table { width: 100%; border-collapse: collapse; font-size: 14px; margin: 6px 0 12px; }
.step th, .step td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--rule); font-variant-numeric: tabular-nums; }
.step th { color: var(--ink-3); font-weight: 500; }
.step td.num, .step th.num { text-align: right; }
.step tr.top td { background: var(--blue-soft); }
.step .eg { background: var(--paper); border-radius: 6px; padding: 12px 14px; margin: 8px 0 12px; font-size: 14.5px; color: var(--ink-2); }
.step .eg q { font-family: var(--serif); font-style: italic; color: var(--ink); }
.step .takeaway { border-top: 1px solid var(--rule); padding-top: 10px; margin-top: 14px; font-size: 15px; color: var(--ink); }
.step figure { margin: 10px 0 4px; }
.step figure img { width: 100%; max-width: 760px; border: 1px solid var(--rule); border-radius: 6px; }
.step figcaption { font-size: 13px; color: var(--ink-3); margin-top: 4px; }

/* ---------- main column ---------- */
main { min-width: 0; }
header.top { padding: 56px 0 24px; }
header.top h1 { font: 600 clamp(30px, 4vw, 44px)/1.12 var(--serif); margin: 0 0 14px; max-width: 22ch; letter-spacing: -0.01em; }
header.top p { max-width: 66ch; color: var(--ink-2); margin: 0 0 10px; }

.results { margin: 22px 0 8px; width: 100%; border-collapse: collapse; background: var(--sheet); border: 1px solid var(--rule); border-radius: 10px; }
.results th, .results td { text-align: left; padding: 12px 14px; border-bottom: 1px solid var(--rule); font-variant-numeric: tabular-nums; vertical-align: top; }
.results th { font-weight: 500; color: var(--ink-3); font-size: 14px; }
.results tr:last-child td { border-bottom: 0; }
.results td.name { min-width: 150px; white-space: nowrap; }
.results td small { display: block; white-space: nowrap; color: var(--ink-3); font-size: 12.5px; line-height: 1.35; margin-top: 2px; }
.results td.strong { font-weight: 600; }
.results th .pipe.hdr { font-weight: 500; color: var(--ink-3); }
.results tr.best td { background: var(--blue-soft); }
.pipe { all: unset; cursor: help; font-weight: 600; border-bottom: 1.5px dotted var(--ink-3); }
.pipe:hover, .pipe:focus-visible { border-bottom-color: var(--blue); color: var(--blue); }
.table-scroll { overflow-x: auto; }
.models { display: flex; flex-wrap: wrap; gap: 10px 28px; margin: 18px 0 0; padding: 12px 16px; background: var(--sheet); border: 1px solid var(--rule); border-radius: 10px; max-width: max-content; }
.models div { min-width: 0; }
.models dt { font-size: 13px; color: var(--ink-3); }
.models dd { margin: 0; font-size: 15px; font-weight: 500; }
.footnote { font-size: 14px; color: var(--ink-3); max-width: 80ch; }

.tip { position: fixed; z-index: 30; width: min(380px, calc(100vw - 24px)); background: var(--ink); color: #eef1f8; border-radius: 10px; padding: 14px 16px; font-size: 14px; line-height: 1.5;
  box-shadow: 0 14px 40px -14px rgba(0,0,0,.5); visibility: hidden; opacity: 0; }
.tip.show { visibility: visible; opacity: 1; }
.tip strong { display: block; color: #fff; font-size: 15px; margin-bottom: 2px; }
.tip .origin { color: #b9c3dc; font-size: 13px; margin: 0 0 8px; }
.tip p { margin: 0 0 8px; } .tip p:last-child { margin: 0; }
.tip q { font-style: italic; color: #fff; }
.tip ul { margin: 0; padding-left: 16px; } .tip li { margin-bottom: 4px; }

h2.section { font: 600 28px/1.2 var(--serif); margin: 44px 0 6px; }
.lede { color: var(--ink-2); margin: 0 0 20px; max-width: 70ch; }

.browser { display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 20px; align-items: start; }
.filters { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.filters button { font: 500 14px var(--sans); border: 1px solid var(--rule); background: var(--sheet); color: var(--ink-2); padding: 6px 12px; border-radius: 999px; cursor: pointer; }
.filters button[aria-pressed="true"] { background: var(--ink); color: #fff; border-color: var(--ink); }
.legend { font-size: 13px; color: var(--ink-3); margin: 0 0 8px; }
.list { list-style: none; margin: 0; padding: 0; max-height: 72vh; overflow: auto; border: 1px solid var(--rule); border-radius: 10px; background: var(--sheet); }
.list button { all: unset; box-sizing: border-box; display: block; width: 100%; padding: 12px 14px; border-bottom: 1px solid var(--rule); cursor: pointer; font: 15px/1.4 var(--serif); }
.list li:last-child button { border-bottom: 0; }
.list button:hover { background: #f0f2f6; }
.list button[aria-current="true"] { background: var(--blue-soft); box-shadow: inset 3px 0 0 var(--blue); }
.list .type { display: block; font: 500 12px var(--sans); color: var(--ink-3); margin-bottom: 3px; }
.list .marks { display: flex; gap: 4px; margin-top: 6px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

.detail { background: var(--sheet); border: 1px solid var(--rule); border-radius: 10px; padding: 26px 28px; min-height: 60vh; }
.detail .qtype { font-size: 14px; color: var(--ink-3); margin: 0 0 6px; }
.detail h3 { font: 600 23px/1.3 var(--serif); margin: 0 0 20px; max-width: 46ch; }
.reference { border-left: 3px solid var(--ink); padding: 4px 0 4px 18px; margin-bottom: 26px; }
.reference .label { font-size: 14px; color: var(--ink-3); margin-bottom: 4px; }
.reference .text { font: 18px/1.6 var(--serif); max-width: 66ch; }
.evidence { margin-top: 10px; font-size: 14px; color: var(--ink-2); }
.evidence q { font-family: var(--serif); font-style: italic; }
.answers { display: grid; gap: 14px; }
.answer { border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; }
.answer.best { border-color: #b9c6ea; }
.answer .head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; flex-wrap: wrap; margin-bottom: 8px; }
.answer .who { font-weight: 600; }
.answer .text { font: 16.5px/1.6 var(--serif); white-space: pre-wrap; max-width: 70ch; margin: 0 0 10px; }
.badge { font: 600 13px var(--sans); padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
.v-correct, .v-declined { color: var(--ok); background: var(--ok-bg); }
.v-partial, .v-explained { color: var(--mid); background: var(--mid-bg); }
.v-incorrect, .v-refused { color: var(--bad); background: var(--bad-bg); }
.d-correct, .d-declined { background: var(--ok); } .d-partial, .d-explained { background: #c8962c; }
.d-incorrect, .d-refused { background: var(--bad); }
.chips { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; font-size: 13px; color: var(--ink-3); }
.chip { border: 1px solid var(--rule); border-radius: 6px; padding: 1px 7px; color: var(--ink-2); background: #fafbfc; }
.chip.hit { border-color: var(--ok); color: var(--ok); background: var(--ok-bg); font-weight: 600; }
.plan { margin-top: 8px; font-size: 14px; color: var(--ink-2); }
.plan ul { margin: 4px 0 0; padding-left: 18px; }
.reason { margin-top: 6px; font-size: 13px; color: var(--ink-3); }
footer { margin: 56px 0 40px; padding-top: 20px; border-top: 1px solid var(--rule); color: var(--ink-3); font-size: 14px; }

@media (max-width: 860px) {
  .page { padding: 0 16px; }
  .browser { grid-template-columns: 1fr; }
  .list { max-height: 40vh; }
  .detail { padding: 20px; }
  .timeline { grid-template-columns: 1fr; gap: 6px; }
  .timeline::before { left: 17px; right: auto; top: 18px; bottom: 18px; width: 2px; height: auto; }
  .tl-step { display: grid; grid-template-columns: 36px 1fr; column-gap: 12px; padding: 6px 0; }
  .tl-step .n { grid-row: span 2; margin: 0; }
  .tl-step[aria-selected="true"]::after { display: none; }
  .tl-panel { padding: 20px 18px; }
}
@media (prefers-reduced-motion: no-preference) {
  .detail.swap { animation: settle .18s ease-out; }
  @keyframes settle { from { opacity: .4; } to { opacity: 1; } }
}
</style>
</head>
<body>
<div class="page">



<main>
<header class="top">
  <h1>Evolving question-answering pipelines for the EU AI Act</h1>
  <p>A question-answering system for the EU AI Act: it searches the law for the relevant passages, then writes an answer from them. I used evolutionary search (NSGA-II) to optimise the RAG pipeline for the best balance of accuracy and cost and finally fine-tuned the search model on the Act's vocabulary.</p>
  <p>The numbers below come from 75 questions set aside for the final test and never used during development. Hover over a pipeline name to see how it works, then browse the questions and compare each answer with the verified reference answer.</p>
  <dl class="models">
    <div><dt>Writes, plans and grades answers</dt><dd>Gemini 3.5 Flash-Lite, used as is through Google's API</dd></div>
    <div><dt>Finds the passages</dt><dd>bge-small-en-v1.5 (BAAI), fine-tuned with LoRA for the Low-cost tuned pipeline</dd></div>
  </dl>
  <div class="table-scroll">
  <table class="results" id="results">
    <thead><tr><th>Pipeline</th><th><button class="pipe hdr" type="button" data-tip="overall" aria-describedby="tip">Overall correct</button></th><th>Single article</th><th>Multi-step</th><th>Numbers and dates</th><th><button class="pipe hdr" type="button" data-tip="unans" aria-describedby="tip">Not in the Act</button></th><th>Grounded in sources</th><th>Cost per 1,000 questions</th></tr></thead>
    <tbody></tbody>
  </table>
  </div>
  <p class="footnote">The two scoring rules differ only on questions the Act doesn't answer. The strict rule counts a reply as correct only if it uses the exact sentence "The provided context does not contain the answer"; the lenient rule also accepts replies that say the same in other words. Every such reply was reviewed by hand, and all were correct. Overall correct shows the strict score first.</p>
</header>

<section aria-labelledby="explore">
  <h2 class="section" id="explore">Compare the answers</h2>
  <p class="lede">Pick a question. The reference answer was written and checked by hand; the article chips show what each pipeline read, highlighted where they match the articles that hold the answer.</p>
  <div class="browser">
    <div>
      <div class="filters" role="group" aria-label="Filter by question type" id="filters"></div>
      <p class="legend" id="legend"></p>
      <ul class="list" id="list"></ul>
    </div>
    <article class="detail" id="detail" aria-live="polite"></article>
  </div>
</section>

<section class="built" aria-labelledby="how">
  <h2 class="section" id="how">How it was built</h2>
  <p class="lede" id="how-hint">Four steps, in order. Hover over or select a step for the details.</p>
  <ol class="timeline" role="tablist" aria-label="Project steps">
    <li><button class="tl-step" type="button" id="tab1" role="tab" aria-selected="false" aria-controls="p1"><span class="n" aria-hidden="true">1</span><strong>Build the test bed</strong><span class="brief">The AI Act, 253 verified questions, an LLM judge and a spending limit.</span></button></li>
    <li><button class="tl-step" type="button" id="tab2" role="tab" aria-selected="false" aria-controls="p2"><span class="n" aria-hidden="true">2</span><strong>Evolve the pipeline</strong><span class="brief">Optimise the RAG pipeline through evolutionary search (NSGA-II) for the best accuracy–cost trade-offs.</span></button></li>
    <li><button class="tl-step" type="button" id="tab3" role="tab" aria-selected="false" aria-controls="p3"><span class="n" aria-hidden="true">3</span><strong>Fix the weak spot</strong><span class="brief">Multi-step questions, fixed with planned searches and the combine instruction.</span></button></li>
    <li><button class="tl-step" type="button" id="tab4" role="tab" aria-selected="false" aria-controls="p4"><span class="n" aria-hidden="true">4</span><strong>Fine-tune the search model</strong><span class="brief">Gradient descent vs CMA-ES at equal size, then bigger adapters. LoRA wins.</span></button></li>
  </ol>
  <div class="tl-panel" id="tl-panel" hidden>
    <button class="tl-close" id="tl-close" type="button" aria-label="Close details">×</button>
  <section class="step" role="tabpanel" id="p1" hidden>
    <h3>Build the test bed</h3>
    <p class="sub">Before optimising anything, you need a score you can trust.</p>
    <h4>The law</h4>
    <p>The AI Act as published in July 2024, split into its 113 articles and 13 annexes. Every passage remembers which article it came from, so the search can be checked against known answers.</p>
    <h4>253 questions with verified answers</h4>
    <ul>
      <li><b>Single article:</b> the answer is in one place.</li>
      <li><b>Multi-step:</b> needs two or more articles, such as a rule and its fine.</li>
      <li><b>Numbers and dates:</b> amounts, percentages, deadlines.</li>
      <li><b>Not in the Act:</b> the right move is to say so.</li>
    </ul>
    <p>Each has a reference answer and the exact sentence of the Act that supports it. Drafted with an LLM, every quote checked automatically against the text, every question checked by hand. 75 were locked away for the final test shown on this page.</p>
    <h4>An LLM as judge</h4>
    <p>Gemini 3.5 Flash-Lite, the same model that writes the answers, scores each one for <b>correctness</b> against the reference (right, partly right, wrong) and for <b>staying grounded</b> in the passages it read. Because a model grading its own kind can be lenient, the judge was checked against human labels. It agreed with my labels on every clear-cut case, and repeated runs varied by under one percentage point.</p>
    <h4>A hard spending limit</h4>
    <p>Every model call is cached and counted, and the code refuses to spend past the budget. The whole project cost about €3.75.</p>
    
  </section>
  <section class="step" role="tabpanel" id="p2" hidden>
    <h3>Evolve the pipeline</h3>
    <p class="sub">Every design choice is a gene. NSGA-II, a multi-objective evolutionary algorithm, breeds pipelines that are both accurate and cheap.</p>
    <h4>The genes being optimised</h4>
    <ul>
      <li><b>Chunker:</b> how the law is cut into passages. Fixed cuts every N characters, recursive cuts at natural breaks such as paragraphs, semantic cuts where the meaning shifts.</li>
      <li><b>Chunk size:</b> how long each passage is, from 200 to 2,000 characters.</li>
      <li><b>Chunk overlap:</b> how much neighbouring passages share (0–30%), so a sentence split at a boundary isn't lost.</li>
      <li><b>Chunk header:</b> whether each passage starts with its article title, such as “Article 5: Prohibited AI practices”.</li>
      <li><b>Embedder:</b> the model that turns text into meaning vectors (bge-small, bge-base or e5-base).</li>
      <li><b>Retrieval method:</b> search by meaning (dense), by keywords (BM25), or both (hybrid).</li>
      <li><b>Hybrid weight:</b> in hybrid search, how much meaning counts against keywords.</li>
      <li><b>Passages read (top-k):</b> how many passages the model reads before answering, 1 to 20.</li>
      <li><b>Query planning:</b> whether the model first plans extra searches (added in step 3).</li>
    </ul>
    <h4>A free stand-in score</h4>
    <p>Judging every candidate with the real model was too expensive, so evolution optimised a free score: does the search find the passages that hold the answer? The first version was gamed. Tiny passages found fragments of the right article without the answer. A stricter version, does the search find the exact answer sentence, fixed it and ranked pipelines almost exactly like real accuracy.</p>
    <h4>Result</h4>
    <p>With the same budget of 127 pipelines, NSGA-II found a better set of trade-offs than random search (+6% hypervolume). The best pipelines were then checked with real answers.</p>
    <figure><img src="evolve.png" alt="Left: trade-off fronts of NSGA-II and random search. Right: hypervolume against pipelines evaluated, with NSGA-II ending ahead."><figcaption>Evolution against random search, at equal budget.</figcaption></figure>
    <p class="takeaway">Optimisers exploit weak scores. Checking the result with a stricter score caught it for free.</p>
    
  </section>
  <section class="step" role="tabpanel" id="p3" hidden>
    <h3>Fix the weak spot</h3>
    <p class="sub">Questions whose answer sits in a different article from the one they mention.</p>
    <h4>Planned searches</h4>
    <div class="eg"><q>What is the maximum fine for using a prohibited AI practice?</q><br>A single search finds Article 5, which lists the practices, but not Article 99, which sets the fine. With planning, the model first writes extra searches such as “penalties for non-compliance”, so Article 99 gets in. Results from each search take turns filling the same number of slots, so planning changes what is read, not how much.</div>
    <p>Numbers and dates on development questions: 83% → 95% correct.</p>
    <h4>The combine instruction</h4>
    <p>Multi-step questions still failed. A diagnosis showed that in every failure, all the evidence had been found and the model refused anyway, because no single passage answered the whole question. A new instruction, to follow references between passages, answer the supported parts and say which part is missing, fixed it: 78% → 89% on development questions, while still declining questions the Act doesn't answer.</p>
    <h4>Evolution learns when to plan</h4>
    <p>Given planning as a gene, NSGA-II learned on its own to use it only when at least 4 passages are read. With fewer slots, the extra searches crowd out the passage the question needed.</p>
    <figure><img src="validate.png" alt="Correctness against cost with real answers: the final set of pipelines sits above the initial random-search baseline."><figcaption>Real accuracy against cost: final pipelines compared with the initial random search.</figcaption></figure>
    <p class="takeaway">Diagnose before optimising: a free failure analysis pointed to a one-instruction fix.</p>
    
  </section>
  <section class="step" role="tabpanel" id="p4" hidden>
    <h3>Fine-tune the search model</h3>
    <p class="sub">Can the small search model (bge-small-en-v1.5, 33 million parameters) learn the Act's vocabulary, and can an evolutionary method train it as well as gradients? Gemini itself was never fine-tuned.</p>
    <h4>Training data</h4>
    <p>The model wrote realistic questions for passages of every article, several hundred pairs. Anything too similar to the test questions was dropped, and the test questions were never used for training.</p>
    <h4>Four methods, same data</h4>
    <table>
      <tr><th>Method</th><th class="num">Size</th><th>Trained by</th><th class="num">Answer found</th></tr>
      <tr><td>No fine-tuning</td><td class="num">–</td><td>–</td><td class="num">77.5%</td></tr>
      <tr><td>Small adapter</td><td class="num">384</td><td>gradient descent</td><td class="num">79.1%</td></tr>
      <tr><td>Small adapter</td><td class="num">384</td><td>CMA-ES (evolutionary)</td><td class="num">77.2%</td></tr>
      <tr><td>Large adapter</td><td class="num">147k</td><td>gradient descent</td><td class="num">85.4%</td></tr>
      <tr class="top"><td>LoRA</td><td class="num">147k</td><td>gradient descent</td><td class="num">85.4%</td></tr>
    </table>
    <ul>
      <li><b>At equal size, evolution and gradients tied.</b> CMA-ES optimised the ranking directly, without gradients.</li>
      <li><b>Size is what helps.</b> 384 weights can only rescale; 147k can reshape how the model reads the Act.</li>
      <li><b>It helps cheap pipelines, not the best one,</b> which already finds about 98% of answers by reading more and adding keyword search.</li>
    </ul>
    <p class="takeaway">LoRA was kept for the Low-cost tuned pipeline. It runs at the same size and speed as the original model.</p>
    
  </section>
  </div>
</section>

<footer>Corpus: Regulation (EU) 2024/1689 as published in OJ L, 12.7.2024; later amendments are not included. Answers were generated in advance; this page makes no live model calls. Not legal advice.</footer>
</main>
</div>

<div class="tip" id="tip" role="tooltip"></div>


<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const TYPE_LABELS = { single_hop: "Single article", multi_hop: "Multi-step", numerical: "Numbers and dates", unanswerable: "Not in the Act" };
const VERDICT_LABELS = { correct: "Correct", partial: "Partly correct", incorrect: "Incorrect", refused: "Declined, but answerable",
  declined: "Correctly declined", explained: "Explained it isn't specified" };
const pct = x => x == null ? "–" : (100 * x).toFixed(0) + "%";
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const hoverable = window.matchMedia("(hover: hover)").matches;

/* ---------- "How it was built" timeline ---------- */
const tlSteps = [...document.querySelectorAll(".tl-step")];
const tlPanel = document.getElementById("tl-panel");
if (!hoverable) document.getElementById("how-hint").textContent = "Four steps, in order. Tap a step for the details.";
let hoverTimer = null;
function selectStep(btn) {
  tlSteps.forEach(b => {
    const on = b === btn;
    b.setAttribute("aria-selected", on);
    b.tabIndex = on || (!btn && b === tlSteps[0]) ? 0 : -1;
    document.getElementById(b.getAttribute("aria-controls")).hidden = !on;
  });
  tlPanel.hidden = !btn;
}
tlSteps.forEach((b, i) => {
  b.addEventListener("click", () => selectStep(b.getAttribute("aria-selected") === "true" ? null : b));
  if (hoverable) {
    b.addEventListener("mouseenter", () => { hoverTimer = setTimeout(() => selectStep(b), 150); });
    b.addEventListener("mouseleave", () => clearTimeout(hoverTimer));
  }
  b.addEventListener("keydown", e => {
    const k = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!k) return;
    e.preventDefault();
    const next = tlSteps[(i + k + tlSteps.length) % tlSteps.length];
    selectStep(next); next.focus();
  });
});
selectStep(null);
document.getElementById("tl-close").addEventListener("click", () => {
  const open = tlSteps.find(b => b.getAttribute("aria-selected") === "true");
  selectStep(null); if (open) open.focus();
});
document.addEventListener("keydown", e => { if (e.key === "Escape") hideTip(); });

/* ---------- results table + pipeline tooltips ---------- */
const tip = document.getElementById("tip");
let tipFor = null;
const HEADER_TIPS = {
  overall: `<strong>Overall correct</strong><p>Share of all 75 test questions answered correctly (partly correct counts half). The big number uses the strict rule; the small line counts replies that correctly explain, in their own words, that the Act doesn't cover something.</p>`,
  unans: `<strong>Not in the Act</strong><p>Questions the Act doesn't answer, where the right move is to say so. Example: <q>What minimum accuracy percentage must high-risk AI systems reach?</q> The Act sets none.</p><p>Strict: only the exact sentence "The provided context does not contain the answer" counts. Lenient: a reply such as <q>The Act sets no minimum percentage; Article 15 only requires an appropriate level of accuracy</q> also counts.</p>`,
};
function showTip(el, p) {
  if (typeof p === "string") tip.innerHTML = HEADER_TIPS[p];
  else {
    const d = p.details || {};
    tip.innerHTML = `<strong>${esc(p.label)}</strong><p class="origin">${esc(p.description)}. ${esc(d.origin || "")}</p><ul>${(d.points || []).map(x => `<li>${esc(x)}</li>`).join("")}</ul>`;
  }
  tip.classList.add("show"); tipFor = el;
  const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
  let left = Math.min(Math.max(12, r.left), window.innerWidth - w - 12);
  let top = r.bottom + 10;
  if (top + h > window.innerHeight - 12) top = Math.max(12, r.top - h - 10);
  tip.style.left = left + "px"; tip.style.top = top + "px";
}
function hideTip() { tip.classList.remove("show"); tipFor = null; }
window.addEventListener("scroll", hideTip, { passive: true });

const tbody = document.querySelector("#results tbody");
DATA.pipelines.forEach(p => {
  const tr = document.createElement("tr");
  if (p.key === "best") tr.className = "best";
  const t = p.byType || {};
  const unans = t.unanswerable_lenient != null && Math.round(100 * t.unanswerable_lenient) !== Math.round(100 * t.unanswerable)
    ? `${pct(t.unanswerable)} <small>${pct(t.unanswerable_lenient)} lenient</small>` : pct(t.unanswerable);
  const overall = Math.round(100 * p.lenient) !== Math.round(100 * p.strict)
    ? `${pct(p.strict)} <small>${pct(p.lenient)} lenient</small>` : pct(p.strict);
  tr.innerHTML = `<td class="name"><button class="pipe" type="button" aria-describedby="tip">${esc(p.label)}</button></td>
    <td class="strong">${overall}</td><td>${pct(t.single_hop)}</td><td>${pct(t.multi_hop)}</td><td>${pct(t.numerical)}</td><td>${unans}</td>
    <td>${pct(p.faith)}</td><td>$${p.cost.toFixed(2)}</td>`;
  bindTip(tr.querySelector(".pipe"), p);
  tbody.appendChild(tr);
});
document.querySelectorAll(".pipe.hdr").forEach(b => bindTip(b, b.dataset.tip));
function bindTip(b, p) {
  b.addEventListener("mouseenter", () => showTip(b, p));
  b.addEventListener("mouseleave", hideTip);
  b.addEventListener("focus", () => showTip(b, p));
  b.addEventListener("blur", hideTip);
  b.addEventListener("click", e => { e.stopPropagation(); tipFor === b ? hideTip() : showTip(b, p); });
}
document.addEventListener("click", e => { if (tipFor && !tip.contains(e.target)) hideTip(); });

/* ---------- question browser ---------- */
document.getElementById("legend").textContent =
  `Dots show ${DATA.pipelines.map(p => p.label).join(", ")}, in that order: green right, amber partly right, red wrong.`;
let filter = "all", current = null;
const filtersEl = document.getElementById("filters"), listEl = document.getElementById("list"), detailEl = document.getElementById("detail");
const types = ["all", ...Object.keys(TYPE_LABELS).filter(t => DATA.questions.some(q => q.type === t))];
types.forEach(t => {
  const b = document.createElement("button");
  b.type = "button"; b.textContent = t === "all" ? "All" : TYPE_LABELS[t];
  b.setAttribute("aria-pressed", t === filter);
  b.onclick = () => { filter = t; [...filtersEl.children].forEach(c => c.setAttribute("aria-pressed", c === b)); renderList(); };
  filtersEl.appendChild(b);
});
const shownQuestions = () => DATA.questions.filter(q => filter === "all" || q.type === filter);

function renderList() {
  const shown = shownQuestions();
  listEl.innerHTML = "";
  shown.forEach(q => {
    const li = document.createElement("li"), b = document.createElement("button");
    b.type = "button";
    const marks = DATA.pipelines.map(p => {
      const a = q.answers[p.key];
      return a ? `<span class="dot d-${a.verdict}" title="${esc(p.label)}: ${esc(VERDICT_LABELS[a.verdict])}"></span>` : "";
    }).join("");
    b.innerHTML = `<span class="type">${esc(TYPE_LABELS[q.type] || q.type)}</span>${esc(q.question)}<span class="marks" aria-hidden="true">${marks}</span>`;
    b.dataset.qid = q.qid;
    b.setAttribute("aria-current", q.qid === current);
    b.onclick = () => show(q.qid);
    li.appendChild(b); listEl.appendChild(li);
  });
  if (!shown.some(q => q.qid === current) && shown.length) show(shown[0].qid);
}

function show(qid) {
  current = qid;
  const q = DATA.questions.find(x => x.qid === qid);
  listEl.querySelectorAll("button").forEach(b => b.setAttribute("aria-current", b.dataset.qid === qid));
  const gold = new Set(q.goldDocs);
  const evidence = q.evidence.length
    ? `<div class="evidence">Evidence: ${q.evidence.map(e => `<span>${esc(e.doc)}, <q>${esc(e.quote)}</q></span>`).join("; ")}</div>` : "";
  const cards = DATA.pipelines.map(p => {
    const a = q.answers[p.key]; if (!a) return "";
    const chips = a.docs.map(d => `<span class="chip${gold.has(d) ? " hit" : ""}">${esc(DATA.docLabels[d] || d)}</span>`).join("");
    const plan = a.queries.length ? `<div class="plan">Planned searches:<ul>${a.queries.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>` : "";
    return `<section class="answer ${p.key === "best" ? "best" : ""}">
      <div class="head"><span class="who">${esc(p.label)}</span><span class="badge v-${a.verdict}">${esc(VERDICT_LABELS[a.verdict])}</span></div>
      <p class="text">${esc(a.text)}</p>
      <div class="chips">Read: ${chips || "nothing"}</div>${plan}
      <div class="reason">Judge: ${esc(a.reason)}</div></section>`;
  }).join("");
  detailEl.classList.remove("swap"); void detailEl.offsetWidth; detailEl.classList.add("swap");
  detailEl.innerHTML = `<p class="qtype">${esc(TYPE_LABELS[q.type] || q.type)}</p><h3>${esc(q.question)}</h3>
    <div class="reference"><div class="label">Reference answer</div><div class="text">${esc(q.gold)}</div>${evidence}</div>
    <div class="answers">${cards}</div>`;
}
renderList();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
