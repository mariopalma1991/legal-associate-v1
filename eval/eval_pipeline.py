"""
eval_pipeline.py — End-to-end evaluation with LLM-as-judge.

For each question in eval_set.json:
  1. Calls query.run() — dual-path DB + RAG + Claude Opus synthesis
  2. Judges the answer with Claude Haiku on Relevance, Faithfulness, Completeness
  3. Appends each result to eval_pipeline_results.jsonl (one JSON line per question)
  4. Writes a markdown summary to eval_pipeline_summary.md when done

Usage:
  python eval_pipeline.py                  # all 295 questions
  python eval_pipeline.py --limit 20       # quick smoke test
  python eval_pipeline.py --skip-existing  # resume an interrupted run
  python eval_pipeline.py --no-rerank      # skip Cohere reranker (faster)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_RESULTS_FILE = "eval_pipeline_results.jsonl"
DEFAULT_SUMMARY_FILE = "eval_pipeline_summary.md"
DEFAULT_LOG_FILE     = "eval_pipeline.log"


def _slug(model: str) -> str:
    """Turn a model name into a safe filename suffix, e.g. gpt-4o-mini → gpt_4o_mini."""
    return model.replace("-", "_").replace(".", "_").replace("/", "_")


class _Tee:
    """Mirrors every write() to both the original stdout and a log file."""
    def __init__(self, log_path: str):
        self._terminal = sys.stdout
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, msg: str):
        # Terminal may not support full unicode on Windows — fall back to ASCII
        try:
            self._terminal.write(msg)
            self._terminal.flush()
        except (UnicodeEncodeError, UnicodeDecodeError):
            self._terminal.write(msg.encode("ascii", "replace").decode("ascii"))
            self._terminal.flush()
        # Log file is always UTF-8
        self._log.write(msg)
        self._log.flush()

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()

_JUDGE_SYSTEM = """\
You are an evaluation assistant scoring an AI procurement assistant's answers.
You receive: the original question, the context provided to the AI, and the AI's answer.

Score each dimension 0–10:
- relevance: Does the answer directly address what was asked?
- faithfulness: Is everything stated in the answer supported by the context? (no hallucinations)
- completeness: Does the answer cover the key information from the context relevant to the question?

Respond with valid JSON only — no other text:
{"relevance": <int>, "faithfulness": <int>, "completeness": <int>, "explanation": "<one sentence>"}\
"""


# ── Judge ─────────────────────────────────────────────────────────────────────

def judge_answer(question: str, context: str, answer: str,
                 model: str = "gpt-4o-mini") -> dict:
    ctx_excerpt = context[:3000] + ("\n[...truncated]" if len(context) > 3000 else "")
    content = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT PROVIDED TO AI:\n{ctx_excerpt}\n\n"
        f"AI ANSWER:\n{answer}"
    )

    def _parse(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {"relevance": 0, "faithfulness": 0, "completeness": 0,
                    "explanation": f"parse error: {text[:80]}"}

    for attempt in range(4):
        try:
            # ── OpenAI judge ──────────────────────────────────────────────
            if not model.startswith("claude"):
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=256,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user",   "content": content},
                    ],
                )
                return _parse(resp.choices[0].message.content.strip())

            # ── Anthropic judge ───────────────────────────────────────────
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=model,
                max_tokens=256,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
            return _parse(resp.content[0].text.strip())

        except Exception as e:
            retryable = (
                "429" in str(e) or "rate" in str(e).lower() or
                "529" in str(e) or "overloaded" in str(e).lower()
            )
            if attempt < 3 and retryable:
                wait = 15 * (2 ** attempt)
                print(f"  [judge] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    return {"relevance": 0, "faithfulness": 0, "completeness": 0,
            "explanation": "max retries exceeded"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_existing_questions(path: str) -> set[str]:
    """Only skip successfully scored questions — errors get retried."""
    done: set[str] = set()
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "error" not in row:
                        done.add(row["question"])
                except Exception:
                    pass
    return done


def load_existing_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def avg(lst: list[dict], key: str) -> float:
    vals = [x[key] for x in lst if isinstance(x.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


# ── Summary writer ────────────────────────────────────────────────────────────

def write_summary(all_results: list[dict], output_path: str):
    by_type: dict[str, list[dict]] = {}
    for r in all_results:
        t = r.get("question_type", "unknown")
        by_type.setdefault(t, []).append(r)

    overall_rel  = avg(all_results, "relevance")
    overall_fai  = avg(all_results, "faithfulness")
    overall_com  = avg(all_results, "completeness")
    overall_avg  = (overall_rel + overall_fai + overall_com) / 3

    lines = [
        "# End-to-End Pipeline Evaluation",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"\nTotal questions evaluated: {len(all_results)}",
        "\n## Overall Scores\n",
        "| Dimension | Score (0–10) |",
        "|-----------|-------------|",
        f"| Relevance    | {overall_rel:.2f} |",
        f"| Faithfulness | {overall_fai:.2f} |",
        f"| Completeness | {overall_com:.2f} |",
        f"| **Average**  | **{overall_avg:.2f}** |",
        "\n## Scores by Question Type\n",
        "| Type | N | Relevance | Faithfulness | Completeness | Average |",
        "|------|---|-----------|--------------|--------------|---------|",
    ]
    for qtype, rows in sorted(by_type.items()):
        r = avg(rows, "relevance")
        f = avg(rows, "faithfulness")
        c = avg(rows, "completeness")
        lines.append(
            f"| {qtype} | {len(rows)} "
            f"| {r:.2f} | {f:.2f} | {c:.2f} | {(r+f+c)/3:.2f} |"
        )

    lines += [
        "\n## Sample Results (best + worst per type)\n",
        "| Type | Question | R | F | C | Note |",
        "|------|----------|---|---|---|------|",
    ]
    for qtype, rows in sorted(by_type.items()):
        scored = sorted(rows, key=lambda x: (
            x.get("relevance", 0) + x.get("faithfulness", 0) + x.get("completeness", 0)
        ))
        for row in (scored[:1] + scored[-1:]):
            q = (row["question"][:55] + "…") if len(row["question"]) > 55 else row["question"]
            note = (row.get("explanation") or "")[:50]
            lines.append(
                f"| {qtype} | {q} "
                f"| {row.get('relevance', 0)} "
                f"| {row.get('faithfulness', 0)} "
                f"| {row.get('completeness', 0)} "
                f"| {note} |"
            )

    latencies = [r["latency_ms"] for r in all_results if "latency_ms" in r]
    if latencies:
        lines += [
            "\n## Performance",
            f"- Avg query latency: {sum(latencies)/len(latencies)/1000:.1f}s",
            f"- Total questions  : {len(all_results)}",
            f"- Skipped/errors   : {sum(1 for r in all_results if 'error' in r)}",
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Summary saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end eval: query.py answers → Claude Haiku judge"
    )
    parser.add_argument("--eval-set",          default="eval_set.json")
    parser.add_argument("--output",            default=None,
                        help="JSONL output (default: eval_pipeline_results_{model}.jsonl)")
    parser.add_argument("--summary",           default=None,
                        help="Markdown summary (default: eval_pipeline_summary_{model}.md)")
    parser.add_argument("--log-file",          default=None,
                        help="Log file (default: eval_pipeline_{model}.log)")
    parser.add_argument("--limit",             type=int, default=None,
                        help="Max questions total (sequential from eval set)")
    parser.add_argument("--limit-per-type",   type=int, default=None,
                        help="Max questions per type (specific/metadata/summary/topic) — ensures balanced variety")
    parser.add_argument("--skip-existing",     action="store_true",
                        help="Resume interrupted run — skip questions already in output")
    parser.add_argument("--model",             choices=["cohere", "openai"], default="cohere",
                        help="Embedding model for RAG (default: cohere)")
    parser.add_argument("--synthesis-model",   default="gpt-4o-mini",
                        help="Model for answer synthesis (default: gpt-4o-mini)")
    parser.add_argument("--judge-model",       default="gpt-4o-mini",
                        help="Model for LLM-as-judge scoring (default: gpt-4o-mini)")
    parser.add_argument("--top-k",             type=int, default=20,
                        help="RAG chunks passed to synthesis (default: 20 for eval cost control)")
    parser.add_argument("--no-rerank",         action="store_true")
    parser.add_argument("--rerank-candidates", type=int, default=200)
    args = parser.parse_args()

    slug = f"{_slug(args.synthesis_model)}__judge_{_slug(args.judge_model)}"
    results_file = args.output  or f"eval_pipeline_results_{slug}.jsonl"
    summary_file = args.summary or f"eval_pipeline_summary_{slug}.md"
    log_file     = args.log_file or f"eval_pipeline_{slug}.log"

    # Mirror all output to log file for real-time monitoring
    tee = _Tee(log_file)
    sys.stdout = tee

    print(f"=== eval_pipeline started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Log file: {os.path.abspath(log_file)}")

    from query import run as query_run, get_conn

    with open(args.eval_set, encoding="utf-8") as f:
        eval_set = json.load(f)

    if args.limit_per_type:
        buckets: dict[str, list] = {}
        for e in eval_set:
            buckets.setdefault(e.get("type", "specific"), []).append(e)
        # Cap each bucket and interleave round-robin for even distribution
        capped = {t: items[:args.limit_per_type] for t, items in buckets.items()}
        order = ["specific", "metadata", "summary", "topic"]
        interleaved = []
        for i in range(args.limit_per_type):
            for t in order:
                if i < len(capped.get(t, [])):
                    interleaved.append(capped[t][i])
        eval_set = interleaved
        counts = {t: len(capped.get(t, [])) for t in order}
        print(f"limit-per-type={args.limit_per_type}  "
              f"→ {len(eval_set)} questions  {counts}")
    elif args.limit:
        eval_set = eval_set[:args.limit]

    done = load_existing_questions(results_file) if args.skip_existing else set()
    todo = [e for e in eval_set if e["question"] not in done]

    prev_rows = load_existing_rows(results_file) if args.skip_existing else []
    print(f"Evaluating {len(todo)} questions ({len(done)} already done, {len(prev_rows)} loaded)")
    print(f"Config: model={args.model}  top_k={args.top_k}  "
          f"rerank={not args.no_rerank}  rerank_candidates={args.rerank_candidates}  "
          f"synthesis={args.synthesis_model}  judge={args.judge_model}")
    print(f"Output: {results_file}  |  Summary: {summary_file}")

    conn = get_conn()
    all_results: list[dict] = list(prev_rows)
    run_start = time.perf_counter()

    with open(results_file, "a", encoding="utf-8") as out_f:
        for i, item in enumerate(todo, 1):
            question = item["question"]
            qtype    = item.get("type", "specific")
            print(f"\n[{i}/{len(todo)}] ({qtype}) {question[:70]}...")

            t0 = time.perf_counter()
            try:
                result = query_run(
                    conn, question,
                    model=args.model,
                    top_k=args.top_k,
                    rerank=not args.no_rerank,
                    rerank_candidates=args.rerank_candidates,
                    synthesis_model=args.synthesis_model,
                    stream=False,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                scores = judge_answer(question, result["context"], result["answer"],
                                     model=args.judge_model)
                r = scores.get("relevance", 0)
                f = scores.get("faithfulness", 0)
                c = scores.get("completeness", 0)
                note = (scores.get("explanation") or "")[:70]
                print(f"  -> R={r}/10  F={f}/10  C={c}/10  | {note}")

                row = {
                    "question":       question,
                    "question_type":  qtype,
                    "answer":         result["answer"],
                    "n_licitaciones": len(result["licitaciones"]),
                    "n_chunks":       len(result["chunks"]),
                    "latency_ms":     elapsed_ms,
                    **scores,
                }

            except Exception as e:
                print(f"  [ERROR] {e}")
                row = {
                    "question":      question,
                    "question_type": qtype,
                    "error":         str(e),
                    "relevance": 0, "faithfulness": 0, "completeness": 0,
                    "explanation": str(e)[:100],
                }

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            all_results.append(row)

            # Print running averages every 10 questions
            if i % 10 == 0:
                scored = [x for x in all_results if "error" not in x]
                if scored:
                    elapsed_total = (time.perf_counter() - run_start) / 60
                    rate = i / elapsed_total if elapsed_total > 0 else 0
                    eta  = (len(todo) - i) / rate if rate > 0 else 0
                    print(f"\n  --- Running avg after {i} questions ---")
                    print(f"  Relevance={avg(scored,'relevance'):.2f}  "
                          f"Faithfulness={avg(scored,'faithfulness'):.2f}  "
                          f"Completeness={avg(scored,'completeness'):.2f}")
                    print(f"  Elapsed: {elapsed_total:.1f} min  ETA: {eta:.1f} min\n")

    conn.close()
    write_summary(all_results, summary_file)

    tee.close()
    sys.stdout = tee._terminal
    print(f"\nDone. Results in {results_file}  |  Summary in {summary_file}")


if __name__ == "__main__":
    main()
