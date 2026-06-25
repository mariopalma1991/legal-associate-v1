"""
Daily ingestion pipeline — runs all 6 steps in sequence.

Steps:
  1. refresh_status    — update status of existing Vigente records
  2. cleanup_storage   — delete files and chunks for Terminado licitaciones
  3. fetch_vigentes    — discover new Vigente licitaciones from XLS export
  4. ingest            — scrape full metadata for discovered records
  5. download_docs     — download PDFs to Supabase Storage
  6. chunk_docs        — extract text and split into chunks (1024/256)
  7. embed_index       — embed chunks (Cohere)

Usage:
  python run_pipeline.py                        # full run
  python run_pipeline.py --from chunk_docs      # resume from a specific step
  python run_pipeline.py --only download_docs   # run one step only
  python run_pipeline.py --dry-run              # print commands, do not execute
  python run_pipeline.py --continue-on-error    # keep going even if a step fails
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

STEPS = [
    {
        "name":  "refresh_status",
        "cmd":   [sys.executable, "ingestion/refresh_status.py"],
        "desc":  "Refresh status of Vigente licitaciones",
    },
    {
        "name":  "cleanup_storage",
        "cmd":   [sys.executable, "infra/cleanup_storage.py"],
        "desc":  "Delete Storage files and chunks for Terminado licitaciones",
    },
    {
        "name":  "fetch_vigentes",
        "cmd":   [sys.executable, "ingestion/fetch_vigentes.py"],
        "desc":  "Discover new Vigente licitaciones via XLS export",
    },
    {
        "name":  "ingest",
        "cmd":   [sys.executable, "ingestion/ingest.py"],
        "desc":  "Scrape metadata for discovered records",
    },
    {
        "name":  "download_docs",
        "cmd":   [sys.executable, "ingestion/download_docs.py"],
        "desc":  "Download PDFs → Supabase Storage",
    },
    {
        "name":  "chunk_docs",
        "cmd":   [sys.executable, "ingestion/chunk_docs.py",
                  "--chunk-size", "1024", "--overlap", "256"],
        "desc":  "Extract text and split into chunks",
    },
    {
        "name":  "embed_index",
        "cmd":   [sys.executable, "ingestion/embed_index.py", "--model", "cohere"],
        "desc":  "Embed chunks (Cohere)",
    },
]

STEP_NAMES = [s["name"] for s in STEPS]


def _separator(char="─", width=70):
    print(char * width)


def _run_step(step: dict, dry_run: bool) -> bool:
    """Run a single step. Returns True on success, False on failure."""
    cmd_str = " ".join(step["cmd"])
    print(f"\n  $ {cmd_str}")
    if dry_run:
        return True

    t0 = time.time()
    result = subprocess.run(step["cmd"])
    elapsed = time.time() - t0

    mins, secs = divmod(int(elapsed), 60)
    duration = f"{mins}m {secs}s" if mins else f"{secs}s"

    if result.returncode == 0:
        print(f"  ✓  done in {duration}")
        return True
    else:
        print(f"  ✗  FAILED (exit {result.returncode}) after {duration}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Chihuahua licitaciones ingestion pipeline"
    )
    parser.add_argument(
        "--from", dest="from_step", metavar="STEP",
        choices=STEP_NAMES,
        help="Resume from this step (skip all earlier ones)",
    )
    parser.add_argument(
        "--only", metavar="STEP",
        choices=STEP_NAMES,
        help="Run exactly one step and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run without executing them",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep running subsequent steps even if one fails",
    )
    args = parser.parse_args()

    # Build the list of steps to execute
    if args.only:
        steps = [s for s in STEPS if s["name"] == args.only]
    elif args.from_step:
        start = STEP_NAMES.index(args.from_step)
        steps = STEPS[start:]
    else:
        steps = STEPS

    started_at = datetime.now()
    _separator("═")
    print(f"  Licitaciones pipeline — {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print("  MODE: dry-run (no commands will be executed)")
    _separator("═")

    results = []
    pipeline_t0 = time.time()

    for i, step in enumerate(steps, 1):
        _separator()
        print(f"  [{i}/{len(steps)}]  {step['name']}  —  {step['desc']}")
        _separator()

        ok = _run_step(step, args.dry_run)
        results.append((step["name"], ok))

        if not ok and not args.continue_on_error:
            print(f"\n  Pipeline stopped at '{step['name']}'. "
                  f"Fix the error and resume with:  --from {step['name']}", file=sys.stderr)
            break

    # Summary
    total = time.time() - pipeline_t0
    total_mins, total_secs = divmod(int(total), 60)
    duration_str = f"{total_mins}m {total_secs}s" if total_mins else f"{total_secs}s"

    _separator("═")
    print(f"  Summary  ({duration_str} total)")
    _separator("─")
    all_ok = True
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"    {status}  {name}")
        if not ok:
            all_ok = False

    skipped = len(steps) - len(results)
    if skipped:
        for step in steps[len(results):]:
            print(f"    –  {step['name']}  (skipped)")

    _separator("═")
    if all_ok:
        print(f"  Pipeline complete  ✓")
    else:
        print(f"  Pipeline finished with errors  ✗", file=sys.stderr)
    _separator("═")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
