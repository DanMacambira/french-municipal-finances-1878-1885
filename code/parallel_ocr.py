"""Concurrent, resumable Mistral OCR for approved Stage-2 panels.

This module is called by situation_financiere.py when --workers > 1.
It deliberately reuses the original API client, approved panel manifests,
content-hash cache format, and atomic JSON writer. No image splitting or
parser behavior changes are made here.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pymupdf as fitz
import situation_financiere as sf


@dataclass(frozen=True)
class OCRJob:
    year: int
    page_no: int
    side: str
    panel: str
    image: Path
    source_pdf: str
    relative_image: str
    checksum: str
    cache_path: Path


class RequestPacer:
    """Optional shared minimum spacing between starts of OCR jobs.

    NOTE: Retries occur inside the original sf.api_ocr client. Therefore this
    is not a hard limit on *all* HTTP attempts, and an account's Mistral quota
    is still authoritative.
    """

    def __init__(self, rpm: float = 0.0):
        self.interval = (60.0 / rpm) if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
            delay = max(0.0, start - now)
        if delay:
            time.sleep(delay)


def cache_matches(job: OCRJob, model: str) -> bool:
    old = sf.read_json(job.cache_path, {})
    return (
        old.get("image_sha256") == job.checksum
        and old.get("model") == model
        and old.get("segmentation_source") == "segmented"
        and old.get("panel") == job.panel
        and old.get("image") == job.relative_image
        and old.get("source_pdf") == job.source_pdf
    )


def gather_jobs(args):
    """Mirror the original sequential OCR's exact selection and cache rules."""
    pdfs = sf.find_pdfs()
    years = sf.select_years(args, pdfs)
    jobs = []
    skipped = 0
    pending_total = 0
    selected_total = 0
    cache_base = sf.DERIVED / "ocr" / "mistral" / "segmented"

    for year in years:
        with fitz.open(pdfs[year]) as pdf:
            start, end = sf.valid_range(year, pdf)
        if args.pages:
            invalid = [p for p in args.pages if p < start or p > end]
            if invalid:
                raise ValueError(
                    f"{year}: requested page(s) {invalid} lie outside "
                    f"the confirmed {start}-{end} range"
                )
            requested = sorted(set(args.pages))
        else:
            end = min(end, start + args.max_pages - 1) if args.max_pages else end
            requested = list(range(start, end + 1))

        panels, pending = sf.segmented_images(year, requested, dpi=args.dpi)
        pending_total += pending
        selected_total += len(panels)
        if pending:
            print(f"{year}: {pending} review-only printed pages excluded")
        for page_no, side, panel, image in panels:
            out = cache_base / str(year) / f"p{page_no:04d}_{side}_{panel}.json"
            job = OCRJob(
                year=year,
                page_no=page_no,
                side=side,
                panel=panel,
                image=image,
                source_pdf=pdfs[year].name,
                relative_image=str(image.relative_to(sf.ROOT)),
                checksum=hashlib.sha256(image.read_bytes()).hexdigest(),
                cache_path=out,
            )
            if not args.force and cache_matches(job, args.model):
                skipped += 1
            else:
                jobs.append(job)
        print(f"{year}: {len(panels)} approved panels selected", flush=True)

    return jobs, selected_total, skipped, pending_total


def execute_job(job: OCRJob, model: str, pacer: RequestPacer) -> None:
    pacer.wait()
    # Exactly the original, tested provider request with its existing retries.
    response = sf.api_ocr(job.image, model)
    sf.atomic_json(job.cache_path, {
        "year": job.year,
        "pdf_page": job.page_no,
        "side": job.side,
        "panel": job.panel,
        "segmentation_source": "segmented",
        "source_pdf": job.source_pdf,
        "image": job.relative_image,
        "image_sha256": job.checksum,
        "model": model,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "response": response,
    })


def run_jobs(jobs: list[OCRJob], *, workers: int, model: str, rpm: float = 0.0) -> list[tuple[OCRJob, Exception]]:
    """Keep at most `workers` requests in flight; save successes immediately."""
    pacer = RequestPacer(rpm)
    errors: list[tuple[OCRJob, Exception]] = []
    completed = 0
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mistral-ocr") as pool:
        in_flight = {}

        def submit_next():
            try:
                item = next(iterator)
            except StopIteration:
                return
            future = pool.submit(execute_job, item, model, pacer)
            in_flight[future] = item

        for _ in range(min(workers, len(jobs))):
            submit_next()

        while in_flight:
            # as_completed iterates over a snapshot. Break after one completion
            # and immediately replace it to keep the pool saturated.
            future = next(as_completed(in_flight))
            item = in_flight.pop(future)
            try:
                future.result()
                completed += 1
                print(f"[{completed + len(errors)}/{len(jobs)}] Saved "
                      f"{item.year} p{item.page_no:04d} {item.side}_{item.panel}", flush=True)
            except Exception as exc:  # retain all successful outputs; don't hide errors
                errors.append((item, exc))
                print(f"[{completed + len(errors)}/{len(jobs)}] FAILED "
                      f"{item.year} p{item.page_no:04d} {item.side}_{item.panel}: {exc}",
                      flush=True)
            submit_next()
    return errors


def parallel_ocr(args) -> None:
    """Drop-in implementation of the existing `ocr` CLI for --workers > 1."""
    workers = getattr(args, "workers", 4)
    rpm = getattr(args, "rpm", 0.0)
    if not 1 <= workers <= 32:
        raise ValueError("--workers must be between 1 and 32 (start with 4)")
    if rpm < 0:
        raise ValueError("--rpm cannot be negative")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if (args.max_pages == 0 and not args.pages
            and not args.yes_all and not args.dry_run):
        raise ValueError("Full paid OCR run requires --yes-all. "
                         "For a pilot use --max-pages 3 or --pages 245")

    jobs, total, hits, pending = gather_jobs(args)
    print(f"{total} selected approved panels; {hits} valid cache hits; "
          f"{len(jobs)} new requests; {pending} pending-review pages excluded")
    if args.dry_run:
        print("DRY RUN ONLY: no Mistral requests made")
        return
    if not jobs:
        print("All selected panels already cached. No Mistral requests made.")
        return
    if not os.environ.get("MISTRAL_API_KEY"):
        raise RuntimeError("Set MISTRAL_API_KEY in your shell before paid OCR")

    print(f"Concurrent OCR: {workers} workers, "
          f"{'unpaced' if rpm == 0 else f'at most {rpm:g} new job starts/minute'}", flush=True)
    failures = run_jobs(jobs, workers=workers, model=args.model, rpm=rpm)
    print(f"Finished: {len(jobs) - len(failures)} saved; {hits} cache hits; "
          f"{len(failures)} failures; {pending} pending pages skipped", flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} OCR jobs failed. Successful caches have been kept. "
            "Rerun the same command to retry only failures."
        )
