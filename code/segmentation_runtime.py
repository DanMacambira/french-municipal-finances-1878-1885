"""Small standard-library helpers shared by the two independent segmentation stages.

No third-party progress dependency; output remains readable in terminals and CI.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


class Progress:
    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(0, int(total))
        self.count = 0
        self.start = time.monotonic()
        self.last = -1.0
        self.tty = sys.stderr.isatty()
        self._print(force=True)

    def update(self, count: int = 1) -> None:
        self.count = min(self.total, self.count + count)
        self._print(force=self.count == self.total)

    def _print(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last < .18:
            return
        self.last = now
        pct = (self.count / self.total) if self.total else 1.0
        filled = min(24, round(pct * 24))
        bar = '#' * filled + '-' * (24 - filled)
        elapsed = now - self.start
        eta = ((self.total - self.count) * elapsed / self.count) if self.count else 0
        line = (f'{self.label}: [{bar}] {self.count}/{self.total} '
                f'({pct:.0%}) elapsed {elapsed:.1f}s'
                + (f' ETA {eta:.1f}s' if self.count and self.count < self.total else ''))
        # Avoid flooding redirected logs, while preserving a real progress bar in terminals.
        if self.tty:
            print('\r' + line.ljust(110), end='\n' if force and self.count == self.total else '',
                  file=sys.stderr, flush=True)
        elif force and (self.count == self.total or self.count == 0):
            print(line, file=sys.stderr, flush=True)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start


def contact_sheets(paths: list[Path], folder: Path, *, name: str = 'review_contact_sheet',
                   max_per_sheet: int = 48, tile: tuple[int, int] = (480, 440),
                   columns: int = 3) -> list[Path]:
    """Bound contact sheet memory, even for volumes containing thousands of pages."""
    folder.mkdir(parents=True, exist_ok=True)
    for existing in folder.glob(f'{name}*.jpg'):
        existing.unlink()
    valid = [p for p in paths if p.is_file()]
    if not valid:
        return []
    result: list[Path] = []
    w, h = tile
    for index in range(0, len(valid), max_per_sheet):
        batch = valid[index:index + max_per_sheet]
        rows = (len(batch) + columns - 1) // columns
        sheet = Image.new('RGB', (w * columns, h * rows), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(batch):
            with Image.open(path) as src:
                im = src.convert('RGB')
                im.thumbnail((w - 14, h - 26), Image.Resampling.LANCZOS)
            x, y = (i % columns) * w + 7, (i // columns) * h + 21
            sheet.paste(im, (x, y))
            draw.text((x, y - 17), path.stem[:75], fill='black')
        suffix = '' if index == 0 else f'_{index // max_per_sheet + 1:03d}'
        dest = folder / f'{name}{suffix}.jpg'
        sheet.save(dest, quality=84)
        result.append(dest)
    return result


def _report(title: str, counts: list[tuple[str, int]], total: int, elapsed: float) -> None:
    label_width = max(22, *(len(label) for label, _ in counts))
    print(f'\n{title} | {total} scans | {elapsed:.1f} seconds')
    print(f'  {"Metric":<{label_width}}  {"Count":>7}  {"Share":>8}')
    print('  ' + '-' * (label_width + 20))
    for label, n in counts:
        print(f'  {label:<{label_width}}  {n:>7}  {n / total:.1%}' if total else
              f'  {label:<{label_width}}  {n:>7}       -')


def summary_spreads(rows: list[dict], elapsed: float) -> None:
    tally = Counter(r.get('status', '') for r in rows)
    total = len(rows)
    counts = [
        ('Accepted (all)', tally['accepted'] + tally['accepted_single']),
        ('Spreads split', tally['accepted']),
        ('Single-page passthrough', tally['accepted_single']),
        ('Needs review', sum(n for status, n in tally.items() if status not in {'accepted', 'accepted_single'})),
    ]
    _report('Stage 1 - book spreads', counts, total, elapsed)


def summary_tables(rows: list[dict], elapsed: float) -> None:
    total = len(rows)
    tally = Counter(r.get('status', '') for r in rows)
    # A failed page may have a proposed vertical line but no actual crop.
    # Exclude it from statistics about cuts/methods.
    cropped = [r for r in rows if str(r.get('table_count', '0')) in {'1', '2'}]
    method = Counter(r.get('cut_method', '') for r in cropped)
    passes = Counter(r.get('detection_pass', '') for r in cropped)
    accepted = tally['accepted'] + tally['accepted_single_table']
    pending = sum(n for status, n in tally.items() if status.startswith('accepted_')
                  and status not in {'accepted_single_table'})
    counts = [
        ('Accepted automatically', accepted),
        ('Pending visual review', pending),
        ('Unresolved / review', total - accepted - pending),
        ('Method: vertical', method['vertical']),
        ('Method: linear', method['linear']),
        ('Method: quadratic', method['quadratic']),
        ('Method: no cut', method['none']),
        ('Pass 1: strict', passes['strict']),
        ('Pass 2: tolerant', passes['tolerant']),
        ('Pass 3: extreme', passes['extreme']),
        ('Pass 4: short-table', passes['short_table']),
        ('Single-table bypass', passes['single_table']),
    ]
    _report('Stage 2 - table panels', counts, total, elapsed)
    print('  Note: cut-method/pass counts exclude uncut failures; shares use all printed pages.')
    print('  Pending-review candidates are not OCR-ready.')
