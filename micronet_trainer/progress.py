"""Konsole: zwei feste Fortschrittsbalken (Gesamt nach Zeit, aktueller Trial) und log.txt.

Meldungen gehen über log() oberhalb der Balken (tqdm.write) und mit Zeitstempel in log.txt.
"""

import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

BAR = "{desc:<9}|{bar:28}| {percentage:3.0f}%  {postfix}"


def hm(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d}"


class Console:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.total = None
        self.trial = None
        self.budget_s = 1
        self._last = 0.0

    def log(self, message: str) -> None:
        tqdm.write(message)
        with open(self.log_path, "a", encoding="utf-8") as file:
            for line in message.splitlines() or [""]:
                file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {line}\n")

    def open_bars(self, budget_s: float) -> None:
        self.budget_s = max(1, int(budget_s))
        self.total = tqdm(total=self.budget_s, desc="Gesamt", position=0, leave=True, bar_format=BAR,
                          dynamic_ncols=True)
        self.trial = tqdm(total=1, desc="–", position=1, leave=True, bar_format=BAR, dynamic_ncols=True)

    def overall(self, used_s: float, text: str, force: bool = False) -> None:
        if self.total is None or (not force and time.time() - self._last < 1.0):
            return
        self._last = time.time()
        self.total.n = min(int(used_s), self.budget_s)
        self.total.set_postfix_str(f"{hm(used_s)} / {hm(self.budget_s)}  {text}", refresh=True)

    def trial_start(self, desc: str, total: int) -> None:
        if self.trial is not None:
            self.trial.reset(total=max(1, total))
            self.trial.set_description_str(desc[:9])

    def trial_update(self, n: int, text: str) -> None:
        if self.trial is not None:
            self.trial.n = n
            self.trial.set_postfix_str(text, refresh=True)

    def close(self) -> None:
        for bar in (self.trial, self.total):
            if bar is not None:
                bar.close()
        self.total = self.trial = None
