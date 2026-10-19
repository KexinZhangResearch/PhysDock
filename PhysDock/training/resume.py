"""Resuming a run: the newest complete checkpoint of a run directory, the wandb run id that stays the same over restarts, and a ModelCheckpoint
that keeps only the newest checkpoint."""
import os
import re
import secrets
import zipfile
from pathlib import Path
from typing import Optional

from lightning.pytorch.callbacks import ModelCheckpoint


class NewestCheckpoint(ModelCheckpoint):
    """ModelCheckpoint(save_top_k=1) that also deletes the checkpoint the run was resumed from once a newer one is written (Lightning keeps
    that one for good, which leaves one stale file per restart). The new file is always written before the previous one is deleted."""

    def _should_remove_checkpoint(self, trainer, previous: str, current: str) -> bool:
        return previous != current


def _is_complete(path: Path) -> bool:
    """A torch checkpoint is a zip archive whose central directory is written last: a file that a killed process left half-written does not
    open (the ModelCheckpoint callback writes the new checkpoint before it deletes the previous one, so an older complete one exists)."""
    try:
        with zipfile.ZipFile(path) as z:
            return len(z.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def latest_checkpoint(directory) -> Optional[str]:
    """Newest complete step_<N>.ckpt (or last.ckpt) in `directory`, or None. Incomplete files are deleted."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = []
    for path in directory.glob("*.ckpt"):
        m = re.fullmatch(r"step_(\d+)(?:-v\d+)?", path.stem)
        candidates.append((int(m.group(1)) if m else -1, path.stat().st_mtime, path))
    for _, _, path in sorted(candidates, reverse=True):
        if _is_complete(path):
            return str(path)
        print(f"[resume] removing the incomplete checkpoint {path}", flush=True)
        path.unlink()
    return None


def wandb_run_id(output_dir) -> str:
    """The wandb run id of this output directory, created on the first start: every restart then continues the same wandb run."""
    path = Path(output_dir) / "wandb_run_id"
    if path.is_file():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = secrets.token_hex(4)
    path.write_text(run_id)
    return run_id


def wandb_online() -> bool:
    """Whether wandb can log online: an API key in the environment or in ~/.netrc (wandb login)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    return netrc.is_file() and "api.wandb.ai" in netrc.read_text()
