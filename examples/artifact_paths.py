# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve where reviewable run artifacts (videos, figures, reports) are written.

Everything a human is meant to *look at* goes under one project-local root so it
is served by labboard without any per-run action. Scattering videos across
``/data/checkpoints/<ad-hoc-name>/`` makes them invisible: labboard serves only
paths beneath a pinned root, and pinning ``/data/checkpoints`` is not an option
(it is terabytes of weights).

Root resolution order:

1. ``$RLINF_ARTIFACTS`` if set — lets a launcher redirect output without edits.
2. ``<repo>/artifacts`` otherwise — the pinned default. See ``artifacts/README.md``.

Scripts should accept a user-supplied output path and pass it through
:func:`resolve_artifact_dir`, so a *relative* value like
``robotwin-cups/e049-eval`` lands inside the pin while an absolute path is
honoured verbatim (escape hatch for one-off scratch output).
"""

import os
from pathlib import Path

__all__ = ["artifact_root", "resolve_artifact_dir"]

_REPO_ROOT = Path(__file__).resolve().parents[1]


def artifact_root() -> Path:
    """Return the artifact root, honouring ``$RLINF_ARTIFACTS``."""
    env = os.environ.get("RLINF_ARTIFACTS")
    return Path(env).expanduser().resolve() if env else _REPO_ROOT / "artifacts"


def resolve_artifact_dir(path: str, *, create: bool = False) -> Path:
    """Resolve ``path`` against the artifact root unless it is already absolute.

    Args:
        path: A run-relative name (``robotwin-cups/e049-eval``) or an absolute
            path, which is returned unchanged.
        create: Create the directory (and parents) before returning.

    Returns:
        The resolved directory.
    """
    p = Path(path).expanduser()
    resolved = p if p.is_absolute() else artifact_root() / p
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved
