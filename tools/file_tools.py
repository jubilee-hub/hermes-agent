#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools."""

import errno
import fnmatch
import heapq
import json
import logging
import os
import posixpath
import re
import stat
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path, PurePosixPath

from agent.file_safety import get_read_block_error
from tools.binary_extensions import has_binary_extension
from tools.file_operations import (
    LintResult,
    PatchResult,
    ReadResult,
    SearchMatch,
    SearchResult,
    ShellFileOperations,
    WriteResult,
    normalize_read_pagination,
    normalize_search_pagination,
)
from tools import file_state
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)


_EXPECTED_WRITE_ERRNOS = {errno.EACCES, errno.EPERM, errno.EROFS}


def _expand_tilde(path: str) -> str:
    """Expand ``~`` using the effective profile home when available.

    In-process file tools share the gateway process's HOME, which may differ
    from the profile-specific HOME that interactive CLI sessions use.  This
    mirrors ``hermes_constants.get_subprocess_home()`` so that ``~`` resolves
    consistently regardless of whether the tool runs interactively or inside a
    gateway-driven cron job (#48552).
    """
    if not path or "~" not in path:
        return path
    try:
        from hermes_constants import get_subprocess_home

        home = get_subprocess_home()
    except Exception:
        home = None
    if home and (path == "~" or path.startswith("~/")):
        return home if path == "~" else os.path.join(home, path[2:])
    return os.path.expanduser(path)


# ---------------------------------------------------------------------------
# Read-size guard: cap the character count returned to the model.
# We're model-agnostic so we can't count tokens; characters are a safe proxy.
# 100K chars ≈ 25–35K tokens across typical tokenisers.  Files larger than
# this in a single read are a context-window hazard — the model should use
# offset+limit to read the relevant section.
#
# Configurable via config.yaml:  file_read_max_chars: 200000
# ---------------------------------------------------------------------------
_DEFAULT_MAX_READ_CHARS = 100_000
_MAX_SESSION_BUFFERED_FILE_BYTES = 32 * 1024 * 1024
_MAX_SESSION_SEARCH_FILE_BYTES = 8 * 1024 * 1024
_MAX_SESSION_SEARCH_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_SESSION_SEARCH_FILES = 4096
_MAX_SESSION_SEARCH_CONTEXT = 100
_MAX_SESSION_SEARCH_RESULTS = 1000
_MAX_SESSION_SEARCH_OFFSET = 10_000
_MAX_SESSION_SEARCH_ENTRIES = 100_000
_SESSION_ROOT_SEARCH_TIMEOUT_SECONDS = 60
_max_read_chars_cached: int | None = None


_SESSION_ROOT_SEARCH_WORKER = r"""
import json
import os
import re
import sys
from collections import deque

root, pattern, mode, raw_context, raw_offset, raw_limit = sys.argv[1:]
context = int(raw_context)
offset = int(raw_offset)
limit = int(raw_limit)


def send(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


try:
    expression = re.compile(pattern)
    total = 0
    matching_files = 0
    truncated = False
    stop_all = False

    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            absolute = os.path.join(directory, filename)
            relative = os.path.relpath(absolute, root).replace(os.sep, "/")

            if mode == "count":
                count = 0
                with open(absolute, "r", encoding="utf-8-sig", errors="replace") as stream:
                    for line in stream:
                        if expression.search(line.rstrip("\r\n")):
                            count += 1
                if count:
                    if offset <= matching_files < offset + limit:
                        send({"type": "count", "path": relative, "count": count})
                    matching_files += 1
                    total += count
                continue

            if mode == "files_only":
                found = False
                with open(absolute, "r", encoding="utf-8-sig", errors="replace") as stream:
                    for line in stream:
                        if expression.search(line.rstrip("\r\n")):
                            found = True
                            break
                if found:
                    if offset <= matching_files < offset + limit:
                        send({"type": "file", "path": relative})
                    matching_files += 1
                continue

            before = deque(maxlen=context)
            after_until = 0
            last_emitted = 0

            def emit_row(line_number, content):
                nonlocal_total[0] += 1
                ordinal = nonlocal_total[0]
                if ordinal <= offset:
                    return False
                if ordinal <= offset + limit:
                    send({
                        "type": "match",
                        "path": relative,
                        "line": line_number,
                        "content": content[:500],
                    })
                    return False
                return True

            nonlocal_total = [total]
            with open(absolute, "r", encoding="utf-8-sig", errors="replace") as stream:
                for line_number, line in enumerate(stream, start=1):
                    content = line.rstrip("\r\n")
                    matched = bool(expression.search(content))
                    if matched:
                        for prior_number, prior_content in before:
                            if prior_number > last_emitted:
                                if emit_row(prior_number, prior_content):
                                    truncated = True
                                    stop_all = True
                                    break
                                last_emitted = prior_number
                        if stop_all:
                            break
                        if line_number > last_emitted:
                            if emit_row(line_number, content):
                                truncated = True
                                stop_all = True
                                break
                            last_emitted = line_number
                        after_until = max(after_until, line_number + context)
                    elif line_number <= after_until and line_number > last_emitted:
                        if emit_row(line_number, content):
                            truncated = True
                            stop_all = True
                            break
                        last_emitted = line_number
                    before.append((line_number, content[:500]))
            total = nonlocal_total[0]
            if stop_all:
                break
        if stop_all:
            break

    if mode in {"count", "files_only"}:
        truncated = matching_files > offset + limit
    send({
        "type": "summary",
        "total_count": total if mode != "files_only" else matching_files,
        "matching_files": matching_files,
        "truncated": truncated,
    })
except Exception as exc:
    send({"type": "summary", "error": str(exc)})
"""


def _get_max_read_chars() -> int:
    """Return the configured max characters per file read.

    Reads ``file_read_max_chars`` from config.yaml on first call, caches
    the result for the lifetime of the process.  Falls back to the
    built-in default if the config is missing or invalid.
    """
    global _max_read_chars_cached
    if _max_read_chars_cached is not None:
        return _max_read_chars_cached
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        val = cfg.get("file_read_max_chars")
        if isinstance(val, (int, float)) and val > 0:
            _max_read_chars_cached = int(val)
            return _max_read_chars_cached
    except Exception:
        pass
    _max_read_chars_cached = _DEFAULT_MAX_READ_CHARS
    return _max_read_chars_cached


def _truncate_to_char_budget(content: str, max_chars: int) -> tuple[str, int, bool]:
    """Trim line-numbered ``read_file`` content to fit a char budget.

    Ported in spirit from nearai/ironclaw#5029 (dual line/byte cap on
    ``read_file``). Where hermes previously hard-rejected an oversized read
    (forcing the model to guess a smaller ``limit`` and burn a round-trip
    returning nothing), this trims the content to the last *complete line*
    that fits within ``max_chars`` and reports how many lines were kept so
    the caller can offer a ``next_offset`` continuation.

    ``content`` is the gutter-rendered text (``LINE_NUM|CONTENT`` joined by
    ``\\n``). Individual lines are already clamped to ``get_max_line_length()``
    upstream, so a single line never blows the whole budget on its own; the
    overflow this handles is the *accumulation* of many lines under the
    line-count limit (logs, wide CSV rows, minified data).

    Returns ``(kept_text, lines_kept, truncated)``. When ``content`` already
    fits, returns it unchanged with ``truncated=False``. If not even the
    first line fits, that single line is clamped on a code-point boundary
    (Python ``str`` slicing never splits a code point) so the read never
    returns empty and the cursor can still advance.
    """
    if len(content) <= max_chars:
        return content, (content.count("\n") + 1 if content else 0), False

    lines = content.split("\n")
    kept: list[str] = []
    running = 0
    for line in lines:
        # +1 for the "\n" that rejoins this line to the previous one.
        addition = len(line) + (1 if kept else 0)
        if running + addition > max_chars:
            break
        kept.append(line)
        running += addition

    if not kept:
        # First line alone exceeds the budget. Clamp on a code-point
        # boundary rather than emitting nothing.
        kept.append(lines[0][:max_chars])

    return "\n".join(kept), len(kept), True


# If the total file size exceeds this AND the caller didn't specify a narrow
# range (limit <= 200), we include a hint encouraging targeted reads.
_LARGE_FILE_HINT_BYTES = 512_000  # 512 KB

# ---------------------------------------------------------------------------
# Device path blocklist — reading these hangs the process (infinite output
# or blocking on input).  Checked by path only (no I/O).
# ---------------------------------------------------------------------------
_BLOCKED_DEVICE_PATHS = frozenset({
    # Infinite output — never reach EOF
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    # Blocks waiting for input
    "/dev/stdin", "/dev/tty", "/dev/console",
    # Nonsensical to read
    "/dev/stdout", "/dev/stderr",
    # fd aliases
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})


def _resolve_path(filepath: str, task_id: str = "default") -> Path | PurePosixPath:
    """Resolve a path relative to TERMINAL_CWD (the worktree base directory)
    instead of the main repository root.
    """
    return _resolve_path_for_task(filepath, task_id)


# Sentinel ``TERMINAL_CWD`` values that mean "not configured", NOT a literal
# directory to resolve against. A stale config / .env commonly leaves the
# literal "." here; "auto"/"cwd" are setup-wizard placeholders. Treating any of
# these as a real relative base silently anchors edits to the agent PROCESS cwd
# (e.g. the main repo while a worktree session is active), routing writes to the
# wrong checkout. The gateway sanitizes the same set at import time
# (gateway/run.py); the file/terminal-tool layer must do likewise so CLI
# sessions get the same protection. See references/worktree-cwd-discipline.md.
_TERMINAL_CWD_SENTINELS = frozenset({"", ".", "./", "auto", "cwd"})
_CONTAINER_PATH_BACKENDS_FALLBACK = frozenset({"docker", "singularity", "modal", "daytona"})


class SessionRootPathError(ValueError):
    """Raised when a file-tool request cannot stay inside its session root."""


class _SessionRootWalkBudget:
    """Bound lazy Session Root traversal before metadata or file reads."""

    __slots__ = ("exhausted", "max_entries", "visited")

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self.visited = 0
        self.exhausted = False

    def consume(self) -> bool:
        if self.visited >= self.max_entries:
            self.exhausted = True
            return False
        self.visited += 1
        return True


def _terminal_env_type_for_task(task_id: str = "default") -> str:
    """Best-effort terminal backend type for path-resolution decisions."""
    try:
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            _get_env_config,
            _resolve_container_task_id,
        )

        try:
            container_key = _resolve_container_task_id(task_id)
        except Exception:
            container_key = task_id
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(task_id)
        if env is not None:
            name = env.__class__.__name__.lower()
            if "local" in name:
                return "local"
            if "ssh" in name:
                return "ssh"
            if "docker" in name:
                return "docker"
            if "singularity" in name:
                return "singularity"
            if "modal" in name:
                return "modal"
            if "daytona" in name:
                return "daytona"
        cfg = _get_env_config()
        return str(cfg.get("env_type") or os.getenv("TERMINAL_ENV") or "local").lower()
    except Exception:
        return str(os.getenv("TERMINAL_ENV") or "local").lower()


def _uses_container_paths(task_id: str = "default") -> bool:
    try:
        from tools.terminal_tool import _CONTAINER_BACKENDS
        container_backends = _CONTAINER_BACKENDS
    except Exception:
        container_backends = _CONTAINER_PATH_BACKENDS_FALLBACK
    return _terminal_env_type_for_task(task_id) in container_backends


def _normalize_without_host_deref(path: str | Path | PurePosixPath) -> PurePosixPath:
    """Normalize path syntax without following host symlinks.

    Container backends use paths that are meaningful inside the sandbox. Calling
    ``Path.resolve()`` on the host can dereference a host-side symlink such as
    ``/workspace`` and rewrite the path before Docker sees it.
    """
    return PurePosixPath(posixpath.normpath(str(path)))


def _sentinel_free_abs_cwd(raw: str | None) -> str | None:
    """Normalize a cwd candidate to an absolute, sentinel-free anchor.

    Returns the expanded path only when *raw* is non-empty, not a sentinel (see
    ``_TERMINAL_CWD_SENTINELS``), and absolute. A relative anchor is meaningless
    without knowing which cwd it is relative to — exactly the ambiguity that
    misroutes worktree edits — so relative/sentinel/empty values yield ``None``.
    """
    raw = str(raw or "").strip()
    if raw.lower() in _TERMINAL_CWD_SENTINELS:
        return None
    expanded = _expand_tilde(raw)
    if not os.path.isabs(expanded):
        return None
    return expanded


def _configured_terminal_cwd() -> str | None:
    """Return ``$TERMINAL_CWD`` only when it names a real directory anchor.

    Sentinel values (see ``_TERMINAL_CWD_SENTINELS``) and relative paths are
    rejected — a relative anchor is meaningless without knowing which cwd it is
    relative to, which is exactly the ambiguity that misroutes worktree edits.
    Only an absolute, sentinel-free value is honored.
    """
    return _sentinel_free_abs_cwd(os.environ.get("TERMINAL_CWD"))


def _registered_task_cwd_override(task_id: str = "default") -> str | None:
    """Return a registered cwd override for the raw task id, when available.

    ``terminal_tool`` intentionally collapses CWD-only task overrides to the
    shared ``"default"`` environment so TUI/dashboard/ACP sessions do not spin
    up isolated sandboxes just because they have different workspaces. The cwd
    value itself is still keyed by the raw session/task id, so file tools must
    read that raw override before falling back to the collapsed container key.
    """
    try:
        from tools.terminal_tool import resolve_task_overrides

        overrides = resolve_task_overrides(task_id)
    except Exception:
        return None

    return _sentinel_free_abs_cwd(overrides.get("cwd"))


def _authoritative_workspace_root(task_id: str = "default") -> str | None:
    """Best-effort absolute workspace root for divergence checks.

    Resolution:

      1. The session's own cwd RECORD (``terminal_tool.get_session_cwd``) —
         written on every completed terminal command and seeded by workspace
         registration, keyed by the raw session id. Because the record is
         per-session, one session's ``cd`` can never leak into another
         session's resolution.
      2. A registered task/session cwd override (TUI/Desktop/ACP sessions
         register a raw-keyed cwd before any tool runs). Normally already
         mirrored into the record at registration; kept as a direct fallback
         so a cleared/never-written record still resolves the workspace.
      3. A sentinel-free absolute ``$TERMINAL_CWD`` (the worktree path set by
         ``cli.py``/``main.py`` for ``-w`` sessions).

    Returns ``None`` only when there is genuinely no reliable anchor, in which
    case callers fall back to the process cwd.
    """
    try:
        from tools.terminal_tool import get_session_cwd

        recorded = get_session_cwd(task_id)
    except Exception:
        recorded = None
    if recorded:
        return recorded
    registered = _registered_task_cwd_override(task_id)
    if registered:
        return registered
    return _configured_terminal_cwd()


def _resolve_base_dir(
    task_id: str = "default",
    *,
    container_paths: bool | None = None,
) -> Path | PurePosixPath:
    """Return the ABSOLUTE base directory for resolving relative paths.

    Resolution order:
      1. The task's live terminal cwd (the directory the agent is actually
         working in — e.g. a git worktree). Authoritative when known.
      2. A registered task/session cwd override (TUI/Desktop/ACP sessions
         register a raw-keyed workspace cwd before any terminal command runs).
      3. A sentinel-free, absolute ``$TERMINAL_CWD`` (the worktree path set by
         ``cli.py``/``main.py`` for ``-w`` sessions). Used even before any
         terminal command has populated the live cwd registry.
      4. The process cwd.

    The returned base is ALWAYS absolute. This is the core invariant that
    prevents the worktree-cwd divergence bug: a relative or sentinel
    ``TERMINAL_CWD`` (commonly the literal ``"."`` from a stale config) is
    meaningless as a resolution anchor — left to ``Path.resolve()`` it silently
    resolves against whatever the agent PROCESS cwd happens to be (e.g. the main
    repo while the terminal is in a worktree), routing edits to the wrong
    checkout. We therefore reject sentinel/relative ``TERMINAL_CWD`` values
    outright (rather than anchoring them to the process cwd) and fall through to
    the process cwd only as a last resort, deterministically.
    """
    root = _authoritative_workspace_root(task_id)
    if container_paths is None:
        container_paths = _uses_container_paths(task_id)
    if root:
        base_text = _expand_tilde(root)
    else:
        base_text = os.getcwd()
    if container_paths:
        if not posixpath.isabs(base_text):
            base_text = posixpath.join(os.getcwd(), base_text)
        return _normalize_without_host_deref(base_text)
    # Git Bash ``pwd -P`` reports ``/c/Users/...``; translate before Path so
    # relative file-tool paths don't anchor under a nonexistent ``\\c\\Users``.
    from tools.environments.local import _msys_to_windows_path

    base_text = _msys_to_windows_path(base_text)
    if sys.platform == "win32":
        import ntpath

        if not ntpath.isabs(base_text):
            base_text = ntpath.join(os.getcwd(), base_text)
        return Path(ntpath.normpath(base_text))
    base = Path(base_text)
    if not base.is_absolute():
        # Last-resort anchoring: a live cwd should already be absolute, but if a
        # terminal backend ever reports a relative cwd, anchor it to the process
        # cwd once, here, so the result no longer depends on cwd at resolve().
        base = Path(os.getcwd()) / base
    return base.resolve()


def _session_file_root(task_id: str = "default") -> Path | None:
    """Return the validated request-scoped file root, if one is active.

    Session-root enforcement currently targets the local file backend used by
    the API runtime.  Failing closed on remote/container path namespaces avoids
    pretending that host ``Path.resolve`` can validate paths inside a separate
    filesystem.
    """
    try:
        from agent.runtime_cwd import resolve_session_file_root

        root = resolve_session_file_root()
    except Exception as exc:
        raise SessionRootPathError(
            "Session file root context is unavailable"
        ) from exc
    if root is None:
        return None
    if _uses_container_paths(task_id):
        raise SessionRootPathError(
            "Session file root enforcement is unavailable for this file backend"
        )
    try:
        resolved = root.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise SessionRootPathError("Configured session file root is not available") from exc
    if not resolved.is_dir():
        raise SessionRootPathError("Configured session file root is not a directory")
    return resolved


def _ensure_under_session_root(path: Path, root: Path) -> Path:
    """Resolve symlinks and reject any path outside ``root``."""
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(root)
    except ValueError as exc:
        raise SessionRootPathError("Path escapes current session file root") from exc
    except OSError as exc:
        raise SessionRootPathError(
            "Path cannot be resolved within current session file root"
        ) from exc
    return resolved


def _session_root_hardlink_error(path: Path, task_id: str = "default") -> str | None:
    """Reject regular files whose other hardlink may live outside the root."""
    if _session_file_root(task_id) is None:
        return None
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return None
    except OSError:
        return "Path cannot be verified within current session file root"
    if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
        return "Path has multiple filesystem links and is unsafe in this session file root"
    return None


def _session_root_tree_error(path: Path, task_id: str = "default") -> str | None:
    """Reject search trees containing symlinks that resolve outside the root."""
    root = _session_file_root(task_id)
    if root is None:
        return None
    hardlink_error = _session_root_hardlink_error(path, task_id)
    if hardlink_error:
        return hardlink_error
    if not path.is_dir():
        return None
    for current, dirnames, filenames in os.walk(path, followlinks=False):
        current_path = Path(current)
        for name in (*dirnames, *filenames):
            candidate = current_path / name
            if candidate.is_symlink():
                try:
                    _ensure_under_session_root(candidate, root)
                except SessionRootPathError:
                    return "Search path contains a symlink that escapes current session file root"
                continue
            hardlink_error = _session_root_hardlink_error(candidate, task_id)
            if hardlink_error:
                return hardlink_error
    return None


def _resolve_path_for_task(filepath: str, task_id: str = "default") -> Path | PurePosixPath:
    """Resolve *filepath* against the task's absolute base directory.

    See :func:`_resolve_base_dir` for how the base is chosen. Absolute input
    paths are returned resolved-but-unanchored.

    On native Windows, Git Bash / MSYS drive paths (``/c/Users/...``) are
    translated to ``C:\\Users\\...`` before resolution so file tools don't
    treat them as relative ``\\c\\Users\\...`` under the process cwd.
    """
    root = _session_file_root(task_id)
    if root is not None:
        expanded = Path(_expand_tilde(filepath))
        if ".." in expanded.parts:
            raise SessionRootPathError(
                "Path contains '..' traversal in current session file root"
            )
        candidate = expanded if expanded.is_absolute() else root / expanded
        return _ensure_under_session_root(candidate, root)

    container_paths = _uses_container_paths(task_id)
    if container_paths:
        expanded = _expand_tilde(filepath)
        if posixpath.isabs(expanded):
            return _normalize_without_host_deref(expanded)
        resolved = _resolve_base_dir(task_id, container_paths=True) / expanded
        return _normalize_without_host_deref(resolved)

    # Host paths only — never rewrite Linux paths inside a container/WSL env.
    from tools.environments.local import _msys_to_windows_path

    expanded = _expand_tilde(_msys_to_windows_path(filepath))
    if sys.platform == "win32":
        import ntpath

        if ntpath.isabs(expanded):
            return Path(ntpath.normpath(expanded))
        joined = ntpath.join(str(_resolve_base_dir(task_id, container_paths=False)), expanded)
        return Path(ntpath.normpath(joined))

    p = Path(expanded)
    if p.is_absolute():
        return p.resolve()
    resolved = _resolve_base_dir(task_id, container_paths=False) / p
    return resolved.resolve()


class _SessionRootFileOperations:
    """Local file operations anchored to directory descriptors.

    ``Path.resolve`` is useful validation but cannot close the race between
    validation and a later shell open.  This adapter walks every component
    from ``/`` with ``O_NOFOLLOW`` and performs reads/writes through pinned
    directory descriptors.  A concurrent rename of an ancestor therefore
    cannot redirect an operation outside the request-scoped root.
    """

    def __init__(self, root: Path, delegate: ShellFileOperations):
        self.root = root
        self.delegate = delegate

    @staticmethod
    def _dir_flags() -> int:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        return flags

    @staticmethod
    def _file_flags(flags: int) -> int:
        return flags | getattr(os, "O_NOFOLLOW", 0)

    def _relative_parts(self, path: str) -> tuple[str, ...]:
        candidate = Path(_expand_tilde(path))
        if ".." in candidate.parts:
            raise SessionRootPathError(
                "Path contains '..' traversal in current session file root"
            )
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(self.root)
            except ValueError as exc:
                raise SessionRootPathError(
                    "Path escapes current session file root"
                ) from exc
        parts = tuple(part for part in candidate.parts if part not in {"", "."})
        if any(part in {"..", os.sep} for part in parts):
            raise SessionRootPathError(
                "Path escapes current session file root"
            )
        return parts

    def _open_root(self) -> int:
        """Open the absolute root without following any path component."""
        if os.name != "posix":
            raise SessionRootPathError(
                "Secure session file root operations require a POSIX runtime"
            )
        current = os.open(os.sep, self._dir_flags())
        try:
            for part in self.root.parts[1:]:
                next_fd = os.open(part, self._dir_flags(), dir_fd=current)
                os.close(current)
                current = next_fd
            return current
        except Exception:
            os.close(current)
            raise

    def _open_parent(
        self,
        path: str,
        *,
        create: bool = False,
    ) -> tuple[int, str, bool]:
        parts = self._relative_parts(path)
        if not parts:
            raise SessionRootPathError("Session file root itself is not a file")
        current = self._open_root()
        created = False
        try:
            for part in parts[:-1]:
                try:
                    next_fd = os.open(part, self._dir_flags(), dir_fd=current)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(part, mode=0o755, dir_fd=current)
                    created = True
                    next_fd = os.open(part, self._dir_flags(), dir_fd=current)
                os.close(current)
                current = next_fd
            return current, parts[-1], created
        except Exception:
            os.close(current)
            raise

    @staticmethod
    def _reject_unsafe_metadata(metadata: os.stat_result) -> None:
        if stat.S_ISLNK(metadata.st_mode):
            raise SessionRootPathError(
                "Path is a symlink in current session file root"
            )
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
            raise SessionRootPathError(
                "Path has multiple filesystem links and is unsafe in this "
                "session file root"
            )

    def _read_bytes(
        self,
        path: str,
        *,
        max_bytes: int = _MAX_SESSION_BUFFERED_FILE_BYTES,
    ) -> tuple[bytes, os.stat_result]:
        parent_fd, name, _ = self._open_parent(path)
        try:
            fd = os.open(
                name,
                self._file_flags(os.O_RDONLY),
                dir_fd=parent_fd,
            )
            try:
                metadata = os.fstat(fd)
                self._reject_unsafe_metadata(metadata)
                if not stat.S_ISREG(metadata.st_mode):
                    raise SessionRootPathError("Path is not a regular file")
                if metadata.st_size > max_bytes:
                    raise SessionRootPathError(
                        f"File exceeds the {max_bytes}-byte secure buffer limit"
                    )
                chunks = []
                bytes_read = 0
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    if bytes_read > max_bytes:
                        raise SessionRootPathError(
                            f"File exceeds the {max_bytes}-byte secure buffer limit"
                        )
                    chunks.append(chunk)
                return b"".join(chunks), metadata
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)

    @staticmethod
    def _decode(data: bytes) -> str:
        return data.decode("utf-8-sig", errors="replace")

    def read_file_raw(self, path: str) -> ReadResult:
        try:
            data, _metadata = self._read_bytes(path)
            text = self._decode(data)
            return ReadResult(
                content=text,
                total_lines=len(text.splitlines()),
                file_size=len(data),
            )
        except FileNotFoundError:
            return ReadResult(error=f"File not found: {path}")
        except Exception as exc:
            return ReadResult(error=str(exc))

    def read_file(self, path: str, offset: int = 1, limit: int = 500) -> ReadResult:
        offset, limit = normalize_read_pagination(offset, limit)
        parent_fd = file_fd = None
        try:
            parent_fd, name, _ = self._open_parent(path)
            file_fd = os.open(
                name,
                self._file_flags(os.O_RDONLY),
                dir_fd=parent_fd,
            )
            metadata = os.fstat(file_fd)
            self._reject_unsafe_metadata(metadata)
            if not stat.S_ISREG(metadata.st_mode):
                raise SessionRootPathError("Path is not a regular file")
            page = []
            total_lines = 0
            clamped_line = False
            from tools.tool_output_limits import get_max_line_length

            max_total_chars = _get_max_read_chars()
            max_line_chars = get_max_line_length()
            remaining_chars = max_total_chars
            with os.fdopen(
                file_fd,
                "r",
                encoding="utf-8-sig",
                errors="replace",
            ) as stream:
                file_fd = None
                while True:
                    first_part = stream.readline(max_line_chars + 1)
                    if first_part == "":
                        break
                    total_lines += 1
                    in_page = offset <= total_lines < offset + limit
                    if in_page:
                        prefix_budget = len(str(total_lines)) + 2
                        available = max(0, remaining_chars - prefix_budget)
                        captured = first_part[
                            : min(max_line_chars, available)
                        ].rstrip("\r\n")
                        if captured or available > 0:
                            page.append(captured)
                            remaining_chars -= len(captured) + prefix_budget
                        if (
                            len(first_part.rstrip("\r\n")) > len(captured)
                            or available == 0
                        ):
                            clamped_line = True
                    line_complete = first_part.endswith("\n")
                    if len(first_part) > max_line_chars and not line_complete:
                        clamped_line = clamped_line or in_page
                    while not line_complete:
                        next_part = stream.readline(max_line_chars + 1)
                        if next_part == "":
                            break
                        if in_page:
                            # A bounded second chunk means this physical line
                            # exceeded its line budget. Keep only the first
                            # bounded segment and drain the rest incrementally.
                            clamped_line = True
                        line_complete = next_part.endswith("\n")
            return ReadResult(
                content=self.delegate._add_line_numbers("\n".join(page), offset)
                if page
                else "",
                total_lines=total_lines,
                file_size=metadata.st_size,
                truncated=clamped_line or offset - 1 + limit < total_lines,
                hint=(
                    "One or more requested lines exceeded the secure read "
                    "budget and were clamped."
                    if clamped_line
                    else None
                ),
            )
        except FileNotFoundError:
            return ReadResult(error=f"File not found: {path}")
        except Exception as exc:
            return ReadResult(error=str(exc))
        finally:
            if file_fd is not None:
                os.close(file_fd)
            if parent_fd is not None:
                os.close(parent_fd)

    def extract_document(self, path: str) -> tuple[str, int]:
        """Extract a document from a descriptor-pinned temporary snapshot."""
        from tools.read_extract import extract_document_text

        data, metadata = self._read_bytes(path)
        suffix = Path(path).suffix.lower()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="hermes-session-document-",
                suffix=suffix,
                delete=False,
            ) as snapshot:
                snapshot.write(data)
                snapshot.flush()
                temp_path = snapshot.name
            return extract_document_text(temp_path), metadata.st_size
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _check_lint(self, path: str, content: str | None = None) -> LintResult:
        from tools.file_operations import LINTERS, LINTERS_INPROC

        extension = Path(path).suffix.lower()
        linter = LINTERS_INPROC.get(extension)
        if linter is not None and content is not None:
            ok, error = linter(content)
            if error == "__SKIP__":
                return LintResult(
                    skipped=True,
                    message=f"No linter available for {extension}",
                )
            return LintResult(success=ok, output="" if ok else error)
        if content is None or extension not in LINTERS:
            return LintResult(
                skipped=True,
                message=f"No linter available for {extension}",
            )

        # Shell linters require a pathname. Run them against an owner-only
        # snapshot rather than the mutable Session Root path, so syntax checks
        # retain their v0.19 behavior without reopening a path race.
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="hermes-session-lint-",
                suffix=extension,
                mode="w",
                encoding="utf-8",
                delete=False,
            ) as snapshot:
                snapshot.write(content)
                snapshot.flush()
                temp_path = snapshot.name
            result = self.delegate._check_lint(temp_path, content=content)
            if result.output:
                result.output = result.output.replace(temp_path, path)
            if result.message:
                result.message = result.message.replace(temp_path, path)
            return result
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def write_file(self, path: str, content: str) -> WriteResult:
        from tools.file_operations import (
            LINTERS_INPROC,
            _FAIL_CLOSED_INPROC_EXTS,
            _UTF8_BOM,
            _has_bom,
            _normalize_line_endings,
        )

        extension = Path(path).suffix.lower()
        linter = (
            LINTERS_INPROC.get(extension)
            if extension in _FAIL_CLOSED_INPROC_EXTS
            else None
        )
        if linter is not None:
            ok, error = linter(content)
            if not ok and error != "__SKIP__":
                return WriteResult(
                    error=(
                        f"Refusing to write '{path}': candidate content fails "
                        f"{extension} syntax validation ({error})."
                    )
                )

        parent_fd = None
        temp_name = None
        try:
            parent_fd, name, dirs_created = self._open_parent(path, create=True)
            existing_mode = None
            existing_text = None
            try:
                existing_fd = os.open(
                    name,
                    self._file_flags(os.O_RDONLY),
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                existing_fd = None
            if existing_fd is not None:
                try:
                    metadata = os.fstat(existing_fd)
                    self._reject_unsafe_metadata(metadata)
                    if not stat.S_ISREG(metadata.st_mode):
                        return WriteResult(error=f"Path is not a regular file: {path}")
                    if metadata.st_size > _MAX_SESSION_BUFFERED_FILE_BYTES:
                        return WriteResult(
                            error=(
                                "Existing file exceeds the secure buffer limit; "
                                "refusing a whole-file replacement"
                            )
                        )
                    existing_mode = stat.S_IMODE(metadata.st_mode)
                    existing_data = bytearray()
                    while True:
                        chunk = os.read(existing_fd, 1024 * 1024)
                        if not chunk:
                            break
                        existing_data.extend(chunk)
                    existing_text = bytes(existing_data).decode(
                        "utf-8",
                        errors="replace",
                    )
                finally:
                    os.close(existing_fd)

            if existing_text is not None:
                if "\r\n" in existing_text and "\n" in content:
                    content = _normalize_line_endings(content, "\r\n")
                if existing_text.startswith(_UTF8_BOM) and not _has_bom(content):
                    content = _UTF8_BOM + content

            payload = content.encode("utf-8")
            temp_name = f".hermes-session-{uuid.uuid4().hex}.tmp"
            temp_fd = os.open(
                temp_name,
                self._file_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
                existing_mode or 0o600,
                dir_fd=parent_fd,
            )
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(temp_fd, view)
                    view = view[written:]
                if existing_mode is not None:
                    os.fchmod(temp_fd, existing_mode)
                os.fsync(temp_fd)
            finally:
                os.close(temp_fd)

            os.replace(
                temp_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temp_name = None
            lint = self._check_lint(path, content=content)
            return WriteResult(
                bytes_written=len(payload),
                dirs_created=dirs_created,
                lint=lint.to_dict(),
            )
        except Exception as exc:
            return WriteResult(error=str(exc))
        finally:
            if parent_fd is not None:
                if temp_name is not None:
                    try:
                        os.unlink(temp_name, dir_fd=parent_fd)
                    except OSError:
                        pass
                os.close(parent_fd)

    def patch_replace(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> PatchResult:
        import difflib

        from tools.fuzzy_match import fuzzy_find_and_replace

        before_result = self.read_file_raw(path)
        if before_result.error:
            return PatchResult(error=before_result.error)
        after, count, _strategy, error = fuzzy_find_and_replace(
            before_result.content,
            old_string,
            new_string,
            replace_all,
        )
        if error or count == 0:
            return PatchResult(error=error or f"Could not find match in {path}")
        write_result = self.write_file(path, after)
        if write_result.error:
            return PatchResult(error=write_result.error)
        diff = "".join(
            difflib.unified_diff(
                before_result.content.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        return PatchResult(
            success=True,
            diff=diff,
            files_modified=[path],
            lint=write_result.lint,
        )

    def patch_v4a(self, patch_content: str) -> PatchResult:
        from tools.patch_parser import apply_v4a_operations, parse_v4a_patch

        operations, error = parse_v4a_patch(patch_content)
        if error:
            return PatchResult(error=f"Failed to parse patch: {error}")
        return apply_v4a_operations(operations, self)

    def delete_file(self, path: str) -> WriteResult:
        parent_fd = None
        try:
            parent_fd, name, _ = self._open_parent(path)
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            self._reject_unsafe_metadata(metadata)
            if not stat.S_ISREG(metadata.st_mode):
                return WriteResult(error=f"Path is not a regular file: {path}")
            os.unlink(name, dir_fd=parent_fd)
            return WriteResult()
        except Exception as exc:
            return WriteResult(error=str(exc))
        finally:
            if parent_fd is not None:
                os.close(parent_fd)

    def move_file(self, src: str, dst: str) -> WriteResult:
        src_fd = dst_fd = None
        moved = False
        try:
            src_fd, src_name, _ = self._open_parent(src)
            dst_fd, dst_name, _ = self._open_parent(dst, create=True)
            pinned_fd = os.open(
                src_name,
                self._file_flags(os.O_RDONLY),
                dir_fd=src_fd,
            )
            try:
                pinned = os.fstat(pinned_fd)
                self._reject_unsafe_metadata(pinned)
                if not stat.S_ISREG(pinned.st_mode):
                    return WriteResult(
                        error=f"Move source is not a regular file: {src}"
                    )
                try:
                    os.stat(dst_name, dir_fd=dst_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    return WriteResult(
                        error=f"Move destination already exists: {dst}"
                    )

                # Rename is atomic and does not follow a swapped symlink.
                # Re-verify the destination against the descriptor-pinned
                # source inode before declaring success.
                os.rename(
                    src_name,
                    dst_name,
                    src_dir_fd=src_fd,
                    dst_dir_fd=dst_fd,
                )
                moved = True
                destination = os.stat(
                    dst_name,
                    dir_fd=dst_fd,
                    follow_symlinks=False,
                )
                self._reject_unsafe_metadata(destination)
                if (destination.st_dev, destination.st_ino) != (
                    pinned.st_dev,
                    pinned.st_ino,
                ):
                    raise SessionRootPathError(
                        "Move source changed during the operation"
                    )
            finally:
                os.close(pinned_fd)
            moved = False
            return WriteResult()
        except Exception as exc:
            return WriteResult(error=str(exc))
        finally:
            if moved and dst_fd is not None:
                try:
                    os.unlink(dst_name, dir_fd=dst_fd)
                except OSError:
                    pass
            if src_fd is not None:
                os.close(src_fd)
            if dst_fd is not None:
                os.close(dst_fd)

    @staticmethod
    def _matches_file_glob(
        relative: tuple[str, ...],
        file_glob: str | None,
    ) -> bool:
        if not file_glob:
            return True
        return (
            fnmatch.fnmatch(relative[-1], file_glob)
            or fnmatch.fnmatch("/".join(relative), file_glob)
        )

    def _walk_files(
        self,
        path: str,
        *,
        include_data: bool,
        file_glob: str | None = None,
        walk_budget: _SessionRootWalkBudget | None = None,
    ):
        parts = self._relative_parts(path)
        current = None
        try:
            if parts:
                if walk_budget is not None and not walk_budget.consume():
                    return
                parent_fd, name, _ = self._open_parent(path)
                try:
                    metadata = os.stat(
                        name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    self._reject_unsafe_metadata(metadata)
                    if stat.S_ISREG(metadata.st_mode):
                        if not self._matches_file_glob(parts, file_glob):
                            return
                        data = None
                        pinned = metadata
                        if (
                            include_data
                            and metadata.st_size
                            <= _MAX_SESSION_SEARCH_FILE_BYTES
                        ):
                            data, pinned = self._read_bytes(
                                path,
                                max_bytes=_MAX_SESSION_SEARCH_FILE_BYTES,
                            )
                        yield parts, data, pinned
                        return
                    if not stat.S_ISDIR(metadata.st_mode):
                        raise SessionRootPathError(
                            "Search path is not a regular file or directory"
                        )
                    current = os.open(
                        name,
                        self._dir_flags(),
                        dir_fd=parent_fd,
                    )
                finally:
                    os.close(parent_fd)
            else:
                current = self._open_root()

            def walk(directory_fd: int, relative: tuple[str, ...]):
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        if (
                            walk_budget is not None
                            and not walk_budget.consume()
                        ):
                            return
                        name = entry.name
                        if name.startswith("."):
                            continue
                        metadata = os.stat(
                            name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                        if stat.S_ISLNK(metadata.st_mode):
                            # Never follow links from a Session Root tree.
                            continue
                        child_relative = relative + (name,)
                        if stat.S_ISDIR(metadata.st_mode):
                            child_fd = os.open(
                                name,
                                self._dir_flags(),
                                dir_fd=directory_fd,
                            )
                            try:
                                yield from walk(child_fd, child_relative)
                            finally:
                                os.close(child_fd)
                            if (
                                walk_budget is not None
                                and walk_budget.exhausted
                            ):
                                return
                        elif stat.S_ISREG(metadata.st_mode):
                            if not self._matches_file_glob(
                                child_relative,
                                file_glob,
                            ):
                                continue
                            file_fd = os.open(
                                name,
                                self._file_flags(os.O_RDONLY),
                                dir_fd=directory_fd,
                            )
                            try:
                                pinned = os.fstat(file_fd)
                                self._reject_unsafe_metadata(pinned)
                                data = None
                                if (
                                    include_data
                                    and pinned.st_size
                                    <= _MAX_SESSION_SEARCH_FILE_BYTES
                                ):
                                    chunks = []
                                    bytes_read = 0
                                    while True:
                                        chunk = os.read(file_fd, 1024 * 1024)
                                        if not chunk:
                                            break
                                        bytes_read += len(chunk)
                                        if (
                                            bytes_read
                                            > _MAX_SESSION_SEARCH_FILE_BYTES
                                        ):
                                            chunks = []
                                            break
                                        chunks.append(chunk)
                                    if chunks or pinned.st_size == 0:
                                        data = b"".join(chunks)
                                yield child_relative, data, pinned
                            finally:
                                os.close(file_fd)

            yield from walk(current, parts)
        finally:
            if current is not None:
                os.close(current)

    def _search_content_snapshot(
        self,
        pattern: str,
        path: str,
        *,
        file_glob: str | None,
        limit: int,
        offset: int,
        output_mode: str,
        context: int,
    ) -> SearchResult:
        matches: list[SearchMatch] = []
        counts: dict[str, int] = {}
        files_only: list[str] = []
        skipped_oversized = 0
        snapshot_limited = False
        walk_budget = _SessionRootWalkBudget(_MAX_SESSION_SEARCH_ENTRIES)

        with tempfile.TemporaryDirectory(
            prefix="hermes-session-search-"
        ) as snapshot_dir:
            snapshot_root = Path(snapshot_dir)
            source_metadata: dict[str, tuple[str, float]] = {}
            total_bytes = 0
            total_files = 0
            records = self._walk_files(
                path,
                include_data=True,
                file_glob=file_glob,
                walk_budget=walk_budget,
            )
            try:
                for relative, data, metadata in records:
                    if data is None:
                        skipped_oversized += 1
                        continue
                    if (
                        total_files >= _MAX_SESSION_SEARCH_FILES
                        or total_bytes + len(data)
                        > _MAX_SESSION_SEARCH_TOTAL_BYTES
                    ):
                        snapshot_limited = True
                        break
                    snapshot_path = snapshot_root.joinpath(*relative)
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    with snapshot_path.open("xb") as snapshot:
                        snapshot.write(data)
                    snapshot_path.chmod(0o600)
                    relative_key = "/".join(relative)
                    source_metadata[relative_key] = (
                        str(self.root.joinpath(*relative)),
                        metadata.st_mtime,
                    )
                    total_bytes += len(data)
                    total_files += 1
            finally:
                records.close()
            enumeration_limited = walk_budget.exhausted

            if not source_metadata:
                reasons = []
                if skipped_oversized:
                    reasons.append(
                        f"Skipped {skipped_oversized} file(s) over the "
                        f"{_MAX_SESSION_SEARCH_FILE_BYTES}-byte Session Root "
                        "search size limit"
                    )
                if snapshot_limited:
                    reasons.append("request snapshot budget reached")
                if enumeration_limited:
                    reasons.append(
                        "Session Root search enumeration budget reached "
                        f"({_MAX_SESSION_SEARCH_ENTRIES} entries)"
                    )
                reason = "; ".join(reasons) or None
                return SearchResult(
                    total_count=0,
                    truncated=bool(reason),
                    limit_reason=reason,
                )

            child_mode = (
                output_mode
                if output_mode in {"count", "files_only"}
                else "content"
            )
            child_env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
            }
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        _SESSION_ROOT_SEARCH_WORKER,
                        str(snapshot_root),
                        pattern,
                        child_mode,
                        str(context),
                        str(offset),
                        str(limit),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=_SESSION_ROOT_SEARCH_TIMEOUT_SECONDS,
                    env=child_env,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return SearchResult(
                    error=(
                        "Session Root search timed out after "
                        f"{_SESSION_ROOT_SEARCH_TIMEOUT_SECONDS} seconds"
                    )
                )
            if completed.returncode != 0:
                return SearchResult(
                    error=(
                        "Session Root search worker failed: "
                        f"{completed.stderr.strip()[:500]}"
                    )
                )

            summary = None
            try:
                for raw_line in completed.stdout.splitlines():
                    if not raw_line:
                        continue
                    event = json.loads(raw_line)
                    event_type = event.get("type")
                    if event_type == "summary":
                        summary = event
                        continue
                    relative_key = event.get("path")
                    source = source_metadata.get(relative_key)
                    if source is None:
                        continue
                    display, mtime = source
                    if event_type == "match":
                        matches.append(
                            SearchMatch(
                                path=display,
                                line_number=int(event["line"]),
                                content=str(event["content"])[:500],
                                mtime=mtime,
                            )
                        )
                    elif event_type == "count":
                        counts[display] = int(event["count"])
                    elif event_type == "file":
                        files_only.append(display)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return SearchResult(
                    error=f"Invalid Session Root search worker output: {exc}"
                )

            if summary is None:
                return SearchResult(
                    error="Session Root search worker returned no summary"
                )
            if summary.get("error"):
                return SearchResult(
                    error=f"Invalid search pattern: {summary['error']}"
                )

            reasons = []
            if skipped_oversized:
                reasons.append(
                    f"Skipped {skipped_oversized} file(s) over the "
                    f"{_MAX_SESSION_SEARCH_FILE_BYTES}-byte Session Root "
                    "search size limit"
                )
            if snapshot_limited:
                reasons.append(
                    "Session Root search request snapshot budget reached "
                    f"({_MAX_SESSION_SEARCH_FILES} files / "
                    f"{_MAX_SESSION_SEARCH_TOTAL_BYTES} bytes)"
                )
            if enumeration_limited:
                reasons.append(
                    "Session Root search enumeration budget reached "
                    f"({_MAX_SESSION_SEARCH_ENTRIES} entries)"
                )
            limit_reason = "; ".join(reasons) or None
            result_kwargs = {
                "total_count": int(summary.get("total_count", 0)),
                "truncated": bool(summary.get("truncated")) or bool(limit_reason),
                "limit_reason": limit_reason,
            }
            if child_mode == "files_only":
                return SearchResult(files=files_only, **result_kwargs)
            if child_mode == "count":
                return SearchResult(counts=counts, **result_kwargs)
            return SearchResult(matches=matches, **result_kwargs)

    def search(
        self,
        pattern: str,
        path: str = ".",
        target: str = "content",
        file_glob: str | None = None,
        limit: int = 50,
        offset: int = 0,
        output_mode: str = "content",
        context: int = 0,
    ) -> SearchResult:
        try:
            if offset > _MAX_SESSION_SEARCH_OFFSET:
                return SearchResult(
                    error=(
                        "Session Root search offset exceeds the "
                        f"{_MAX_SESSION_SEARCH_OFFSET}-result pagination limit"
                    )
                )
            limit = min(max(1, limit), _MAX_SESSION_SEARCH_RESULTS)
            context = min(max(0, context), _MAX_SESSION_SEARCH_CONTEXT)
            if target == "files":
                glob = pattern if any(ch in pattern for ch in "*?[]") else f"*{pattern}"
                keep = max(offset + limit, 1)
                newest: list[tuple[float, str]] = []
                total = 0
                walk_budget = _SessionRootWalkBudget(
                    _MAX_SESSION_SEARCH_ENTRIES
                )
                records = self._walk_files(
                    path,
                    include_data=False,
                    walk_budget=walk_budget,
                )
                try:
                    for relative, _data, metadata in records:
                        if not fnmatch.fnmatch(relative[-1], glob):
                            continue
                        total += 1
                        candidate = (
                            metadata.st_mtime,
                            str(self.root.joinpath(*relative)),
                        )
                        if len(newest) < keep:
                            heapq.heappush(newest, candidate)
                        elif candidate > newest[0]:
                            heapq.heapreplace(newest, candidate)
                finally:
                    records.close()
                enumeration_limited = walk_budget.exhausted
                files = [
                    item[1]
                    for item in sorted(newest, reverse=True)
                ]
                return SearchResult(
                    files=files[offset:offset + limit],
                    total_count=total,
                    truncated=offset + limit < total or enumeration_limited,
                    limit_reason=(
                        "Session Root file search enumeration budget reached "
                        f"({_MAX_SESSION_SEARCH_ENTRIES} entries)"
                        if enumeration_limited
                        else None
                    ),
                )

            return self._search_content_snapshot(
                pattern,
                path,
                file_glob=file_glob,
                limit=limit,
                offset=offset,
                output_mode=output_mode,
                context=context,
            )
        except Exception as exc:
            return SearchResult(error=str(exc))


def _path_resolution_warning(filepath: str, resolved: Path, task_id: str = "default") -> str | None:
    """Warn when a relative path resolved OUTSIDE the task's workspace root.

    Surfaces the worktree-cwd divergence the moment it would matter: if the
    agent passes a relative path but it resolves under a directory that is not
    the workspace root (i.e. the edit is about to land in a different checkout
    than the one the agent is working in), return a message naming the absolute
    target. ``None`` when the path is absolute, the base is unknown, or the
    resolved path is correctly under the workspace root.

    The workspace root is the live terminal cwd when known, else a registered
    task/session cwd override, else a sentinel-free absolute ``$TERMINAL_CWD``
    — so a worktree or Desktop session whose terminal registry is still empty
    (no ``cd`` run yet) is warned on the very first write.
    """
    try:
        if Path(_expand_tilde(filepath)).is_absolute():
            return None
        workspace_root = _authoritative_workspace_root(task_id)
        if not workspace_root:
            return None  # No authoritative workspace root to compare against.
        if _uses_container_paths(task_id):
            root = _normalize_without_host_deref(Path(_expand_tilde(workspace_root)))
        else:
            root = Path(_expand_tilde(workspace_root)).resolve()
        # Is `resolved` inside `root`?
        try:
            resolved.relative_to(root)
            return None  # Inside the workspace — expected.
        except ValueError:
            return (
                f"Relative path {filepath!r} resolved to {str(resolved)!r}, which is "
                f"OUTSIDE the active workspace ({str(root)!r}). The edit will land in "
                f"a different directory than the terminal's cwd. If this is not "
                f"intended (e.g. a git-worktree session writing into the main "
                f"checkout), pass an absolute path under the workspace instead."
            )
    except Exception:
        return None


def _is_blocked_device_path(path: str) -> bool:
    """Return True for concrete device/fd paths that can hang reads."""
    normalized = os.path.normpath(_expand_tilde(path))
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    # /proc/self/fd/0-2 and /proc/<pid>/fd/0-2 are Linux aliases for stdio
    if normalized.startswith("/proc/") and normalized.endswith(
        ("/fd/0", "/fd/1", "/fd/2")
    ):
        return True
    # /proc/*/environ, /proc/*/cmdline, /proc/*/maps (and the maps variants
    # smaps, smaps_rollup, numa_maps) can leak secrets, command-line args, and
    # memory layout (ASLR bypass) from the host process (issue #4427).
    # /proc/*/mem exposes raw process memory; block it as defense-in-depth even
    # though it requires address knowledge to exploit usefully.
    # /proc/*/auxv leaks AT_RANDOM (stack canary seed) plus AT_BASE/AT_PHDR
    # load addresses — an ASLR oracle on par with maps. /proc/*/pagemap exposes
    # virtual->physical translation. Both are blocked alongside the maps family.
    # endswith matches both /proc/<pid>/X and /proc/<pid>/task/<tid>/X.
    if normalized.startswith("/proc/") and normalized.endswith(
        (
            "/environ",
            "/cmdline",
            "/maps",
            "/smaps",
            "/smaps_rollup",
            "/numa_maps",
            "/mem",
            "/auxv",
            "/pagemap",
        )
    ):
        return True
    return False


def _is_blocked_device(filepath: str, base_dir: str | Path | None = None) -> bool:
    """Return True if the path would hang the process (infinite output or blocking input).

    Check the literal path first so aliases like /dev/stdin are caught before
    they resolve to terminal-specific paths. Then check each symlink hop before
    the final resolved path so aliases to devices cannot bypass the guard.
    """
    expanded = _expand_tilde(filepath)
    if base_dir is not None and not os.path.isabs(expanded):
        expanded = os.path.join(os.fspath(base_dir), expanded)
    normalized = os.path.normpath(expanded)
    if _is_blocked_device_path(normalized):
        return True

    seen: set[str] = set()
    current = normalized
    for _ in range(20):
        try:
            target = os.readlink(current)
        except OSError:
            break
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(current), target)
        target = os.path.normpath(target)
        if _is_blocked_device_path(target):
            return True
        if target in seen:
            break
        seen.add(target)
        current = target

    try:
        resolved = os.path.normpath(os.path.realpath(normalized))
    except (OSError, ValueError):
        return False
    if _is_blocked_device_path(resolved):
        return True
    return False


def _search_result_read_block_error(path: str, task_id: str = "default") -> str | None:
    """Return the read-safety error for a search result path.

    Search backends may return paths relative to the task cwd, while
    ``get_read_block_error`` expects an already-resolved path when the task cwd
    can differ from the Python process cwd. Mirror ``read_file_tool``'s path
    resolution before applying the shared read guard.
    """
    try:
        resolved = _resolve_path_for_task(path, task_id)
    except (OSError, ValueError, RuntimeError):
        return get_read_block_error(path)
    return get_read_block_error(str(resolved))


def _filter_read_blocked_search_results(result, task_id: str = "default") -> int:
    """Remove credential/cache/env paths from a SearchResult in-place."""
    omitted = 0

    if hasattr(result, "matches") and result.matches:
        allowed_matches = []
        for match in result.matches:
            if _search_result_read_block_error(match.path, task_id):
                omitted += 1
                continue
            allowed_matches.append(match)
        result.matches = allowed_matches

    if hasattr(result, "files") and result.files:
        allowed_files = []
        for file_path in result.files:
            if _search_result_read_block_error(file_path, task_id):
                omitted += 1
                continue
            allowed_files.append(file_path)
        result.files = allowed_files

    if hasattr(result, "counts") and result.counts:
        allowed_counts = {}
        for file_path, count in result.counts.items():
            if _search_result_read_block_error(file_path, task_id):
                omitted += 1
                continue
            allowed_counts[file_path] = count
        result.counts = allowed_counts

    return omitted


# Paths that file tools should refuse to write to without going through the
# terminal tool's approval system.  These match prefixes after os.path.realpath.
_SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/", "/private/var/",
)
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

_hermes_config_resolved: str | None = None
_hermes_config_resolved_loaded = False


def _get_hermes_config_resolved() -> str | None:
    """Return the resolved absolute path of the Hermes config file (cached)."""
    global _hermes_config_resolved, _hermes_config_resolved_loaded
    if _hermes_config_resolved_loaded:
        return _hermes_config_resolved
    _hermes_config_resolved_loaded = True
    try:
        from hermes_cli.config import get_config_path
        _hermes_config_resolved = str(get_config_path().resolve())
    except Exception:
        try:
            _hermes_config_resolved = str(Path(_expand_tilde("~/.hermes/config.yaml")).resolve())
        except Exception:
            _hermes_config_resolved = None
    return _hermes_config_resolved


def _check_sensitive_path(filepath: str, task_id: str = "default") -> str | None:
    """Return an error message if the path targets a sensitive system location."""
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except SessionRootPathError as exc:
        return str(exc)
    except (OSError, ValueError):
        resolved = filepath
    normalized = os.path.normpath(_expand_tilde(filepath))
    _err = (
        f"Refusing to write to sensitive system path: {filepath}\n"
        "Use the terminal tool with sudo if you need to modify system files."
    )
    for prefix in _SENSITIVE_PATH_PREFIXES:
        if resolved.startswith(prefix) or normalized.startswith(prefix):
            return _err
    if resolved in _SENSITIVE_EXACT_PATHS or normalized in _SENSITIVE_EXACT_PATHS:
        return _err
    # Prevent agents from modifying the Hermes config file directly.
    # approvals.mode and other security settings live here; a malicious or
    # prompt-injected agent could silently disable exec approval by writing to
    # this file.
    hermes_config = _get_hermes_config_resolved()
    if hermes_config and (resolved == hermes_config or normalized == hermes_config):
        return (
            f"Refusing to write to Hermes config file: {filepath}\n"
            "Agent cannot modify security-sensitive configuration. "
            "Edit ~/.hermes/config.yaml directly or use 'hermes config' instead."
        )
    return None


def _get_container_mirror_prefix_for_task(task_id: str = "default") -> str | None:
    """Return the container-side Hermes mirror prefix for Docker file tools."""
    try:
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            _get_env_config,
            _resolve_container_task_id,
        )

        container_key = _resolve_container_task_id(task_id)
    except Exception:
        return None

    try:
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(task_id)

        if env is not None:
            if env.__class__.__name__ == "DockerEnvironment" and bool(
                getattr(env, "_persistent", False)
            ):
                return "/root/.hermes"
            return None

        config = _get_env_config()
    except Exception:
        return None

    if config.get("env_type") == "docker" and config.get("container_persistent", True):
        return "/root/.hermes"
    return None


def _check_cross_profile_path(filepath: str, task_id: str = "default") -> str | None:
    """Return a soft-guard warning when ``filepath`` lands in another Hermes
    profile's scoped area, a host-side sandbox-mirror of authoritative profile
    state, or the Docker container's sandbox mirror of Hermes state.

    Three detectors run in order:

    * cross-profile — writes that hit another profile's
      ``skills/plugins/cron/memories`` directory.
    * sandbox-mirror (#32049) — writes that hit the
      ``…/sandboxes/<backend>/<task>/home/.hermes/…`` mirror created by a
      non-local terminal backend (Docker, Daytona, etc.), where the host
      Hermes process never reads the mirror and the authoritative file is
      left untouched.
    * container-mirror (#32049 follow-up) — writes from inside a Docker
      container whose bind-mounted home strips the ``sandboxes/`` prefix, so
      the agent sees a plain ``/root/.hermes/…`` path.

    Returns ``None`` when the write is in-scope or outside Hermes scope.
    All detectors are soft guards — the agent can override any by
    passing ``cross_profile=True`` to its write tool after explicit user
    direction. Defense-in-depth, NOT a security boundary — the terminal
    tool runs as the same OS user and can write any of these paths
    directly. See ``agent/file_safety.classify_cross_profile_target``,
    ``classify_sandbox_mirror_target`` and ``classify_container_mirror_target``
    for the detection rules.
    """
    try:
        from agent.file_safety import (
            get_container_mirror_warning,
            get_cross_profile_warning,
            get_sandbox_mirror_warning,
        )
    except Exception:
        # Fail open on import error — the existing sensitive-path guard
        # plus the write_denied list still apply.
        return None

    # Resolve via the task's cwd so a relative ``skills/foo/SKILL.md``
    # in a session that cd'd into ``~/.hermes/profiles/other/`` is
    # classified against the right base.
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except SessionRootPathError as exc:
        return str(exc)
    except (OSError, ValueError):
        resolved = filepath

    warning = get_cross_profile_warning(resolved)
    if warning is not None:
        return warning

    warning = get_sandbox_mirror_warning(resolved)
    if warning is not None:
        return warning

    return get_container_mirror_warning(
        resolved,
        mirror_prefix=_get_container_mirror_prefix_for_task(task_id),
    )


def _is_expected_write_exception(exc: Exception) -> bool:
    """Return True for expected write denials that should not hit error logs."""
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError) and exc.errno in _EXPECTED_WRITE_ERRNOS:
        return True
    return False


_file_ops_lock = threading.Lock()
_file_ops_cache: dict = {}

# Track files read per task to detect re-read loops and deduplicate reads.
# Per task_id we store:
#   "last_key":     the key of the most recent read/search call (or None)
#   "consecutive":  how many times that exact call has been repeated in a row
#   "read_history": set of (path, offset, limit) tuples for get_read_files_summary
#   "dedup":        dict mapping (resolved_path, offset, limit) → mtime float
#                   Used to skip re-reads of unchanged files.  Reset on
#                   context compression (the original content is summarised
#                   away so the model needs the full content again).
#   "read_timestamps": dict mapping resolved_path → modification-time float
#                      recorded when the file was last read (or written) by
#                      this task.  Used by write_file and patch to detect
#                      external changes between the agent's read and write.
#                      Updated after successful writes so consecutive edits
#                      by the same task don't trigger false warnings.
_read_tracker_lock = threading.Lock()
_read_tracker: dict = {}

# Track consecutive patch failures per (task_id, resolved_path).  Used to
# escalate the hint when the model repeatedly fails to patch the same file
# (typical cause: stale view of file contents, ambiguous old_string, or
# the file was modified externally between the agent's read and patch
# attempt).  Reset on a successful patch to that path.
_patch_failure_lock = threading.Lock()
_patch_failure_tracker: dict = {}  # {task_id: {resolved_path: count}}


def _record_patch_failure(task_id: str, resolved_path: str) -> int:
    """Increment and return the consecutive-failure count for this path."""
    with _patch_failure_lock:
        task_failures = _patch_failure_tracker.setdefault(task_id, {})
        # Cap dict size per task to avoid unbounded growth in long sessions
        # where the agent fails on many distinct files.  64 distinct
        # failing files per task is generous; older entries get evicted.
        if len(task_failures) >= 64 and resolved_path not in task_failures:
            try:
                first_key = next(iter(task_failures))
                del task_failures[first_key]
            except StopIteration:
                pass
        task_failures[resolved_path] = task_failures.get(resolved_path, 0) + 1
        return task_failures[resolved_path]


def _reset_patch_failures(task_id: str, resolved_paths: list) -> None:
    """Clear consecutive-failure counts for the given paths."""
    if not resolved_paths:
        return
    with _patch_failure_lock:
        task_failures = _patch_failure_tracker.get(task_id)
        if not task_failures:
            return
        for rp in resolved_paths:
            task_failures.pop(rp, None)

# Per-task bounds for the containers inside each _read_tracker[task_id].
# A CLI session uses one stable task_id for its lifetime; without these
# caps, a 10k-read session would accumulate ~1.5MB of dict/set state that
# is never referenced again (only the most recent reads matter for dedup,
# loop detection, and external-edit warnings).  Hard caps bound the
# accretion to a few hundred KB regardless of session length.
_READ_HISTORY_CAP = 500       # set; used only by get_read_files_summary
_DEDUP_CAP = 1000             # dict; skip-identical-reread guard
_READ_TIMESTAMPS_CAP = 1000   # dict; external-edit detection for write/patch
_READ_DEDUP_STATUS_MESSAGE = (
    "File unchanged since last read. The content from "
    "the earlier read_file result in this conversation is "
    "still current — refer to that instead of re-reading."
)


def _cap_read_tracker_data(task_data: dict) -> None:
    """Enforce size caps on the per-task read-tracker sub-containers.

    Must be called with ``_read_tracker_lock`` held.  Eviction policy:

      * ``read_history`` (set): pop arbitrary entries on overflow.  This
        is fine because the set only feeds diagnostic summaries; losing
        old entries just trims the summary's tail.
      * ``dedup`` / ``read_timestamps`` (dict): pop oldest by insertion
        order (Python 3.7+ dicts).  Evicted entries lose their dedup
        skip on a future re-read (the file gets re-sent once) and
        external-edit mtime comparison (the write/patch falls back to
        a non-mtime check).  Both are graceful degradations, not bugs.
    """
    rh = task_data.get("read_history")
    if rh is not None and len(rh) > _READ_HISTORY_CAP:
        excess = len(rh) - _READ_HISTORY_CAP
        for _ in range(excess):
            try:
                rh.pop()
            except KeyError:
                break

    dedup = task_data.get("dedup")
    if dedup is not None and len(dedup) > _DEDUP_CAP:
        excess = len(dedup) - _DEDUP_CAP
        for _ in range(excess):
            try:
                dedup.pop(next(iter(dedup)))
            except (StopIteration, KeyError):
                break

    dedup_hits = task_data.get("dedup_hits")
    if dedup_hits is not None and len(dedup_hits) > _DEDUP_CAP:
        excess = len(dedup_hits) - _DEDUP_CAP
        for _ in range(excess):
            try:
                dedup_hits.pop(next(iter(dedup_hits)))
            except (StopIteration, KeyError):
                break

    ts = task_data.get("read_timestamps")
    if ts is not None and len(ts) > _READ_TIMESTAMPS_CAP:
        excess = len(ts) - _READ_TIMESTAMPS_CAP
        for _ in range(excess):
            try:
                ts.pop(next(iter(ts)))
            except (StopIteration, KeyError):
                break


def _is_internal_file_status_text(content: str) -> bool:
    """Return True when content looks like an internal file-tool status, not real file bytes.

    The read_file dedup status message must never be persisted as file
    content.  The obvious shape is the model echoing the message verbatim,
    but in practice it also wraps it with small framing text (a leading
    "Note:", a trailing newline + short comment, etc.) before calling
    write_file.  We treat any short-ish write whose body is dominated by
    the status message as the same class of corruption.

    Heuristic:
      * Strict equality (after strip) — the verbatim shape.
      * OR the stripped content contains the full status message AND is
        short enough that the status dominates it (<=2x the message length).
        Short, status-dominated writes can't plausibly be real files —
        legitimate docs/notes that happen to quote this internal message
        are always dramatically longer.
    """
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    if not stripped:
        return False
    if stripped == _READ_DEDUP_STATUS_MESSAGE:
        return True
    if _READ_DEDUP_STATUS_MESSAGE in stripped and \
            len(stripped) <= 2 * len(_READ_DEDUP_STATUS_MESSAGE):
        return True
    return False


def _looks_like_read_file_line_numbered_content(content: str) -> bool:
    """Return True for content dominated by read_file's ``LINE_NUM|CONTENT`` display.

    ``read_file`` intentionally returns line-numbered text to the model. If
    that display format is echoed into ``write_file``, config/source files are
    silently corrupted with prefixes like `` 1|``.  We reject writes where the
    non-empty lines are mostly consecutive read_file-style numbered lines, while
    allowing sparse literal pipe content such as a single ``1|value`` line.
    """
    if not isinstance(content, str):
        return False

    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    numbered: list[int] = []
    for line in lines:
        stripped = line.lstrip()
        prefix, sep, _rest = stripped.partition("|")
        if sep and prefix.isdigit():
            numbered.append(int(prefix))

    if len(numbered) < 2:
        return False
    if len(numbered) / len(lines) < 0.6:
        return False

    consecutive_pairs = sum(
        1 for prev, current in zip(numbered, numbered[1:])
        if current == prev + 1
    )
    return consecutive_pairs >= len(numbered) - 1


def _is_internal_file_tool_content(content: str) -> bool:
    """Return True when content is file-tool display text, not intended file bytes."""
    return (
        _is_internal_file_status_text(content)
        or _looks_like_read_file_line_numbered_content(content)
    )


def _get_file_ops(task_id: str = "default") -> ShellFileOperations:
    """Get or create ShellFileOperations for a terminal environment.

    Respects the TERMINAL_ENV setting -- if the task_id doesn't have an
    environment yet, creates one using the configured backend (local, docker,
    modal, etc.) rather than always defaulting to local.

    Thread-safe: uses the same per-task creation locks as terminal_tool to
    prevent duplicate sandbox creation from concurrent tool calls.

    Note: subagent task_ids are collapsed to "default" via
    ``_resolve_container_task_id`` so delegate_task children share the
    parent's container and its cached file_ops. RL/benchmark task_ids with
    a registered env override keep their isolation.
    """
    from tools.terminal_tool import (
        _active_environments, _env_lock, _create_environment,
        _get_env_config, _last_activity, _start_cleanup_thread,
        _creation_locks,
        _creation_locks_lock,
        _resolve_container_task_id,
        _is_unusable_container_cwd,
        _CONTAINER_BACKENDS,
    )
    import time

    raw_task_id = task_id or "default"
    task_id = _resolve_container_task_id(raw_task_id)

    # Fast path: check cache -- but also verify the underlying environment
    # is still alive (it may have been killed by the cleanup thread).
    with _file_ops_lock:
        cached = _file_ops_cache.get(task_id)
    if cached is not None:
        with _env_lock:
            if task_id in _active_environments:
                _last_activity[task_id] = time.time()
                return cached
            else:
                # Environment was cleaned up -- preserve the old cwd in the
                # session record before invalidating the stale cache entry
                # (fixes #26211: silent file-creation failures in long-running
                # conversations). Usually a no-op: every completed command
                # already recorded its cwd.
                old_cwd = getattr(cached, "cwd", None)
                if old_cwd:
                    try:
                        from tools.terminal_tool import record_session_cwd
                        record_session_cwd(raw_task_id, old_cwd)
                    except Exception:
                        pass
                with _file_ops_lock:
                    _file_ops_cache.pop(task_id, None)

    # Need to ensure the environment exists before building file_ops.
    # Acquire per-task lock so only one thread creates the sandbox.
    with _creation_locks_lock:
        if task_id not in _creation_locks:
            _creation_locks[task_id] = threading.Lock()
        task_lock = _creation_locks[task_id]

    with task_lock:
        # Double-check: another thread may have created it while we waited
        with _env_lock:
            if task_id in _active_environments:
                _last_activity[task_id] = time.time()
                terminal_env = _active_environments[task_id]
            else:
                terminal_env = None

        if terminal_env is None:
            from tools.terminal_tool import resolve_task_overrides

            config = _get_env_config()
            env_type = config["env_type"]
            overrides = resolve_task_overrides(raw_task_id)

            if env_type == "docker":
                image = overrides.get("docker_image") or config["docker_image"]
            elif env_type == "singularity":
                image = overrides.get("singularity_image") or config["singularity_image"]
            elif env_type == "modal":
                image = overrides.get("modal_image") or config["modal_image"]
            elif env_type == "daytona":
                image = overrides.get("daytona_image") or config["daytona_image"]
            else:
                image = ""

            try:
                from tools.terminal_tool import get_session_cwd
                recorded_cwd = get_session_cwd(raw_task_id)
            except Exception:
                recorded_cwd = None
            cwd = overrides.get("cwd") or recorded_cwd or config["cwd"]
            # Re-apply the container cwd guard that _get_env_config() already
            # ran on config["cwd"] (see #50636).  A per-task cwd override
            # registered by the gateway/TUI/ACP for workspace tracking is a
            # raw host path (e.g. a Desktop session's /Users/<me>/workspace or
            # C:\\Users\\<me>). On a container backend that reaches
            # ``docker run -w <host-path>`` and the container starts in a
            # directory that doesn't exist inside the sandbox, so search_files
            # and friends silently return empty results (#54447).  Sanitize it
            # back to the already-validated config["cwd"] so the override can't
            # bypass the guard.  Valid in-container override paths (RL/benchmark
            # sandboxes that set cwd to /workspace, /root, etc.) are absolute
            # non-host paths and pass through untouched.
            if env_type in _CONTAINER_BACKENDS and _is_unusable_container_cwd(cwd):
                if cwd != config["cwd"]:
                    logger.info(
                        "Ignoring host/relative cwd override %r for %s backend "
                        "(won't exist in sandbox). Using %r instead.",
                        cwd, env_type, config["cwd"],
                    )
                cwd = config["cwd"]
            logger.info("Creating new %s environment for task %s...", env_type, task_id[:8])

            container_config = None
            if env_type in {"docker", "singularity", "modal", "daytona"}:
                container_config = {
                    "container_cpu": config.get("container_cpu", 1),
                    "container_memory": config.get("container_memory", 5120),
                    "container_disk": config.get("container_disk", 51200),
                    "container_persistent": config.get("container_persistent", True),
                    "docker_volumes": config.get("docker_volumes", []),
                    "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
                    "docker_forward_env": config.get("docker_forward_env", []),
                    "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
                    "docker_network": config.get("docker_network", True),
                }

            ssh_config = None
            if env_type == "ssh":
                ssh_config = {
                    "host": config.get("ssh_host", ""),
                    "user": config.get("ssh_user", ""),
                    "port": config.get("ssh_port", 22),
                    "key": config.get("ssh_key", ""),
                    "persistent": config.get("ssh_persistent", False),
                }

            local_config = None
            if env_type == "local":
                local_config = {
                    "persistent": config.get("local_persistent", False),
                }

            terminal_env = _create_environment(
                env_type=env_type,
                image=image,
                cwd=cwd,
                timeout=config["timeout"],
                ssh_config=ssh_config,
                container_config=container_config,
                local_config=local_config,
                task_id=task_id,
                host_cwd=config.get("host_cwd"),
            )

            with _env_lock:
                _active_environments[task_id] = terminal_env
                _last_activity[task_id] = time.time()

            _start_cleanup_thread()
            logger.info("%s environment ready for task %s", env_type, task_id[:8])

    # Build file_ops from the (guaranteed live) environment and cache it
    file_ops = ShellFileOperations(terminal_env)
    with _file_ops_lock:
        _file_ops_cache[task_id] = file_ops
        return file_ops


def _get_session_scoped_file_ops(task_id: str = "default"):
    """Return descriptor-anchored operations when a Session Root is active."""
    file_ops = _get_file_ops(task_id)
    root = _session_file_root(task_id)
    if root is None:
        return file_ops
    return _SessionRootFileOperations(root, file_ops)


def clear_file_ops_cache(task_id: str = None):
    """Clear the file operations cache."""
    with _file_ops_lock:
        if task_id:
            _file_ops_cache.pop(task_id, None)
        else:
            _file_ops_cache.clear()


def read_file_tool(path: str, offset: int = 1, limit: int = 500, task_id: str = "default") -> str:
    """Read a file with pagination and line numbers."""
    try:
        offset, limit = normalize_read_pagination(offset, limit)

        # ── Device path guard ─────────────────────────────────────────
        # Block paths that would hang the process (infinite output,
        # blocking on input).  Pure path check — no I/O.
        device_base = None if Path(path).expanduser().is_absolute() else _resolve_base_dir(task_id)
        if _is_blocked_device(path, base_dir=device_base):
            return json.dumps({
                "error": (
                    f"Cannot read '{path}': this is a device file that would "
                    "block or produce infinite output."
                ),
            })

        _resolved = _resolve_path_for_task(path, task_id)
        if _session_file_root(task_id) is None:
            hardlink_error = _session_root_hardlink_error(Path(_resolved), task_id)
            if hardlink_error:
                return tool_error(hardlink_error)

        # ── Structured-document extraction ────────────────────────────
        # Try before the binary-extension guard so .docx/.xlsx can render as text.
        # Malformed documents fall through to the normal path/binary guard.
        from tools.read_extract import ExtractionError, extract_document_text, is_extractable_document

        if is_extractable_document(str(_resolved)):
            try:
                file_ops = _get_session_scoped_file_ops(task_id)
                if isinstance(file_ops, _SessionRootFileOperations):
                    extracted_text, extracted_size = file_ops.extract_document(
                        str(_resolved)
                    )
                else:
                    extracted_text = extract_document_text(str(_resolved))
                    extracted_size = os.path.getsize(_resolved)
            except ExtractionError:
                logger.debug("document extraction failed for %s", path, exc_info=True)
            else:
                lines = extracted_text.splitlines()
                total_lines = len(lines)
                end_line = offset + limit - 1
                page_text = "\n".join(lines[offset - 1:end_line])
                result_dict = {
                    "content": (
                        (
                            file_ops.delegate
                            if isinstance(file_ops, _SessionRootFileOperations)
                            else file_ops
                        )._add_line_numbers(page_text, offset)
                        if page_text
                        else ""
                    ),
                    "total_lines": total_lines,
                    "file_size": extracted_size,
                    "truncated": total_lines > end_line,
                    "extracted_document": True,
                }
                if result_dict["truncated"]:
                    result_dict["hint"] = (
                        f"Use offset={end_line + 1} to continue reading "
                        f"(showing {offset}-{min(end_line, total_lines)} of {total_lines} lines)"
                    )
                content_len = len(result_dict["content"])
                max_chars = _get_max_read_chars()
                if content_len > max_chars:
                    # Graceful char-budget truncation (nearai/ironclaw#5029):
                    # trim to the last complete line that fits and offer a
                    # next_offset rather than rejecting the whole extraction.
                    trimmed, lines_kept, _ = _truncate_to_char_budget(
                        result_dict["content"], max_chars
                    )
                    next_offset = offset + lines_kept
                    shown_end = offset + lines_kept - 1
                    result_dict["content"] = trimmed
                    result_dict["truncated"] = True
                    result_dict["truncated_by"] = "bytes"
                    result_dict["next_offset"] = next_offset
                    result_dict["hint"] = (
                        f"Output truncated at the {max_chars:,}-char read budget "
                        f"after {lines_kept} line(s) (showing lines {offset}-"
                        f"{shown_end} of {total_lines}). Use offset={next_offset} "
                        "to continue."
                    )
                    if len(trimmed.split("\n", 1)[0]) >= max_chars:
                        result_dict["hint"] += (
                            " Note: the first line alone exceeded the budget and "
                            "was clamped mid-line; its remainder is not "
                            "retrievable via offset."
                        )
                if result_dict["content"]:
                    result_dict["content"] = redact_sensitive_text(result_dict["content"], file_read=True)
                return json.dumps(result_dict, ensure_ascii=False)

        # ── Binary file guard ─────────────────────────────────────────
        # Block binary files by extension (no I/O).
        if has_binary_extension(str(_resolved)):
            _ext = _resolved.suffix.lower()
            return json.dumps({
                "error": (
                    f"Cannot read binary file '{path}' ({_ext}). "
                    "Use vision_analyze for images, or terminal to inspect binary files."
                ),
            })

        # ── Hermes internal path guard ────────────────────────────────
        # Prevent prompt injection via catalog or hub metadata files,
        # and block credential stores under HERMES_HOME.  Pass the
        # already-resolved path so a relative-path read against
        # TERMINAL_CWD == HERMES_HOME (e.g. "auth.json") still hits the
        # denylist — get_read_block_error's own resolve() runs against
        # the Python process cwd, which can differ.
        block_error = get_read_block_error(str(_resolved))
        if block_error:
            return json.dumps({"error": block_error})

        # ── Dedup check ───────────────────────────────────────────────
        # If we already read this exact (path, offset, limit) and the
        # file hasn't been modified since, return a lightweight stub
        # instead of re-sending the same content.  Saves context tokens.
        resolved_str = str(_resolved)
        dedup_key = (resolved_str, offset, limit)
        with _read_tracker_lock:
            task_data = _read_tracker.setdefault(task_id, {
                "last_key": None, "consecutive": 0,
                "read_history": set(), "dedup": {},
                "dedup_hits": {}, "read_timestamps": {},
            })
            # Backward-compat for pre-existing tracker entries that predate
            # dedup_hits/read_timestamps (long-lived task or crossed an
            # upgrade boundary).
            if "dedup_hits" not in task_data:
                task_data["dedup_hits"] = {}
            if "read_timestamps" not in task_data:
                task_data["read_timestamps"] = {}
            cached_mtime = task_data.get("dedup", {}).get(dedup_key)

        if cached_mtime is not None:
            try:
                current_mtime = os.path.getmtime(resolved_str)
                if current_mtime == cached_mtime:
                    # Count repeated stub returns so weak tool-followers that
                    # ignore the "refer to earlier result" hint don't burn
                    # their iteration budget in an infinite read loop.  After
                    # 2 stubs for the same key we escalate to a hard block
                    # mirroring the count>=4 path on real reads.
                    with _read_tracker_lock:
                        hits = task_data["dedup_hits"].get(dedup_key, 0) + 1
                        task_data["dedup_hits"][dedup_key] = hits
                        _cap_read_tracker_data(task_data)

                    if hits >= 2:
                        return json.dumps({
                            "error": (
                                f"BLOCKED: You have called read_file on this "
                                f"exact region {hits + 1} times and the file "
                                "has NOT changed. STOP calling read_file for "
                                "this path — the content from your earlier "
                                "read_file result in this conversation is "
                                "still current. Proceed with your task using "
                                "the information you already have."
                            ),
                            "path": path,
                            "already_read": hits + 1,
                        }, ensure_ascii=False)

                    return json.dumps({
                        "status": "unchanged",
                        "message": _READ_DEDUP_STATUS_MESSAGE,
                        "path": path,
                        "dedup": True,
                        "content_returned": False,
                    }, ensure_ascii=False)
            except OSError:
                pass  # stat failed — fall through to full read

        # ── Perform the read ──────────────────────────────────────────
        file_ops = _get_session_scoped_file_ops(task_id)
        file_ops_path = resolved_str if _session_file_root(task_id) is not None else path
        result = file_ops.read_file(file_ops_path, offset, limit)
        result_dict = result.to_dict()

        # ── Character-count guard ─────────────────────────────────────
        # We're model-agnostic so we can't count tokens; characters are
        # the best proxy we have.  If the read produced an unreasonable
        # amount of content, reject it and tell the model to narrow down.
        # Note: we check the formatted content (with line-number prefixes),
        # not the raw file size, because that's what actually enters context.
        # Check BEFORE redaction to avoid expensive regex on huge content.
        content_len = len(result.content or "")
        file_size = result_dict.get("file_size", 0)
        max_chars = _get_max_read_chars()
        if content_len > max_chars:
            # Graceful char-budget truncation (ported from nearai/ironclaw#5029).
            # Instead of rejecting the whole read — which forces the model to
            # guess a smaller `limit` and wastes a round-trip returning nothing
            # — trim to the last complete line that fits and offer a
            # `next_offset` so the model can paginate forward. This rescues the
            # "few but very long lines" case (logs, wide CSVs, minified data)
            # that sails past the line-count `limit` but blows the char budget.
            total_lines = result_dict.get("total_lines", "unknown")
            trimmed, lines_kept, _ = _truncate_to_char_budget(
                result.content or "", max_chars
            )
            next_offset = offset + lines_kept
            shown_end = offset + lines_kept - 1
            result.content = trimmed
            result_dict["content"] = trimmed
            result_dict["truncated"] = True
            result_dict["truncated_by"] = "bytes"
            result_dict["next_offset"] = next_offset
            result_dict["hint"] = (
                f"Output truncated at the {max_chars:,}-char read budget after "
                f"{lines_kept} line(s) (showing lines {offset}-{shown_end} of "
                f"{total_lines}). Use offset={next_offset} to continue."
            )
            if len(trimmed.split("\n", 1)[0]) >= max_chars:
                result_dict["hint"] += (
                    " Note: the first line alone exceeded the budget and was "
                    "clamped mid-line; its remainder is not retrievable via "
                    "offset."
                )
            content_len = len(trimmed)

        # ── Redact secrets (after guard check to skip oversized content) ──
        if result.content:
            result.content = redact_sensitive_text(result.content, file_read=True)
            result_dict["content"] = result.content

        # Large-file hint: if the file is big and the caller didn't ask
        # for a narrow window, nudge toward targeted reads.
        if (file_size and file_size > _LARGE_FILE_HINT_BYTES
                and limit > 200
                and result_dict.get("truncated")):
            result_dict.setdefault("_hint", (
                f"This file is large ({file_size:,} bytes). "
                "Consider reading only the section you need with offset and limit "
                "to keep context usage efficient."
            ))

        # ── Track for consecutive-loop detection ──────────────────────
        read_key = ("read", path, offset, limit)
        with _read_tracker_lock:
            # Ensure "dedup" / "dedup_hits" keys exist (backward compat with
            # old tracker state from pre-dedup-guard sessions).
            if "dedup" not in task_data:
                task_data["dedup"] = {}
            if "dedup_hits" not in task_data:
                task_data["dedup_hits"] = {}
            # Real read succeeded — this key is no longer in a stub-loop, so
            # reset its hit counter.  (File either changed or stat failed
            # earlier and we fell through.)
            task_data["dedup_hits"].pop(dedup_key, None)
            task_data["read_history"].add((path, offset, limit))
            if task_data["last_key"] == read_key:
                task_data["consecutive"] += 1
            else:
                task_data["last_key"] = read_key
                task_data["consecutive"] = 1
            count = task_data["consecutive"]

            # Store mtime at read time for two purposes:
            # 1. Dedup: skip identical re-reads of unchanged files.
            # 2. Staleness: warn on write/patch if the file changed since
            #    the agent last read it (external edit, concurrent agent, etc.).
            try:
                _mtime_now = os.path.getmtime(resolved_str)
                task_data["dedup"][dedup_key] = _mtime_now
                task_data.setdefault("read_timestamps", {})[resolved_str] = _mtime_now
            except OSError:
                pass  # Can't stat — skip tracking for this entry

            # Bound the per-task containers so a long CLI session doesn't
            # accumulate megabytes of dict/set state.  See _cap_read_tracker_data.
            _cap_read_tracker_data(task_data)

        # Cross-agent file-state registry (separate from per-task read
        # tracker above): records that THIS agent has read this path so
        # write/patch can detect sibling-subagent writes that happened
        # after our read.  Partial read when offset>1 or the read was
        # truncated (large file with more content than limit covered).
        # Outside the _read_tracker_lock so the registry's own locking
        # isn't nested under ours.
        try:
            _partial = (offset > 1) or bool(result_dict.get("truncated"))
            file_state.record_read(task_id, resolved_str, partial=_partial)
        except Exception:
            logger.debug("file_state.record_read failed", exc_info=True)

        if count >= 4:
            # Hard block: stop returning content to break the loop
            return json.dumps({
                "error": (
                    f"BLOCKED: You have read this exact file region {count} times in a row. "
                    "The content has NOT changed. You already have this information. "
                    "STOP re-reading and proceed with your task."
                ),
                "path": path,
                "already_read": count,
            }, ensure_ascii=False)
        elif count >= 3:
            result_dict["_warning"] = (
                f"You have read this exact file region {count} times consecutively. "
                "The content has not changed since your last read. Use the information you already have. "
                "If you are stuck in a loop, stop reading and proceed with writing or responding."
            )

        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))




def reset_file_dedup(task_id: str = None):
    """Clear the deduplication cache for file reads.

    Called after context compression — the original read content has been
    summarised away, so the model needs the full content if it reads the
    same file again.  Without this, reads after compression would return
    a "file unchanged" stub pointing at content that no longer exists in
    context.

    Call with a task_id to clear just that task, or without to clear all.
    """
    with _read_tracker_lock:
        if task_id:
            task_data = _read_tracker.get(task_id)
            if task_data:
                if "dedup" in task_data:
                    task_data["dedup"].clear()
                if "dedup_hits" in task_data:
                    task_data["dedup_hits"].clear()
        else:
            for task_data in _read_tracker.values():
                if "dedup" in task_data:
                    task_data["dedup"].clear()
                if "dedup_hits" in task_data:
                    task_data["dedup_hits"].clear()


def notify_other_tool_call(task_id: str = "default"):
    """Reset consecutive read/search counter for a task.

    Called by the tool dispatcher (model_tools.py) whenever a tool OTHER
    than read_file / search_files is executed.  This ensures we only warn
    or block on *truly consecutive* repeated reads — if the agent does
    anything else in between (write, patch, terminal, etc.) the counter
    resets and the next read is treated as fresh.
    """
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data:
            task_data["last_key"] = None
            task_data["consecutive"] = 0
            # An intervening non-read tool call breaks any stub-loop in
            # progress, so clear per-key dedup hit counters too.
            if "dedup_hits" in task_data:
                task_data["dedup_hits"].clear()


def _invalidate_dedup_for_path(filepath: str, task_id: str) -> None:
    """Remove all dedup cache entries whose resolved path matches *filepath*.

    Called after write_file and patch so that a subsequent read_file on
    the same path always returns fresh content instead of a stale
    "File unchanged" stub.  The dedup cache keys are tuples of
    ``(resolved_path, offset, limit)``; we must evict **all** offset/limit
    combinations for the written path because any cached range could now
    be stale.

    Must be called with ``_read_tracker_lock`` **not** held — acquires it
    internally.
    """
    try:
        resolved = str(_resolve_path(filepath))
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is None:
            return
        dedup = task_data.get("dedup")
        if not dedup:
            return
        # Collect keys to remove (can't mutate dict during iteration).
        stale_keys = [k for k in dedup if k[0] == resolved]
        for k in stale_keys:
            del dedup[k]


def _update_read_timestamp(filepath: str, task_id: str) -> None:
    """Record the file's current modification time after a successful write.

    Called after write_file and patch so that consecutive edits by the
    same task don't trigger false staleness warnings — each write
    refreshes the stored timestamp to match the file's new state.

    Also invalidates the dedup cache for the written path so that
    subsequent reads return fresh content (fixes #13144).
    """
    # Invalidate dedup first (before acquiring lock for timestamp update).
    _invalidate_dedup_for_path(filepath, task_id)
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
        current_mtime = os.path.getmtime(resolved)
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is not None:
            task_data.setdefault("read_timestamps", {})[resolved] = current_mtime
            _cap_read_tracker_data(task_data)


def _check_file_staleness(filepath: str, task_id: str) -> str | None:
    """Check whether a file was modified since the agent last read it.

    Returns a warning string if the file is stale (mtime changed since
    the last read_file call for this task), or None if the file is fresh
    or was never read.  Does not block — the write still proceeds.
    """
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        return None
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if not task_data:
            return None
        read_mtime = task_data.get("read_timestamps", {}).get(resolved)
    if read_mtime is None:
        return None  # File was never read — nothing to compare against
    try:
        current_mtime = os.path.getmtime(resolved)
    except OSError:
        return None  # Can't stat — file may have been deleted, let write handle it
    if current_mtime != read_mtime:
        return (
            f"Warning: {filepath} was modified since you last read it "
            "(external edit or concurrent agent). The content you read may be "
            "stale. Consider re-reading the file to verify before writing."
        )
    return None


def _mark_verification_stale(
    task_id: str,
    resolved_paths: list[str],
    session_id: str | None = None,
) -> None:
    """Best-effort note that successful edits made prior verification stale."""
    paths = [p for p in resolved_paths if p]
    if not paths:
        return
    try:
        from agent.coding_context import project_facts_for
        from agent.verification_evidence import mark_workspace_edited

        cwd = None
        for path in paths:
            try:
                candidate = str(Path(path).parent)
            except Exception:
                continue
            if project_facts_for(candidate):
                cwd = candidate
                break
        if cwd is None:
            cwd = _authoritative_workspace_root(task_id)
        if cwd is None:
            try:
                cwd = str(Path(paths[0]).parent)
            except Exception:
                cwd = None
        mark_workspace_edited(session_id=session_id or task_id, cwd=cwd, paths=paths)
    except Exception:
        logger.debug("verification stale marker failed", exc_info=True)


def write_file_tool(path: str, content: str, task_id: str = "default",
                    cross_profile: bool = False,
                    session_id: str | None = None) -> str:
    """Write content to a file.

    ``cross_profile`` opts out of the soft cross-Hermes-profile guard. The
    guard fires only on writes that land in another profile's
    skills/plugins/cron/memories directory; everything else is unaffected.
    Pass ``True`` after explicit user direction — same shape as ``force``
    on the terminal tool.
    """
    sensitive_err = _check_sensitive_path(path, task_id)
    if sensitive_err:
        return tool_error(sensitive_err)
    if not cross_profile:
        cross_warning = _check_cross_profile_path(path, task_id)
        if cross_warning:
            return tool_error(cross_warning)
    if _is_internal_file_tool_content(content):
        return tool_error(
            "Refusing to write internal read_file display text as file content. "
            "Strip read_file line-number prefixes or reconstruct the intended "
            "file contents before writing."
        )
    try:
        # Resolve once for the registry lock + stale check.  Failures here
        # fall back to the legacy path — write proceeds, per-task staleness
        # check below still runs.
        try:
            _resolved = str(_resolve_path_for_task(path, task_id))
        except SessionRootPathError as exc:
            return tool_error(str(exc))
        except Exception:
            _resolved = None

        if _resolved is None:
            stale_warning = _check_file_staleness(path, task_id)
            file_ops = _get_file_ops(task_id)
            result = file_ops.write_file(path, content)
            result_dict = result.to_dict()
            if stale_warning:
                result_dict["_warning"] = stale_warning
            if not result_dict.get("error"):
                _mark_verification_stale(task_id, [path], session_id=session_id)
            _update_read_timestamp(path, task_id)
            return json.dumps(result_dict, ensure_ascii=False)

        # Serialize the read→modify→write region per-path so concurrent
        # subagents can't interleave on the same file.  Different paths
        # remain fully parallel.
        with file_state.lock_path(_resolved):
            if _session_file_root(task_id) is None:
                hardlink_error = _session_root_hardlink_error(Path(_resolved), task_id)
                if hardlink_error:
                    return tool_error(hardlink_error)
            # Cross-agent staleness wins over per-task warning when both
            # fire — its message names the sibling subagent.
            cross_warning = file_state.check_stale(task_id, _resolved)
            stale_warning = _check_file_staleness(path, task_id)
            # Workspace-divergence warning: relative path resolving outside the
            # terminal's cwd (the worktree-cwd bug). Lowest priority of the three.
            cwd_warning = _path_resolution_warning(path, Path(_resolved), task_id)
            file_ops = _get_session_scoped_file_ops(task_id)
            result = file_ops.write_file(_resolved, content)
            result_dict = result.to_dict()
            effective_warning = cross_warning or stale_warning or cwd_warning
            if effective_warning:
                result_dict["_warning"] = effective_warning
            # Always report the ABSOLUTE path actually written, so a wrong-cwd
            # mismatch is visible in the response instead of silently routing
            # the edit to the wrong checkout.
            result_dict["resolved_path"] = _resolved
            if not result_dict.get("error"):
                result_dict["files_modified"] = [_resolved]
                _mark_verification_stale(task_id, [_resolved], session_id=session_id)
            # Refresh stamps after the successful write so consecutive
            # writes by this task don't trigger false staleness warnings.
            _update_read_timestamp(path, task_id)
            if not result_dict.get("error"):
                file_state.note_write(task_id, _resolved)
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        if _is_expected_write_exception(e):
            logger.debug("write_file expected denial: %s: %s", type(e).__name__, e)
        else:
            logger.error("write_file error: %s: %s", type(e).__name__, e, exc_info=True)
        return tool_error(str(e))


def _rewrite_v4a_patch_paths(
    patch_content: str,
    path_to_resolved: dict[str, str | None],
) -> str:
    """Rewrite V4A headers to the absolute paths already validated above."""
    import re as _re

    def _resolved(raw: str) -> str:
        key = raw.strip()
        return path_to_resolved.get(key) or key

    rewritten = _re.sub(
        r"^(\*\*\*\s*(?:Update|Add|Delete)\s+File:\s*)(.+?)\s*$",
        lambda match: f"{match.group(1)}{_resolved(match.group(2))}",
        patch_content,
        flags=_re.MULTILINE,
    )
    return _re.sub(
        r"^(\*\*\*\s*Move\s+File:\s*)(.+?)\s*->\s*(.+?)\s*$",
        lambda match: (
            f"{match.group(1)}{_resolved(match.group(2))} -> "
            f"{_resolved(match.group(3))}"
        ),
        rewritten,
        flags=_re.MULTILINE,
    )


def patch_tool(mode: str = "replace", path: str = None, old_string: str = None,
               new_string: str = None, replace_all: bool = False, patch: str = None,
               task_id: str = "default", cross_profile: bool = False,
               session_id: str | None = None) -> str:
    """Patch a file using replace mode or V4A patch format.

    ``cross_profile`` opts out of the soft cross-Hermes-profile guard for
    targets under another profile's skills/plugins/cron/memories
    directory. Same shape as ``write_file``'s flag.
    """
    # Check sensitive paths for both replace (explicit path) and V4A patch (extract paths)
    _paths_to_check = []
    if path:
        _paths_to_check.append(path)
    if mode == "patch" and patch:
        import re as _re
        from tools.path_security import has_traversal_component
        def _reject_v4a_traversal(v4a_path: str) -> str | None:
            # V4A path headers come from patch CONTENT, not the explicit
            # ``path=`` arg — so they're more attacker-influenceable (skill
            # content, web extract, prompt injection). Reject ``..`` traversal
            # in V4A headers: a legitimate multi-file patch from a single cwd
            # can always emit absolute paths or paths relative to the agent's
            # cwd without ``..``. The explicit ``path=`` arg is unchanged
            # because the agent uses relative ``..`` paths legitimately
            # (e.g. ``patch path="../other_module/x.py"`` from a worktree).
            if has_traversal_component(v4a_path):
                return tool_error(
                    f"V4A patch header contains '..' traversal: {v4a_path!r}. "
                    "Use the agent's cwd-relative path (no '..') or an absolute "
                    "path in '*** Update File:' / '*** Add File:' / "
                    "'*** Delete File:' / '*** Move File:' headers."
                )
            return None

        # ``\s*`` (not ``\s+``) after ``***`` matches patch_parser leniency:
        # it accepts ``***Update File:`` with no space after the asterisks
        # (patch_parser.py uses ``\*\*\*\s*Update\s+File:``). Requiring a space
        # here let a no-space header parse + apply while skipping this check.
        for _m in _re.finditer(r'^\*\*\*\s*(?:Update|Add|Delete)\s+File:\s*(.+)$', patch, _re.MULTILINE):
            v4a_path = _m.group(1).strip()
            _err = _reject_v4a_traversal(v4a_path)
            if _err:
                return _err
            _paths_to_check.append(v4a_path)
        # ``*** Move File: src -> dst`` is a valid V4A op (patch_parser.py:114)
        # but was never extracted, so a Move targeting /etc/crontab skipped the
        # sensitive-path pre-check. Check BOTH endpoints, and run them through
        # the same ``..`` traversal rejection as the other headers.
        for _m in _re.finditer(r'^\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)$', patch, _re.MULTILINE):
            for v4a_path in (_m.group(1).strip(), _m.group(2).strip()):
                _err = _reject_v4a_traversal(v4a_path)
                if _err:
                    return _err
                _paths_to_check.append(v4a_path)
    for _p in _paths_to_check:
        sensitive_err = _check_sensitive_path(_p, task_id)
        if sensitive_err:
            return tool_error(sensitive_err)
        if not cross_profile:
            cross_warning = _check_cross_profile_path(_p, task_id)
            if cross_warning:
                return tool_error(cross_warning)
    try:
        # Resolve paths for locking.  Ordered + deduplicated so concurrent
        # callers lock in the same order — prevents deadlock on overlapping
        # multi-file V4A patches.
        _resolved_paths: list[str] = []
        _seen: set[str] = set()
        for _p in _paths_to_check:
            try:
                _r = str(_resolve_path_for_task(_p, task_id))
            except SessionRootPathError as exc:
                return tool_error(str(exc))
            except Exception:
                _r = None
            if _r and _r not in _seen:
                _resolved_paths.append(_r)
                _seen.add(_r)
        _resolved_paths.sort()

        # Acquire per-path locks in sorted order via ExitStack.  On single
        # path this degenerates to one lock; on empty list (unresolvable)
        # it's a no-op and execution falls through unchanged.
        from contextlib import ExitStack
        with ExitStack() as _locks:
            for _r in _resolved_paths:
                _locks.enter_context(file_state.lock_path(_r))

            # Collect warnings — cross-agent registry first (names sibling),
            # then per-task tracker as a fallback.
            stale_warnings: list[str] = []
            _path_to_resolved: dict[str, str] = {}
            for _p in _paths_to_check:
                try:
                    _r = str(_resolve_path_for_task(_p, task_id))
                except SessionRootPathError as exc:
                    return tool_error(str(exc))
                except Exception:
                    _r = None
                _path_to_resolved[_p] = _r
                if _r and _session_file_root(task_id) is None:
                    hardlink_error = _session_root_hardlink_error(Path(_r), task_id)
                    if hardlink_error:
                        return tool_error(hardlink_error)
                _cross = file_state.check_stale(task_id, _r) if _r else None
                _sw = _cross or _check_file_staleness(_p, task_id)
                if not _sw and _r:
                    # Workspace-divergence warning (worktree-cwd bug): relative
                    # path resolving outside the terminal's cwd.
                    _sw = _path_resolution_warning(_p, Path(_r), task_id)
                if _sw:
                    stale_warnings.append(_sw)

            file_ops = _get_session_scoped_file_ops(task_id)

            if mode == "replace":
                if not path:
                    return tool_error("path required")
                if old_string is None or new_string is None:
                    return tool_error("old_string and new_string required")
                # Pass the resolved ABSOLUTE path to the shell layer so it
                # operates on the exact file the tool layer resolved — the
                # shell's own cwd may differ (worktree-cwd bug), and a relative
                # path would let the two layers disagree about which file is
                # being edited.
                _replace_target = _path_to_resolved.get(path) or path
                result = file_ops.patch_replace(_replace_target, old_string, new_string, replace_all)
            elif mode == "patch":
                if not patch:
                    return tool_error("patch content required")
                patch_payload = (
                    _rewrite_v4a_patch_paths(patch, _path_to_resolved)
                    if _session_file_root(task_id) is not None
                    else patch
                )
                result = file_ops.patch_v4a(patch_payload)
            else:
                return tool_error(f"Unknown mode: {mode}")

            result_dict = result.to_dict()
            if stale_warnings:
                result_dict["_warning"] = stale_warnings[0] if len(stale_warnings) == 1 else " | ".join(stale_warnings)
            # Report the ABSOLUTE path(s) actually patched so a wrong-cwd
            # mismatch (e.g. a worktree session editing the main checkout) is
            # visible in the response instead of silently landing elsewhere.
            _resolved_modified = [
                _path_to_resolved.get(_p) or _p for _p in _paths_to_check
            ]
            # Refresh stored timestamps for all successfully-patched paths so
            # consecutive edits by this task don't trigger false warnings.
            if not result_dict.get("error"):
                result_dict["files_modified"] = _resolved_modified
                if len(_resolved_modified) == 1:
                    result_dict["resolved_path"] = _resolved_modified[0]
                _mark_verification_stale(task_id, _resolved_modified, session_id=session_id)
                for _p in _paths_to_check:
                    _update_read_timestamp(_p, task_id)
                    _r = _path_to_resolved.get(_p)
                    if _r:
                        file_state.note_write(task_id, _r)
                # Successful patch: clear any prior consecutive-failure
                # counters for the touched paths so a future failure on
                # the same path starts the escalation cycle fresh.
                _reset_patch_failures(task_id, [
                    _r for _r in (_path_to_resolved.get(_p) for _p in _paths_to_check) if _r
                ])
        # Hint when old_string not found — saves iterations where the agent
        # retries with stale content instead of re-reading the file.
        # Suppressed when patch_replace already attached a rich "Did you mean?"
        # snippet (which is strictly more useful than the generic hint).
        if result_dict.get("error") and "Could not find" in str(result_dict["error"]):
            # Track per-file consecutive failures for replace mode.  The
            # ``path`` arg only exists for replace mode; for V4A patches
            # we'd need to walk the headers, but in practice V4A failures
            # are far rarer and the existing _hint covers them adequately.
            failure_count = 0
            if mode == "replace" and path:
                resolved = _path_to_resolved.get(path) or path
                failure_count = _record_patch_failure(task_id, resolved)

            if failure_count >= 3:
                # Escalating hint after multiple consecutive failures on the
                # same path.  Most common cause is a stale view of the file —
                # the model is retrying with the same old_string against
                # content that has since changed.  Surface the failure count
                # so the model recognises it's in a loop and breaks out by
                # re-reading or falling back to write_file.
                result_dict["_hint"] = (
                    f"This is failure #{failure_count} patching {path!r}. "
                    "Stop retrying with variations of the same old_string. "
                    "Either: (1) re-read the file fresh to verify current "
                    "content, (2) use a longer / more unique old_string with "
                    "surrounding context lines, or (3) use write_file to "
                    "replace the entire file if the targeted region is hard "
                    "to anchor."
                )
            elif "Did you mean one of these sections?" not in str(result_dict["error"]):
                result_dict["_hint"] = (
                    "old_string not found. Use read_file to verify the current "
                    "content, or search_files to locate the text."
                )
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))


def search_tool(pattern: str, target: str = "content", path: str = ".",
                file_glob: str = None, limit: int = 50, offset: int = 0,
                output_mode: str = "content", context: int = 0,
                task_id: str = "default") -> str:
    """Search for content or files."""
    try:
        offset, limit = normalize_search_pagination(offset, limit)
        if _session_file_root(task_id) is not None:
            if offset > _MAX_SESSION_SEARCH_OFFSET:
                return tool_error(
                    "Session Root search offset exceeds the "
                    f"{_MAX_SESSION_SEARCH_OFFSET}-result pagination limit"
                )
            limit = min(limit, _MAX_SESSION_SEARCH_RESULTS)
            context = min(
                max(0, context),
                _MAX_SESSION_SEARCH_CONTEXT,
            )

        # Track searches to detect *consecutive* repeated search loops.
        # Include pagination args so users can page through truncated
        # results without tripping the repeated-search guard.
        search_key = (
            "search",
            pattern,
            target,
            str(path),
            file_glob or "",
            limit,
            offset,
        )
        with _read_tracker_lock:
            task_data = _read_tracker.setdefault(task_id, {
                "last_key": None, "consecutive": 0, "read_history": set(),
            })
            if task_data["last_key"] == search_key:
                task_data["consecutive"] += 1
            else:
                task_data["last_key"] = search_key
                task_data["consecutive"] = 1
            count = task_data["consecutive"]

        if count >= 4:
            return json.dumps({
                "error": (
                    f"BLOCKED: You have run this exact search {count} times in a row. "
                    "The results have NOT changed. You already have this information. "
                    "STOP re-searching and proceed with your task."
                ),
                "pattern": pattern,
                "already_searched": count,
            }, ensure_ascii=False)

        try:
            resolved_path = _resolve_path_for_task(path, task_id)
        except SessionRootPathError as exc:
            return tool_error(str(exc))
        except (OSError, ValueError, RuntimeError):
            resolved_path = None
        block_error = get_read_block_error(str(resolved_path) if resolved_path else path)
        if block_error:
            return json.dumps({"error": block_error}, ensure_ascii=False)
        if (
            resolved_path is not None
            and _session_file_root(task_id) is None
        ):
            tree_error = _session_root_tree_error(Path(resolved_path), task_id)
            if tree_error:
                return tool_error(tree_error)

        file_ops = _get_session_scoped_file_ops(task_id)
        file_ops_path = (
            str(resolved_path)
            if resolved_path is not None and _session_file_root(task_id) is not None
            else path
        )
        result = file_ops.search(
            pattern=pattern, path=file_ops_path, target=target, file_glob=file_glob,
            limit=limit, offset=offset, output_mode=output_mode, context=context
        )
        omitted = _filter_read_blocked_search_results(result, task_id)
        if hasattr(result, 'matches'):
            for m in result.matches:
                if hasattr(m, 'content') and m.content:
                    m.content = redact_sensitive_text(m.content, file_read=True)
        result_dict = result.to_dict(densify=True)

        if omitted:
            result_dict["_omitted"] = (
                f"{omitted} result(s) omitted because they target credential, "
                "token, cache, or secret-bearing environment files."
            )

        if count >= 3:
            result_dict["_warning"] = (
                f"You have run this exact search {count} times consecutively. "
                "The results have not changed. Use the information you already have."
            )

        result_json = json.dumps(result_dict, ensure_ascii=False)
        # Hint when results were truncated — explicit next offset is clearer
        # than relying on the model to infer it from total_count vs match count.
        if result_dict.get("truncated"):
            next_offset = offset + limit
            result_json += f"\n\n[Hint: Results truncated. Use offset={next_offset} to see more, or narrow with a more specific pattern or file_glob.]"
        return result_json
    except Exception as e:
        return tool_error(str(e))




# ---------------------------------------------------------------------------
# Schemas + Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error


def _check_file_reqs():
    """Lazy wrapper to avoid circular import with tools/__init__.py."""
    from tools import check_file_requirements
    return check_file_requirements()

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. Reads exceeding ~100K characters are truncated on a line boundary and return a next_offset; continue with offset to read the rest. Jupyter notebooks (.ipynb), Word documents (.docx), and Excel workbooks (.xlsx) are auto-extracted to readable text. NOTE: Cannot read images or other binary files — use vision_analyze for images.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (absolute, relative, or ~/path)"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed, default: 1)", "default": 1, "minimum": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines to read (default: 500, max: 2000)", "default": 500, "maximum": 2000}
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by this write are surfaced (pre-existing errors are filtered out).",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"},
            "content": {"type": "string", "description": "Complete content to write to the file"},
            "cross_profile": {
                "type": "boolean",
                "description": "Opt out of the cross-profile soft guard. Defaults to false. Set true ONLY after explicit user direction to edit another Hermes profile's skills/plugins/cron/memories — by default these writes are blocked with a warning because they affect a different profile than the one this session is running under.",
                "default": False,
            },
        },
        "required": ["path", "content"]
    }
}

PATCH_SCHEMA = {
    "name": "patch",
    "description": (
        "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. "
        "Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. "
        "Returns a unified diff. Auto-runs syntax checks after editing.\n\n"
        "REPLACE MODE (mode='replace', default): find a unique string and replace it. "
        "REQUIRED PARAMETERS: mode, path, old_string, new_string.\n"
        "PATCH MODE (mode='patch'): apply V4A multi-file patches for bulk changes. "
        "REQUIRED PARAMETERS: mode, patch."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["replace", "patch"],
                "description": "Edit mode. 'replace' (default): requires path + old_string + new_string. 'patch': requires patch content only.",
                "default": "replace",
            },
            "path": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. File path to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. Exact text to find and replace. Must be unique in the file unless replace_all=true. Include surrounding context lines to ensure uniqueness.",
            },
            "new_string": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. Replacement text. Pass empty string '' to delete the matched text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of requiring a unique match (default: false)",
                "default": False,
            },
            "patch": {
                "type": "string",
                "description": "REQUIRED when mode='patch'. V4A format patch content. Format:\n*** Begin Patch\n*** Update File: path/to/file\n@@ context hint @@\n context line\n-removed line\n+added line\n*** End Patch",
            },
            "cross_profile": {
                "type": "boolean",
                "description": "Opt out of the cross-profile soft guard. Defaults to false. Set true ONLY after explicit user direction to edit another Hermes profile's skills/plugins/cron/memories.",
                "default": False,
            },
        },
        "required": ["mode"],
    },
}

SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": "Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents.\n\nContent search (target='content'): Regex search inside files. Output modes: full matches with line numbers, file paths only, or match counts.\n\nFile search (target='files'): Find files by glob pattern (e.g., '*.py', '*config*'). Also use this instead of ls — results sorted by modification time.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search"},
            "target": {"type": "string", "enum": ["content", "files"], "description": "'content' searches inside file contents, 'files' searches for files by name", "default": "content"},
            "path": {"type": "string", "description": "Directory or file to search in (default: current working directory)", "default": "."},
            "file_glob": {"type": "string", "description": "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)"},
            "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50)", "default": 50, "minimum": 1, "maximum": 1000},
            "offset": {"type": "integer", "description": "Skip first N results for pagination (default: 0)", "default": 0, "minimum": 0, "maximum": 10000},
            "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "Output format for grep mode: 'content' shows matching lines with line numbers, 'files_only' lists file paths, 'count' shows match counts per file", "default": "content"},
            "context": {"type": "integer", "description": "Number of context lines before and after each match (grep mode only)", "default": 0, "minimum": 0, "maximum": 100}
        },
        "required": ["pattern"]
    }
}


def _handle_read_file(args, **kw):
    tid = kw.get("task_id") or "default"
    return read_file_tool(path=args.get("path", ""), offset=args.get("offset", 1), limit=args.get("limit", 500), task_id=tid)


def _handle_write_file(args, **kw):
    tid = kw.get("task_id") or "default"
    if not args.get("path") or not isinstance(args.get("path"), str):
        return tool_error(
            "write_file: missing required field 'path'. Re-emit the tool call with "
            "both 'path' and 'content' set."
        )
    if "content" not in args:
        return tool_error(
            "write_file: missing required field 'content'. The tool call included a "
            "path but no content argument — this is almost always a dropped-arg bug "
            "under context pressure. Re-emit the tool call with the full content "
            "payload, or use execute_code with hermes_tools.write_file() for very "
            "large files."
        )
    if not isinstance(args["content"], str):
        return tool_error(
            f"write_file: 'content' must be a string, got "
            f"{type(args['content']).__name__}."
        )
    return write_file_tool(
        path=args["path"], content=args["content"], task_id=tid,
        cross_profile=bool(args.get("cross_profile", False)),
        session_id=kw.get("session_id"),
    )


def _handle_patch(args, **kw):
    tid = kw.get("task_id") or "default"
    return patch_tool(
        mode=args.get("mode", "replace"), path=args.get("path"),
        old_string=args.get("old_string"), new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False), patch=args.get("patch"), task_id=tid,
        cross_profile=bool(args.get("cross_profile", False)),
        session_id=kw.get("session_id"),
    )


def _handle_search_files(args, **kw):
    tid = kw.get("task_id") or "default"
    target_map = {"grep": "content", "find": "files"}
    raw_target = args.get("target", "content")
    target = target_map.get(raw_target, raw_target)
    return search_tool(
        pattern=args.get("pattern", ""), target=target, path=args.get("path", "."),
        file_glob=args.get("file_glob"), limit=args.get("limit", 50), offset=args.get("offset", 0),
        output_mode=args.get("output_mode", "content"), context=args.get("context", 0), task_id=tid)


registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA, handler=_handle_read_file, check_fn=_check_file_reqs, emoji="📖", max_result_size_chars=100_000)
registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA, handler=_handle_write_file, check_fn=_check_file_reqs, emoji="✍️", max_result_size_chars=100_000)
registry.register(name="patch", toolset="file", schema=PATCH_SCHEMA, handler=_handle_patch, check_fn=_check_file_reqs, emoji="🔧", max_result_size_chars=100_000)
registry.register(name="search_files", toolset="file", schema=SEARCH_FILES_SCHEMA, handler=_handle_search_files, check_fn=_check_file_reqs, emoji="🔎", max_result_size_chars=100_000)
