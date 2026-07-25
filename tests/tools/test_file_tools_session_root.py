import json
import os
import shutil

import pytest


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="descriptor-anchored Session Root operations require POSIX dir_fd support",
)


@pytest.fixture(autouse=True)
def session_root_cleanup(monkeypatch):
    from agent.runtime_cwd import clear_session_cwd, clear_session_file_root
    from tools import file_tools

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    # macOS pytest roots live below /private/var, which is intentionally in
    # the general system-path denylist. These tests exercise the independent
    # Session Root boundary, so keep that unrelated guard out of the fixture.
    monkeypatch.setattr(file_tools, "_SENSITIVE_PATH_PREFIXES", ())
    clear_session_cwd()
    clear_session_file_root()
    file_tools.clear_file_ops_cache()
    yield
    clear_session_cwd()
    clear_session_file_root()
    file_tools.clear_file_ops_cache()


def _set_session_root(path):
    from agent.runtime_cwd import set_session_cwd, set_session_file_root

    set_session_cwd(str(path))
    set_session_file_root(str(path))


def _payload(raw):
    return json.JSONDecoder().raw_decode(raw)[0]


def test_session_root_allows_relative_write_and_read(tmp_path):
    from tools.file_tools import read_file_tool, write_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    _set_session_root(root)

    written = _payload(
        write_file_tool("notes/sentinel.md", "user-a only\n", task_id="root-allow")
    )
    read = _payload(read_file_tool("notes/sentinel.md", task_id="root-allow"))

    assert "error" not in written
    assert (root / "notes" / "sentinel.md").read_text() == "user-a only\n"
    assert "error" not in read
    assert "user-a only" in read["content"]


def test_session_root_rejects_absolute_and_dotdot_escapes(tmp_path):
    from tools.file_tools import patch_tool, read_file_tool, search_tool, write_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    outside = tmp_path / "user-b-secret.txt"
    outside.write_text("outside\n")
    _set_session_root(root)

    read = _payload(read_file_tool(str(outside), task_id="root-escape"))
    write = _payload(
        write_file_tool("../user-b-secret.txt", "pwned\n", task_id="root-escape")
    )
    search = _payload(
        search_tool("outside", path=str(outside), task_id="root-escape")
    )
    patch = _payload(
        patch_tool(
            mode="replace",
            path=str(outside),
            old_string="outside",
            new_string="pwned",
            task_id="root-escape",
        )
    )

    for result in (read, write, search, patch):
        assert "current session file root" in result["error"]
    assert outside.read_text() == "outside\n"


@pytest.mark.skipif(
    os.name == "nt",
    reason="creating symlinks requires privileges not guaranteed on Windows",
)
def test_session_root_rejects_symlink_escape_and_search_tree(tmp_path):
    from tools.file_tools import read_file_tool, search_tool, write_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    outside = tmp_path / "user-b-secret.txt"
    outside.write_text("outside\n")
    (root / "link.txt").symlink_to(outside)
    _set_session_root(root)

    read = _payload(read_file_tool("link.txt", task_id="root-symlink"))
    write = _payload(
        write_file_tool("link.txt", "pwned\n", task_id="root-symlink")
    )
    search = _payload(
        search_tool("outside", path=".", task_id="root-symlink")
    )

    assert "current session file root" in read["error"]
    assert "current session file root" in write["error"]
    assert "error" not in search
    assert search["total_count"] == 0
    assert outside.read_text() == "outside\n"


def test_session_root_resolution_error_fails_closed(monkeypatch, tmp_path):
    from tools.file_tools import write_file_tool

    monkeypatch.setattr("tools.file_tools._SENSITIVE_PATH_PREFIXES", ())
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "must-not-be-written.txt"

    def fail_root_resolution():
        raise RuntimeError("context unavailable")

    monkeypatch.setattr(
        "agent.runtime_cwd.resolve_session_file_root",
        fail_root_resolution,
    )

    result = _payload(
        write_file_tool("must-not-be-written.txt", "pwned\n", task_id="root-context-error")
    )

    assert "session file root context is unavailable" in result["error"].lower()
    assert not outside.exists()


@pytest.mark.skipif(
    os.name != "posix",
    reason="descriptor-anchored Session Root operations require POSIX dir_fd support",
)
def test_session_root_write_resists_ancestor_symlink_swap(monkeypatch, tmp_path):
    from tools.file_tools import write_file_tool

    root = tmp_path / "user-a"
    nested = root / "nested"
    pinned = root / "nested-pinned"
    outside = tmp_path / "user-b"
    nested.mkdir(parents=True)
    outside.mkdir()
    _set_session_root(root)
    monkeypatch.setattr("tools.file_tools._SENSITIVE_PATH_PREFIXES", ())

    original_replace = os.replace
    swapped = False

    def replace_after_swap(src, dst, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            nested.rename(pinned)
            nested.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace_after_swap)

    result = _payload(
        write_file_tool("nested/proof.txt", "inside\n", task_id="root-race-write")
    )

    assert "error" not in result
    assert swapped is True
    assert (pinned / "proof.txt").read_text() == "inside\n"
    assert not (outside / "proof.txt").exists()


@pytest.mark.skipif(
    os.name != "posix",
    reason="descriptor-anchored Session Root operations require POSIX dir_fd support",
)
def test_session_root_read_resists_ancestor_symlink_swap(monkeypatch, tmp_path):
    from tools.file_tools import read_file_tool

    root = tmp_path / "user-a"
    nested = root / "nested"
    pinned = root / "nested-pinned"
    outside = tmp_path / "user-b"
    nested.mkdir(parents=True)
    outside.mkdir()
    (nested / "proof.txt").write_text("inside\n")
    (outside / "proof.txt").write_text("outside\n")
    _set_session_root(root)

    original_open = os.open
    swapped = False

    def open_after_swap(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "proof.txt" and kwargs.get("dir_fd") is not None and not swapped:
            nested.rename(pinned)
            nested.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_after_swap)

    result = _payload(
        read_file_tool("nested/proof.txt", task_id="root-race-read")
    )

    assert "error" not in result
    assert swapped is True
    assert "inside" in result["content"]
    assert "outside" not in result["content"]


@pytest.mark.skipif(
    os.name != "posix",
    reason="descriptor-anchored Session Root operations require POSIX dir_fd support",
)
def test_session_root_search_resists_ancestor_symlink_swap(monkeypatch, tmp_path):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    nested = root / "nested"
    pinned = root / "nested-pinned"
    outside = tmp_path / "user-b"
    nested.mkdir(parents=True)
    outside.mkdir()
    (nested / "proof.txt").write_text("INSIDE-ONLY\n")
    (outside / "proof.txt").write_text("OUTSIDE-SECRET\n")
    _set_session_root(root)

    original_scandir = os.scandir
    swapped = False

    def scandir_after_swap(path):
        nonlocal swapped
        if isinstance(path, int) and not swapped:
            nested.rename(pinned)
            nested.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir_after_swap)

    result = _payload(
        search_tool(
            "INSIDE-ONLY|OUTSIDE-SECRET",
            path="nested",
            task_id="root-race-search",
        )
    )

    assert "error" not in result
    assert swapped is True
    assert result["total_count"] == 1
    assert result["matches"][0]["content"] == "INSIDE-ONLY"


def test_session_root_rejects_hardlink_aliases(tmp_path):
    from tools.file_tools import read_file_tool, search_tool, write_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    outside = tmp_path / "user-b-secret.txt"
    outside.write_text("outside\n")
    link = root / "linked.txt"
    try:
        os.link(outside, link)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable on this filesystem: {exc}")
    _set_session_root(root)

    read = _payload(read_file_tool("linked.txt", task_id="root-hardlink"))
    write = _payload(
        write_file_tool("linked.txt", "pwned\n", task_id="root-hardlink")
    )
    search = _payload(
        search_tool("outside", path=".", task_id="root-hardlink")
    )

    for result in (read, write, search):
        assert "multiple filesystem links" in result["error"]
    assert outside.read_text() == "outside\n"


def test_session_root_rejects_hardlink_as_direct_search_target(tmp_path):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    outside = tmp_path / "user-b-secret.txt"
    outside.write_text("outside\n")
    link = root / "linked.txt"
    try:
        os.link(outside, link)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable on this filesystem: {exc}")
    _set_session_root(root)

    result = _payload(
        search_tool("outside", path="linked.txt", task_id="root-hardlink-file")
    )

    assert "multiple filesystem links" in result["error"]


def test_session_root_rewrites_v4a_patch_paths_to_validated_root(tmp_path):
    from tools.file_tools import patch_tool

    root = tmp_path / "user-a"
    root.mkdir()
    target = root / "app.py"
    target.write_text("print('old')\n")
    _set_session_root(root)

    result = _payload(
        patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Update File: app.py\n"
                "@@\n"
                "-print('old')\n"
                "+print('new')\n"
                "*** End Patch\n"
            ),
            task_id="root-v4a",
        )
    )

    assert result["success"] is True
    assert target.read_text() == "print('new')\n"
    assert result["files_modified"] == [str(target.resolve())]


@pytest.mark.parametrize(
    "operation",
    ("Update", "Add", "Delete"),
)
def test_session_root_rejects_v4a_file_headers_outside_root(tmp_path, operation):
    from tools.file_tools import patch_tool

    root = tmp_path / "user-a"
    root.mkdir()
    outside = tmp_path / "user-b-secret.txt"
    outside.write_text("outside\n")
    _set_session_root(root)

    result = _payload(
        patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                f"*** {operation} File: {outside}\n"
                "*** End Patch\n"
            ),
            task_id=f"root-v4a-{operation.lower()}",
        )
    )

    assert "current session file root" in result["error"]
    assert outside.read_text() == "outside\n"


def test_session_root_rejects_v4a_move_destination_outside_root(tmp_path):
    from tools.file_tools import patch_tool

    root = tmp_path / "user-a"
    root.mkdir()
    source = root / "source.txt"
    source.write_text("inside\n")
    outside = tmp_path / "user-b-secret.txt"
    _set_session_root(root)

    result = _payload(
        patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                f"*** Move File: source.txt -> {outside}\n"
                "*** End Patch\n"
            ),
            task_id="root-v4a-move",
        )
    )

    assert "current session file root" in result["error"]
    assert source.read_text() == "inside\n"
    assert not outside.exists()


def test_session_root_v4a_move_rejects_source_swap(monkeypatch, tmp_path):
    from tools.file_tools import patch_tool

    root = tmp_path / "user-a"
    root.mkdir()
    source = root / "source.txt"
    pinned = root / "source-pinned.txt"
    destination = root / "destination.txt"
    outside = tmp_path / "user-b-secret.txt"
    source.write_text("inside\n")
    outside.write_text("outside\n")
    _set_session_root(root)

    original_rename = os.rename
    swapped = False

    def rename_after_swap(src, dst, *args, **kwargs):
        nonlocal swapped
        if (
            src == "source.txt"
            and kwargs.get("src_dir_fd") is not None
            and not swapped
        ):
            original_rename(source, pinned)
            source.symlink_to(outside)
            swapped = True
        return original_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", rename_after_swap)

    result = _payload(
        patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Move File: source.txt -> destination.txt\n"
                "*** End Patch\n"
            ),
            task_id="root-v4a-move-race",
        )
    )

    assert "error" in result
    assert swapped is True
    assert pinned.read_text() == "inside\n"
    assert outside.read_text() == "outside\n"
    assert not destination.exists()


def test_session_root_search_preserves_context_count_and_file_order(tmp_path):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    older = root / "older.txt"
    newer = root / "newer.txt"
    older.write_text("before\nMATCH\nafter\nMATCH\n")
    newer.write_text("MATCH\n")
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))
    _set_session_root(root)

    context = _payload(
        search_tool(
            "MATCH",
            path=".",
            context=1,
            task_id="root-search-context",
        )
    )
    counts = _payload(
        search_tool(
            "MATCH",
            path=".",
            output_mode="count",
            task_id="root-search-count",
        )
    )
    files = _payload(
        search_tool(
            "*.txt",
            path=".",
            target="files",
            task_id="root-search-files",
        )
    )

    assert context["total_count"] == 5
    assert "1: before" in context["matches_text"]
    assert "2: MATCH" in context["matches_text"]
    assert "3: after" in context["matches_text"]
    assert "4: MATCH" in context["matches_text"]
    assert counts["total_count"] == 3
    assert counts["counts"][str(older)] == 2
    assert counts["counts"][str(newer)] == 1
    assert files["files"] == [str(newer), str(older)]


def test_session_root_search_skips_oversized_files(monkeypatch, tmp_path):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    (root / "large.txt").write_text("MATCH-ME\n" * 10)
    _set_session_root(root)
    monkeypatch.setattr("tools.file_tools._MAX_SESSION_SEARCH_FILE_BYTES", 8)

    result = _payload(
        search_tool("MATCH-ME", path=".", task_id="root-search-large")
    )

    assert result["total_count"] == 0
    assert result["truncated"] is True
    assert "size limit" in result["limit_reason"]


def test_session_root_extracts_ipynb_from_pinned_snapshot(monkeypatch, tmp_path):
    from tools.file_tools import read_file_tool
    from tools.file_tools import _SessionRootFileOperations

    root = tmp_path / "user-a"
    root.mkdir()
    notebook = root / "lesson.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": ["SESSION-ROOT-NOTEBOOK"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    _set_session_root(root)
    snapshot_calls = 0
    original_extract = _SessionRootFileOperations.extract_document

    def tracking_extract(self, path):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_extract(self, path)

    monkeypatch.setattr(
        _SessionRootFileOperations,
        "extract_document",
        tracking_extract,
    )

    result = _payload(
        read_file_tool("lesson.ipynb", task_id="root-document-extraction")
    )

    assert "error" not in result
    assert snapshot_calls == 1
    assert result["extracted_document"] is True
    assert "SESSION-ROOT-NOTEBOOK" in result["content"]


def test_session_root_document_extraction_rejects_symlink_swap(
    monkeypatch,
    tmp_path,
):
    from tools.file_tools import read_file_tool
    from tools.file_tools import _SessionRootFileOperations

    root = tmp_path / "user-a"
    root.mkdir()
    notebook = root / "lesson.ipynb"
    pinned = root / "lesson-pinned.ipynb"
    outside = tmp_path / "user-b-secret.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [{"cell_type": "markdown", "source": ["INSIDE"]}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    outside.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["OUTSIDE-SECRET"]}
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    _set_session_root(root)
    original_extract = _SessionRootFileOperations.extract_document
    swapped = False

    def extract_after_swap(self, path):
        nonlocal swapped
        notebook.rename(pinned)
        notebook.symlink_to(outside)
        swapped = True
        return original_extract(self, path)

    monkeypatch.setattr(
        _SessionRootFileOperations,
        "extract_document",
        extract_after_swap,
    )

    raw = read_file_tool("lesson.ipynb", task_id="root-document-race")
    result = _payload(raw)

    assert swapped is True
    assert "error" in result
    assert "OUTSIDE-SECRET" not in raw


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_session_root_runs_shell_lint_against_pinned_snapshot(tmp_path):
    from tools.file_tools import write_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    _set_session_root(root)

    result = _payload(
        write_file_tool(
            "broken.js",
            "const broken = ;\n",
            task_id="root-shell-lint",
        )
    )

    assert result["lint"]["status"] == "error"
    assert result["lint"]["output"]
    assert "hermes-session-lint-" not in result["lint"]["output"]
    assert str(root / "broken.js") in result["lint"]["output"]


def test_session_root_read_clamps_single_unbounded_line(monkeypatch, tmp_path):
    from tools.file_tools import read_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    (root / "huge.txt").write_text("x" * 100_000)
    _set_session_root(root)
    monkeypatch.setattr("tools.file_tools._max_read_chars_cached", 1_000)

    result = _payload(
        read_file_tool("huge.txt", task_id="root-read-long-line")
    )

    assert "error" not in result
    assert len(result["content"]) <= 1_010
    assert result["truncated"] is True
    assert "clamped" in result["hint"].lower()


def test_session_root_read_enforces_request_budget_across_many_wide_lines(
    monkeypatch,
    tmp_path,
):
    from tools.file_tools import read_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    (root / "many-wide.txt").write_text(("x" * 5_000 + "\n") * 100)
    _set_session_root(root)
    monkeypatch.setattr("tools.file_tools._max_read_chars_cached", 1_000)

    result = _payload(
        read_file_tool(
            "many-wide.txt",
            limit=100,
            task_id="root-read-many-wide",
        )
    )

    assert "error" not in result
    assert len(result["content"]) <= 1_010
    assert result["truncated"] is True


def test_session_root_search_caps_match_width_and_normalizes_negative_context(
    tmp_path,
):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    (root / "wide.txt").write_text("MATCH-" + ("x" * 10_000) + "\n")
    _set_session_root(root)

    result = _payload(
        search_tool(
            "MATCH",
            path=".",
            context=-5,
            task_id="root-search-wide",
        )
    )

    assert result["total_count"] == 1
    assert len(result["matches"][0]["content"]) == 500


def test_session_root_search_applies_glob_before_size_limit(
    monkeypatch,
    tmp_path,
):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    (root / "ignored.log").write_text("MATCH\n" * 10)
    (root / "included.txt").write_text("MATCH\n")
    _set_session_root(root)
    monkeypatch.setattr("tools.file_tools._MAX_SESSION_SEARCH_FILE_BYTES", 8)

    result = _payload(
        search_tool(
            "MATCH",
            path=".",
            file_glob="*.txt",
            task_id="root-search-glob-size",
        )
    )

    assert result["total_count"] == 1
    assert result.get("truncated") is not True
    assert "limit_reason" not in result


def test_session_root_search_stops_after_bounded_result_page(tmp_path):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    (root / "many.txt").write_text("MATCH\n" * 100_000)
    _set_session_root(root)

    result = _payload(
        search_tool(
            "MATCH",
            path=".",
            limit=3,
            task_id="root-search-result-budget",
        )
    )

    assert len(result["matches"]) == 3
    assert result["total_count"] == 4
    assert result["truncated"] is True


def test_session_root_search_clamps_untrusted_result_limit(tmp_path):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    (root / "many.txt").write_text("MATCH\n" * 2_000)
    _set_session_root(root)

    result = _payload(
        search_tool(
            "MATCH",
            path=".",
            limit=1_000_000,
            task_id="root-search-limit-clamp",
        )
    )

    assert result["total_count"] == 1_001
    assert result["truncated"] is True
    assert result["matches_text"].count("\n  ") == 1_000


def test_session_root_search_rejects_unbounded_offset(tmp_path):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    (root / "match.txt").write_text("MATCH\n")
    _set_session_root(root)

    result = _payload(
        search_tool(
            "MATCH",
            path=".",
            offset=10_001,
            task_id="root-search-offset-limit",
        )
    )

    assert "offset exceeds" in result["error"]


def test_session_root_file_search_stops_at_enumeration_budget(
    monkeypatch,
    tmp_path,
):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    for index in range(3):
        (root / f"match-{index}.txt").write_text("content\n")
    _set_session_root(root)
    monkeypatch.setattr(
        "tools.file_tools._MAX_SESSION_SEARCH_ENTRIES",
        2,
    )

    result = _payload(
        search_tool(
            "*.txt",
            target="files",
            path=".",
            task_id="root-file-search-enumeration-limit",
        )
    )

    assert len(result["files"]) == 2
    assert result["total_count"] == 2
    assert result["truncated"] is True
    assert "enumeration budget" in result["limit_reason"]


@pytest.mark.parametrize(
    ("target", "pattern"),
    [("content", "MATCH"), ("files", "*.txt")],
)
def test_session_root_search_counts_empty_directories_against_walk_budget(
    monkeypatch,
    tmp_path,
    target,
    pattern,
):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    for index in range(3):
        (root / f"empty-{index}").mkdir()
    _set_session_root(root)
    monkeypatch.setattr(
        "tools.file_tools._MAX_SESSION_SEARCH_ENTRIES",
        2,
    )

    result = _payload(
        search_tool(
            pattern,
            target=target,
            path=".",
            task_id=f"root-{target}-search-empty-dir-limit",
        )
    )

    assert result["truncated"] is True
    assert "enumeration budget" in result["limit_reason"]


def test_session_root_search_times_out_catastrophic_regex(
    monkeypatch,
    tmp_path,
):
    from tools.file_tools import search_tool

    root = tmp_path / "user-a"
    root.mkdir()
    (root / "adversarial.txt").write_text(("a" * 100_000) + "!\n")
    _set_session_root(root)
    monkeypatch.setattr(
        "tools.file_tools._SESSION_ROOT_SEARCH_TIMEOUT_SECONDS",
        0.05,
    )

    result = _payload(
        search_tool(
            "(a+)+$",
            path=".",
            task_id="root-search-regex-timeout",
        )
    )

    assert "timed out" in result["error"]
