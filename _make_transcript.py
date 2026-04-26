"""Convert Claude Code JSONL transcript into a navigable markdown file.

Adds a 'Git history' section at the top (output of `git log`) so a future
reader can correlate conversation turns with code commits.

Usage:
    python _make_transcript.py <jsonl_path> <output_md_path>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

TRUNCATE_TOOL_RESULT = 2000
TRUNCATE_TOOL_INPUT = 400
MAX_TITLE_CHARS = 90


def load_lines(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def first_line_snippet(text: str, max_chars: int = MAX_TITLE_CHARS) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "…"
    return text


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s[:60]


def fence(text: str, lang: str = "") -> str:
    safe = text.replace("```", "``​`")
    return f"```{lang}\n{safe}\n```"


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def extract_tool_uses(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def extract_tool_results(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


def normalize_tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def strip_system_reminders(text: str) -> str:
    return re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL).strip()


def strip_command_noise(text: str) -> str:
    text = re.sub(r"<command-name>.*?</command-name>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<command-message>.*?</command-message>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<command-args>.*?</command-args>\s*", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<local-command-stdout>(.*?)</local-command-stdout>",
        r"\n> *[local command output]*\n> ```\n\1\n> ```\n",
        text, flags=re.DOTALL,
    )
    text = re.sub(r"<local-command-caveat>.*?</local-command-caveat>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<task-notification>.*?</task-notification>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def render_tool_input(name: str, inp: dict[str, Any]) -> str:
    summary = ""
    if name == "Bash":
        summary = inp.get("description") or inp.get("command", "")[:120]
    elif name in ("Read", "Edit", "Write"):
        summary = inp.get("file_path", "")
    elif name == "Glob":
        summary = inp.get("pattern", "")
    elif name == "Grep":
        summary = f"pattern={inp.get('pattern', '')!r} path={inp.get('path', '')}"
    elif name in ("TaskCreate", "TaskUpdate"):
        summary = inp.get("subject") or f"taskId={inp.get('taskId', '')} status={inp.get('status', '')}"
    elif name == "Agent":
        summary = inp.get("description", "") or (inp.get("prompt", "")[:100])
    else:
        first_val = next(iter(inp.values()), "")
        summary = str(first_val)[:120] if first_val else ""

    summary_escaped = summary.replace("\n", " ").strip()
    if len(summary_escaped) > 120:
        summary_escaped = summary_escaped[:120] + "…"

    full_json = json.dumps(inp, indent=2, ensure_ascii=False)
    if len(full_json) <= TRUNCATE_TOOL_INPUT:
        return f"**{name}**: `{summary_escaped}`\n\n{fence(full_json, 'json')}"
    return (
        f"**{name}**: `{summary_escaped}`\n\n"
        f"<details><summary>show full input</summary>\n\n{fence(full_json, 'json')}\n\n</details>"
    )


def render_tool_result(text: str) -> str:
    text = strip_system_reminders(text)
    if not text:
        return "_(empty)_"
    if len(text) <= TRUNCATE_TOOL_RESULT:
        return fence(text)
    truncated = text[:TRUNCATE_TOOL_RESULT]
    rest = text[TRUNCATE_TOOL_RESULT:]
    return (
        f"{fence(truncated)}\n\n"
        f"<details><summary>... {len(rest):,} more chars</summary>\n\n"
        f"{fence(rest)}\n\n</details>"
    )


def get_git_history(repo_root: Path) -> str:
    """Return git log output. Empty string if not a git repo or git unavailable."""
    if not (repo_root / ".git").exists():
        return ""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "log",
             "--pretty=format:%h | %ad | %s%n  %b%n",
             "--date=short", "--no-merges"],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        return out
    except Exception as e:
        return f"(git log failed: {e})"


def build_markdown(lines: list[dict[str, Any]], git_history: str) -> str:
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for entry in lines:
        msg = entry.get("message") or {}
        role = msg.get("role") or entry.get("type")
        content = msg.get("content", "")

        if role == "user":
            tool_results = extract_tool_results(content)
            text = extract_text(content) if isinstance(content, list) else (content if isinstance(content, str) else "")

            if tool_results and not text.strip():
                if current is not None:
                    current["tool_results"].extend(tool_results)
                continue

            if tool_results and current is not None:
                current["tool_results"].extend(tool_results)

            user_text = strip_command_noise(strip_system_reminders(text)).strip()
            if not user_text:
                continue

            current = {
                "user_text": user_text,
                "assistant_text_blocks": [],
                "tool_uses": [],
                "tool_results": [],
            }
            turns.append(current)

        elif role == "assistant":
            if current is None:
                current = {
                    "user_text": "_(assistant spoke first)_",
                    "assistant_text_blocks": [],
                    "tool_uses": [],
                    "tool_results": [],
                }
                turns.append(current)
            asst_text = extract_text(content)
            if asst_text.strip():
                current["assistant_text_blocks"].append(asst_text)
            current["tool_uses"].extend(extract_tool_uses(content))

    out: list[str] = []
    out.append("# UGP_amarb — Road Anomaly Analysis: Project Conversation")
    out.append("")
    out.append(f"_{len(turns)} turns of dialogue. Auto-generated from the local Claude Code transcript._")
    out.append("")
    out.append("This document is the complete development history of the project, captured")
    out.append("turn-by-turn between the user and the AI pair-programmer (Claude Opus 4.7).")
    out.append("")
    out.append("If you are an AI or developer picking up this project for the first time:")
    out.append("- `CLAUDE.md` (sibling file) is your primary architectural reference")
    out.append("- `project_memory/` holds locked-in technical decisions")
    out.append("- This transcript shows *how* and *why* those decisions were reached")
    out.append("")
    out.append("---")
    out.append("")

    if git_history.strip():
        out.append("## Git history")
        out.append("")
        out.append("Every phase of work was committed with a detailed message describing what")
        out.append("changed and why. The commits below are reverse-chronological (newest first).")
        out.append("")
        out.append(fence(git_history.strip()))
        out.append("")
        out.append("---")
        out.append("")

    out.append("## Table of contents")
    out.append("")

    toc_entries = []
    for i, t in enumerate(turns, 1):
        title = first_line_snippet(t["user_text"])
        anchor = f"turn-{i}-{slugify(title)}"
        toc_entries.append((i, title, anchor))
        out.append(f"- [Turn {i} — {title}](#{anchor})")
    out.append("")
    out.append("---")
    out.append("")

    for (i, title, anchor), t in zip(toc_entries, turns):
        out.append(f'<a id="{anchor}"></a>')
        out.append(f"## Turn {i} — {title}")
        out.append("")
        out.append("### 🧑 User")
        out.append("")
        out.append(t["user_text"])
        out.append("")

        if t["assistant_text_blocks"] or t["tool_uses"]:
            out.append("### 🤖 Assistant")
            out.append("")
            for block in t["assistant_text_blocks"]:
                out.append(block.strip())
                out.append("")

            if t["tool_uses"]:
                result_by_id = {}
                for r in t["tool_results"]:
                    rid = r.get("tool_use_id") or r.get("id")
                    result_by_id[rid] = normalize_tool_result_text(r.get("content", ""))

                out.append(f"<details><summary>🔧 Tool calls ({len(t['tool_uses'])})</summary>")
                out.append("")
                for tu in t["tool_uses"]:
                    name = tu.get("name", "?")
                    inp = tu.get("input", {}) or {}
                    tu_id = tu.get("id")
                    out.append(render_tool_input(name, inp))
                    out.append("")
                    result_text = result_by_id.get(tu_id, "")
                    if result_text:
                        out.append("**Result:**")
                        out.append("")
                        out.append(render_tool_result(result_text))
                        out.append("")
                out.append("</details>")
                out.append("")

        out.append("---")
        out.append("")

    return "\n".join(out)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        return 1
    repo_root = dst.parent
    print(f"Reading {src}")
    lines = load_lines(src)
    print(f"  {len(lines):,} JSONL entries")
    print(f"Reading git history from {repo_root}")
    git_hist = get_git_history(repo_root)
    print(f"  {len(git_hist):,} chars of git log")
    md = build_markdown(lines, git_hist)
    dst.write_text(md, encoding="utf-8")
    print(f"Wrote {dst}")
    print(f"  {len(md):,} chars, {md.count(chr(10)):,} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
