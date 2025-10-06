#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import sys
from pathlib import Path


EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "venv",
    "env",
    ".venv",
    "build",
    "dist",
    "site-packages",
    ".cache",
    ".local",
}


def strip_base64_images(output_text):
    """Remove base64 image blobs and flatten lists."""
    if isinstance(output_text, list):
        output_text = "".join(output_text)
    if not isinstance(output_text, str):
        output_text = str(output_text)
    return re.sub(r"data:image\/[^;]+;base64,[A-Za-z0-9+/=\n]+", "[image stripped]", output_text)


def clean_lines(lines):
    """Remove extra blank lines inside code cells."""
    return "".join(line.rstrip() + "\n" for line in lines if line.strip() != "")


def extract_output_text(out):
    """Robustly extract textual output from any cell output."""
    out_type = out.get("output_type", "")
    text_fragments = []

    if out_type == "stream":
        text_fragments.append(out.get("text", ""))
    elif out_type in ("display_data", "execute_result"):
        data = out.get("data", {})
        if "text/plain" in data:
            text_fragments.append(data["text/plain"])
        elif "application/json" in data:
            text_fragments.append(json.dumps(data["application/json"], indent=2))
        elif "image/png" in data:
            text_fragments.append("[image stripped]")
    elif out_type == "error":
        text_fragments.append(f"{out.get('ename', '')}: {out.get('evalue', '')}")

    combined = "".join(str(x) for x in text_fragments)
    return strip_base64_images(combined)


def convert_notebook_to_markdown(ipynb_path: Path) -> str:
    """Convert a single .ipynb to markdown string (no writing)."""
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    md_lines = [f"# Notebook: {ipynb_path.name}", f"_Path: {ipynb_path.as_posix()}_", ""]

    cell_counter = 1
    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")

        if cell_type == "markdown":
            md_lines.extend(cell.get("source", []))
            md_lines.append("")  # spacing
            cell_counter += 1
            continue

        if cell_type == "code":
            md_lines.append(f"### Cell {cell_counter}:")
            md_lines.append("```python")
            md_lines.append(clean_lines(cell.get("source", [])))
            md_lines.append("```")

            outputs = cell.get("outputs", [])
            if outputs:
                all_out_text = "".join(extract_output_text(o) for o in outputs).strip()
                if all_out_text:
                    md_lines.append(f"**Cell {cell_counter} output:**")
                    md_lines.append("```python")
                    md_lines.append(all_out_text)
                    md_lines.append("```")
            md_lines.append("")  # spacing
            cell_counter += 1

    return "\n".join(md_lines)


def convert_text_file_to_markdown(file_path: Path) -> str:
    """Convert plain .py, .txt, .md files to markdown section."""
    ext = file_path.suffix.lower()
    title = {".py": "Python Script", ".txt": "Text File", ".md": "Markdown File"}.get(ext, "File")

    md_lines = [f"# {title}: {file_path.name}", f"_Path: {file_path.as_posix()}_", ""]
    code_fence = "python" if ext == ".py" else ""
    md_lines.append(f"```{code_fence}")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        md_lines.append(f.read().rstrip())
    md_lines.append("```")
    md_lines.append("")  # spacing
    return "\n".join(md_lines)


def should_exclude(file_path: Path) -> bool:
    """Return True if file should be excluded based on parent directories."""
    for part in file_path.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False


def export_all_to_markdown(root_dir: str = ".", output_file: str = "all_exports.md"):
    """Find all notebooks, py, txt, and md files, and combine them into one export."""
    root = Path(root_dir)
    output_path = Path(output_file).resolve()
    all_files = sorted(
        [
            f
            for f in root.rglob("*")
            if f.is_file()
            and f.suffix.lower() in (".ipynb", ".py", ".txt", ".md")
            and f.resolve() != output_path
            and not should_exclude(f)
        ]
    )

    if not all_files:
        print("❌ No target files found.")
        return

    combined_lines = []
    for file in all_files:
        print(f"🧾 Processing {file} ...")
        try:
            if file.suffix.lower() == ".ipynb":
                combined_lines.append(convert_notebook_to_markdown(file))
            else:
                combined_lines.append(convert_text_file_to_markdown(file))
            combined_lines.append("\n---\n")  # divider
        except Exception as e:
            print(f"⚠️ Failed to process {file}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_lines))

    print(f"✅ Combined export written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != ".":
        export_all_to_markdown(sys.argv[1])
    else:
        export_all_to_markdown(".")
