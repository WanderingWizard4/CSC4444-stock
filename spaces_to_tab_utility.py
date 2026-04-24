#!/usr/bin/env python3
"""
convert_to_tabs_in_place.py

Converts Python files to pure tab indentation
by RENAMING the original file to *_original.py
and writing the tab-converted file using the TRUE
original filename.

Example:
    agentic_controller.py  →  agentic_controller_original.py
    agentic_controller.py  (new tabbed version written here)
"""

import os
from pathlib import Path


def rename_original_file(path: Path) -> Path:
    """Rename original → *_original.py (never overwrite)."""
    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    new_name = parent / f"{stem}_original{suffix}"

    # Avoid overwriting an existing backup
    n = 1
    while new_name.exists():
        new_name = parent / f"{stem}_original{n}{suffix}"
        n += 1

    path.rename(new_name)
    return new_name


def convert_line_to_tabs(line: str) -> str:
    """Convert leading 4-space groups to tabs."""
    # Count leading spaces ONLY
    stripped = line.lstrip(" ")
    leading_spaces = len(line) - len(stripped)

    if leading_spaces > 0 and stripped:
        tabs = leading_spaces // 4
        indent = "\t" * tabs
        content = stripped
        return indent + content
    else:
        return line


def convert_file(path: Path):
    try:
        original_text = path.read_text().splitlines(keepends=False)

        # Convert each line
        converted_lines = [convert_line_to_tabs(line) + "\n"
                           for line in original_text]

        # Rename original → *_original.py
        backup_path = rename_original_file(path)
        print(f"Backed up original → {backup_path}")

        # Write converted file using the **original filename**
        path.write_text("".join(converted_lines))
        print(f"Converted (tabs) → {path}")

    except Exception as e:
        print(f"Error converting {path}: {e}")


def main():
    root = Path(".")
    py_files = [p for p in root.rglob("*.py")
                if not p.name.endswith("_original.py")
                and p.name != Path(__file__).name]

    if not py_files:
        print("No Python files found.")
        return

    print(f"Found {len(py_files)} Python files.")
    print("Originals will be renamed *_original.py.")
    print("Converted tab-indented files will keep the original filename.\n")

    confirm = input("Proceed? (y/N): ")
    if confirm.strip().lower() != "y":
        print("Canceled.")
        return

    for f in py_files:
        convert_file(f)

    print("\nDone! Originals preserved, new files tabbed.")


if __name__ == "__main__":
    main()
