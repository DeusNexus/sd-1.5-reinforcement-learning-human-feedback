"""Grid visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def save_image_grid(
    image_paths: Iterable[Path],
    labels: Iterable[str],
    output_path: Path,
    grid_size: int = 16,
    thumb_size: int = 160,
    font_path: str | None = None,
) -> None:
    image_paths = list(image_paths)
    labels = list(labels)
    count = min(len(image_paths), grid_size)
    cols = int(count**0.5)
    cols = max(1, cols)
    rows = (count + cols - 1) // cols

    grid_w = cols * thumb_size
    grid_h = rows * thumb_size
    grid = Image.new("RGB", (grid_w, grid_h), color=(10, 10, 10))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype(font_path, 12) if font_path else ImageFont.load_default()

    for idx in range(count):
        img = Image.open(image_paths[idx]).convert("RGB")
        img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        x = (idx % cols) * thumb_size
        y = (idx // cols) * thumb_size
        grid.paste(img, (x, y))
        label = labels[idx] if idx < len(labels) else ""
        if label:
            draw.rectangle([x, y, x + thumb_size, y + 14], fill=(0, 0, 0))
            draw.text((x + 2, y + 1), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
