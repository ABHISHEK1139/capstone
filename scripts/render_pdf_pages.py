"""Render PDF pages to PNG images for visual layout inspection."""

import argparse
import os
import fitz  # PyMuPDF


def render_pages(pdf_path: str = "paper/paper-ieee.pdf", output_dir: str = "paper/inspect_pages", dpi: int = 150):
    """Render each page of a PDF to PNG images inside output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    # Remove stale PNGs so page counts remain exact
    for fname in os.listdir(output_dir):
        if fname.endswith(".png"):
            os.remove(os.path.join(output_dir, fname))

    doc = fitz.open(pdf_path)
    print(f"Opened {pdf_path}: {len(doc)} pages found.")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    saved_images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        out_name = os.path.join(output_dir, f"page_{i + 1:02d}.png")
        pix.save(out_name)
        saved_images.append(out_name)
        print(f"Rendered Page {i + 1} -> {out_name} ({pix.width}x{pix.height})")

    return saved_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render PDF pages to PNG images for inspection")
    parser.add_argument("--pdf", default="paper/paper-ieee.pdf", help="Path to PDF file")
    parser.add_argument("--out-dir", default="paper/inspect_pages", help="Output PNG directory")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI")
    args = parser.parse_args()
    render_pages(args.pdf, args.out_dir, args.dpi)
