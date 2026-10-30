"""Render PDF pages to PNG images for visual inspection."""
import os
import fitz  # PyMuPDF

def render_pages(pdf_path="paper/paper_presentation.pdf", output_dir="paper/pages", dpi=150):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    print(f"Opened {pdf_path}: {len(doc)} pages found.")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    saved_images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        out_name = os.path.join(output_dir, f"page_{i+1:02d}.png")
        pix.save(out_name)
        saved_images.append(out_name)
        print(f"Rendered Page {i+1} -> {out_name} ({pix.width}x{pix.height})")
    
    return saved_images

if __name__ == "__main__":
    render_pages()
