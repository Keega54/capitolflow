"""Regenerate the binary PDF fixtures. Kept out of git so the repo stays text-only."""
import io, sys
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"


def main():
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    src = FIX / "house_ptr_sample.txt"
    pdf_path = FIX / "house_ptr_sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setFont("Helvetica", 8)
    y = 750
    for ln in src.read_text().splitlines():
        c.drawString(30, y, ln[:130])
        y -= 12
    c.save()

    # A scanned filing: same page rasterized, so the text layer is gone.
    import img2pdf
    from pdf2image import convert_from_bytes
    imgs = convert_from_bytes(pdf_path.read_bytes(), dpi=200)
    bufs = []
    for im in imgs:
        b = io.BytesIO()
        im.convert("RGB").save(b, format="PNG")
        bufs.append(b.getvalue())
    (FIX / "house_ptr_scanned.pdf").write_bytes(img2pdf.convert(bufs))
    print("fixtures rebuilt:", pdf_path.name, "house_ptr_scanned.pdf")


if __name__ == "__main__":
    main()
