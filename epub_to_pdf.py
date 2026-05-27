import os
import sys

import ebooklib
from ebooklib import epub
from reportlab.pdfgen import canvas


def convert_epub_to_pdf(epub_path, pdf_path):
    book = epub.read_epub(epub_path)

    pdf_canvas = canvas.Canvas(pdf_path)

    for item in book.get_items():
        if isinstance(item, ebooklib.epub.EpubHtml):
            content = item.content
            # Process the content (e.g., convert HTML to text)
            pdf_canvas.drawString(100, 700, content)

    pdf_canvas.save()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query from command line
        epub_path_o = sys.argv[1].strip().replace("\"", "")
        if not os.path.isfile(epub_path_o) or not epub_path_o.endswith(".epub"):
            print(f"File is not EPUB")
            sys.exit()

        pdf_path_o = epub_path_o.replace(".epub", "_converted.pdf")
        if os.path.exists(pdf_path_o):
            print("Output file already exists")
            sys.exit()

        convert_epub_to_pdf(epub_path_o, pdf_path_o)

    else:
        while True:
            # Get input from user
            epub_input = input("Enter the path to the EPUB file: ").strip().replace("\"", "")
            pdf_output = input("Enter the path for the output PDF file: ").strip().replace("\"", "")

            if not os.path.isfile(epub_input) or not epub_input.endswith(".epub") or os.path.exists(pdf_output) or not pdf_output.endswith(".pdf"):
                print(f"Args incorrect")
                continue

            # Convert EPUB to PDF
            convert_epub_to_pdf(epub_input, pdf_output)
