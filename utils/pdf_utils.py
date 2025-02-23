import fitz  # PyMuPDF


def load_pdf_text(pdf_path):
    """
    Extracts text from all pages of a given PDF file.
    
    :param pdf_path: Path to the PDF file
    :return: A list of strings, each representing the text of a page
    """
    doc = fitz.open(pdf_path)
    return [page.get_text("text") for page in doc]
