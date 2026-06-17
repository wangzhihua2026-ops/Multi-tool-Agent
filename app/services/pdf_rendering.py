from io import BytesIO

from app.services.ocr import MissingOcrDependencyError


def render_pdf_pages_to_png_bytes(file_bytes: bytes, max_pages: int) -> list[bytes]:
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingOcrDependencyError("pypdfium2") from exc

    pdf = pdfium.PdfDocument(file_bytes)
    try:
        page_count = len(pdf)
        if page_count > max_pages:
            raise ValueError(f"PDF OCR page limit exceeded: {page_count} pages is greater than {max_pages}.")

        rendered_pages: list[bytes] = []
        for page_index in range(page_count):
            page = pdf[page_index]
            bitmap = None
            image = None
            try:
                bitmap = page.render(scale=2)
                image = bitmap.to_pil()
                output = BytesIO()
                image.save(output, format="PNG")
                rendered_pages.append(output.getvalue())
            finally:
                _close_if_possible(image)
                _close_if_possible(bitmap)
                _close_if_possible(page)
        return rendered_pages
    finally:
        _close_if_possible(pdf)


def _close_if_possible(value: object | None) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
