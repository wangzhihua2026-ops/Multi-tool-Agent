from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol


class MissingOcrDependencyError(RuntimeError):
    def __init__(self, dependency: str) -> None:
        super().__init__(
            f"OCR support is not installed. Install the optional dependency {dependency!r} to enable OCR."
        )


@dataclass(frozen=True)
class OcrResult:
    lines: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.strip() for line in self.lines if line.strip()).strip()


class OcrEngine(Protocol):
    def extract_text_from_image(self, image_bytes: bytes) -> OcrResult:
        ...


class PaddleOcrEngine:
    def __init__(self, language: str = "ch") -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise MissingOcrDependencyError("paddleocr") from exc

        self._ocr = PaddleOCR(use_angle_cls=True, lang=language)

    def extract_text_from_image(self, image_bytes: bytes) -> OcrResult:
        try:
            from PIL import Image
        except ImportError as exc:
            raise MissingOcrDependencyError("Pillow") from exc

        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            temp_path: str | None = None
            try:
                with NamedTemporaryFile(suffix=".png", delete=False) as image_file:
                    temp_path = image_file.name
                    image.save(image_file, format="PNG")

                raw_result = self._ocr.ocr(temp_path, cls=True)
            finally:
                if temp_path is not None:
                    Path(temp_path).unlink(missing_ok=True)

        return OcrResult(lines=_extract_text_lines_from_paddle_result(raw_result))


def get_default_ocr_engine() -> OcrEngine:
    return PaddleOcrEngine()


def _extract_text_lines_from_paddle_result(result: Any, min_confidence: float = 0.5) -> list[str]:
    lines: list[str] = []

    def visit(value: Any) -> None:
        if not isinstance(value, (list, tuple)):
            return

        if len(value) >= 2 and _append_text_line(value[1]):
            return

        for item in value:
            visit(item)

    def _append_text_line(value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False

        text, score = value[0], value[1]
        if not isinstance(text, str):
            return False
        try:
            confidence = float(score)
        except (TypeError, ValueError):
            return False
        if confidence < min_confidence:
            return True

        stripped_text = text.strip()
        if stripped_text:
            lines.append(stripped_text)
        return True

    visit(result)
    return lines
