from app.services.ocr import MissingOcrDependencyError, OcrResult, _extract_text_lines_from_paddle_result


def test_extract_text_lines_from_common_paddleocr_shape() -> None:
    result = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("first line", 0.98)],
            [[[0, 2], [1, 2], [1, 3], [0, 3]], ("second line", 0.95)],
        ]
    ]

    assert _extract_text_lines_from_paddle_result(result) == ["first line", "second line"]


def test_extract_text_lines_ignores_low_confidence() -> None:
    result = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("clear text", 0.88)],
            [[[0, 2], [1, 2], [1, 3], [0, 3]], ("uncertain text", 0.31)],
        ]
    ]

    assert _extract_text_lines_from_paddle_result(result, min_confidence=0.5) == ["clear text"]


def test_extract_text_lines_accepts_numeric_string_confidence() -> None:
    result = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("clear text", "0.98")],
            [[[0, 2], [1, 2], [1, 3], [0, 3]], ("uncertain text", "0.31")],
        ]
    ]

    assert _extract_text_lines_from_paddle_result(result, min_confidence=0.5) == ["clear text"]


def test_ocr_result_joins_text_lines() -> None:
    result = OcrResult(lines=["alpha", "beta"], warnings=[])

    assert result.text == "alpha\nbeta"


def test_missing_ocr_dependency_error_message() -> None:
    error = MissingOcrDependencyError("paddleocr")

    assert "OCR support is not installed" in str(error)
    assert "paddleocr" in str(error)
