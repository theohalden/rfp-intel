"""Optional local OCR through the Tesseract executable; no Python OCR wrapper."""
from __future__ import annotations

import shutil
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
import pymupdf

from config import Parsing


def ocr_page(page: pymupdf.Page, options: Parsing) -> str:
    """Render one page and OCR it with a bounded process and temporary files."""
    executable = shutil.which(options.ocr_executable)
    if not executable:
        raise RuntimeError('Tesseract not found; install it or set parsing.ocr_enabled=false')
    with TemporaryDirectory(prefix='rfp-ocr-') as directory:
        image = Path(directory) / 'page.png'
        page.get_pixmap(dpi=options.ocr_dpi, alpha=False).save(image)
        result = subprocess.run(
            [executable, str(image), 'stdout', '-l', options.ocr_language, '--psm', '3'],
            capture_output=True, text=True, encoding='utf-8', check=True,
            timeout=options.ocr_timeout_seconds,
        )
    return result.stdout
