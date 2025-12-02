# PDF Document Processing Pipeline

A Python tool that extracts and processes PDF documents in two stages:

1. **Docling** - Converts PDFs to JSON, Markdown, and extracts page images
2. **PaddleOCR-VL** - Processes extracted images for optical character recognition

## Quick Start

```bash
pip install -r requirements.txt
python pdf_processor.py
```

## What It Does

- Extracts all pages from a PDF as PNG images
- Generates JSON and Markdown versions of the document structure
- Runs OCR on each extracted image
- Saves all outputs to organized directories

## Output

- `output_docling/` - Docling JSON, Markdown files, and page images
- `output_ocr/` - OCR results for each image (JSON and Markdown)

## Key Features

- Completely local (no external APIs)
- Automatic model downloads on first run
- Detailed logging of processing steps
- Organized output structure

## Configuration

Adjust `IMAGE_RESOLUTION_SCALE` in `pdf_processor.py` to control image quality (default: 5.0 = 360 DPI)

## Requirements

- Python 3.8+
- docling, docling-core, paddleocr, Pillow

## Troubleshooting

**Memory issues?** Reduce `IMAGE_RESOLUTION_SCALE` to 2.0 or lower

**First run slow?** PaddleOCR models are downloading (~100MB) - this only happens once

## Resources

- [Docling Documentation](https://github.com/DS4SD/docling)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
