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

**OCR run is very slow around 5-10 min** 

## Resources

- [Docling Documentation](https://github.com/DS4SD/docling)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)

---

# Architectural Plan Processing Tools

A set of specialized tools for processing German architectural floor plans (Baupläne). These tools work together as a pipeline to extract structured information from PDF plans.

## Pipeline Overview

```
PDF Plan → [plan_splitter] → Floor Plan + Info Panel
                                    ↓
                            [info_panel_splitter] → Section Images (ARCHITEKT, BAUHERR, LEGENDE, etc.)
                                    ↓
                            [simple_extractor] → JSON with extracted text per section
```

---

## 1. Plan Splitter (`plan_splitter.py`)

Splits architectural PDF plans into two parts:
- **Left**: The actual floor plan drawing
- **Right**: Info panel (legend, architect info, project details, etc.)

### Usage

```bash
# Single PDF file
python plan_splitter.py input.pdf -o output/

# With preview image showing split line
python plan_splitter.py input.pdf -o output/ --preview

# Process entire directory
python plan_splitter.py input_folder/ -o output/

# Custom DPI for higher quality (default: 150)
python plan_splitter.py input.pdf --dpi 300
```

### How It Works

1. **Keyword Detection**: Searches for German architectural keywords (ARCHITEKT, BAUHERR, LEGENDE, PROJEKT, etc.) in the right portion of the document
2. **Line Detection**: Finds vertical separator lines near the keywords using edge detection
3. **Smart Splitting**: Snaps to the nearest separator line to ensure clean cuts

### Output

- `{filename}_floor_plan.png` - The floor plan drawing
- `{filename}_info_panel.png` - The info panel
- `{filename}_preview.png` - (optional) Preview showing the detected split line

### Supported Keywords

Primary: `ARCHITEKT`, `BAUHERR`, `LEGENDE`, `PROJEKT`, `WERKPLANUNG`

Secondary: `GEMARKUNG`, `MASSTAB`, `DATUM`, `INDEX`, `PLANINHALT`, etc.

---

## 2. Info Panel Splitter (`info_panel_splitter.py`)

Splits the info panel image into individual group sections based on visual layout detection.

### Usage

```bash
# Single info panel image
python info_panel_splitter.py output/input_info_panel.png -o output/groups/

# With preview showing detected groups
python info_panel_splitter.py output/input_info_panel.png -o output/groups/ --preview

# Process all info panels in a directory
python info_panel_splitter.py output/ -o output/groups/
```

### How It Works

1. **Layout Detection**: Identifies tables, graphics, and text regions
2. **Horizontal Line Analysis**: Finds section dividers (strong horizontal lines spanning >80% width)
3. **Table Region Handling**: Keeps tables as single units (doesn't split table rows)
4. **Title Merging**: Automatically merges title boxes with their content below
5. **Vertical Splitting**: Splits side-by-side sections when both have substantial content
6. **Keyword Identification**: Labels sections using OCR (ARCHITEKT, BAUHERR, LEGENDE, etc.)

### Output

Individual PNG files for each detected section:
- `{plan_name}_group_ARCHITEKT.png`
- `{plan_name}_group_BAUHERR.png`
- `{plan_name}_group_LEGENDE.png`
- `{plan_name}_group_PROJEKT.png`
- `{plan_name}_group_INDEX.png`
- `{plan_name}_group_WERKPLANUNG.png`
- `{plan_name}_group_UNKNOWN.png` (unidentified sections)
- `{plan_name}_groups_preview.png` (optional preview with colored boxes)

### Detected Section Types

| Section | Description |
|---------|-------------|
| ARCHITEKT | Architect information |
| BAUHERR | Client/Building owner |
| LEGENDE | Legend/Symbol explanations |
| PROJEKT | Project name and details |
| INDEX | Revision history |
| WERKPLANUNG | Work planning details |
| LAGEPLAN | Site plan reference |
| PLANINFO | Plan metadata |
| BEZEICHNUNG | Designation/Title |
| DATUM | Date information |
| AKTENZEICHEN | File reference number |

---

## 3. Simple Extractor (`simple_extractor.py`)

Extracts raw text from section images using OCR and outputs clean JSON.

### Usage

```bash
# Extract text from all sections of a plan
python simple_extractor.py output/groups/ "plan_name"

# Custom output path
python simple_extractor.py output/groups/ "plan_name" -o results/plan_data.json
```

### How It Works

1. **Image Preprocessing**: Converts to grayscale, enhances contrast, applies thresholding
2. **OCR**: Uses Tesseract with German language support (`deu`)
3. **Text Cleanup**: Removes excessive whitespace while preserving structure
4. **JSON Output**: Saves structured data with section names as keys

### Output Format

```json
{
  "plan_name": "input1",
  "extraction_date": "2025-12-04T12:00:00.000000",
  "sections": {
    "ARCHITEKT": "BAUERLE architekten+brandschutz\nHeinrich-Barth-Straße 20\nD-66115 Saarbrücken\nTel: +49(681) 094 931-0",
    "BAUHERR": "THOR Zweite GmbH & Co. KG\nHerr Thomas Schulze-Wischeler\nUlmenstraße 22\n60325 Frankfurt am Main",
    "LEGENDE": "Kalksandsteinmauerwerk\nOrtbeton bewehrt\n...",
    "PROJEKT": "206-18_WP_Florapark_SB\nMeerwiesertalweg 4a, 4b, 4c\n66123 Saarbrücken"
  }
}
```

---

## Complete Pipeline Example

```bash
# Step 1: Split PDF into floor plan and info panel
python plan_splitter.py plans/bauplan.pdf -o output/ --preview

# Step 2: Split info panel into sections
python info_panel_splitter.py output/bauplan_info_panel.png -o output/sections/ --preview

# Step 3: Extract text from sections
python simple_extractor.py output/sections/ "bauplan"

# Result: output/sections/bauplan_simple.json
```

---

## Requirements

### Additional Dependencies

```bash
pip install opencv-python numpy Pillow PyMuPDF pytesseract
```

### Tesseract OCR

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to `C:\Program Files\Tesseract-OCR\`
3. Add German language data during installation

**Linux:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-deu
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

---

## Tips for Best Results

1. **High-quality PDFs**: Use original PDFs, not scanned copies when possible
2. **DPI Setting**: Increase `--dpi` for small text (300+ recommended for detailed plans)
3. **Preview First**: Always use `--preview` flag first to verify detection accuracy
4. **OCR Language**: German (`deu`) is used by default; ensure Tesseract has German language pack installed

## Troubleshooting

**No sections detected?**
- Check if the info panel has clear horizontal divider lines
- Try adjusting image contrast/brightness before processing

**Wrong split position?**
- The keywords might be positioned differently; check the preview image
- Some plans may require manual adjustment

**OCR errors?**
- Ensure Tesseract is properly installed with German language support
- Try increasing DPI for better image quality
- Poor scan quality may result in OCR errors