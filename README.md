# Architectural Plan Processing Pipeline

A set of specialized Python tools for processing German architectural floor plans (Baupläne). These tools implement the **Image Division Strategy** to extract structured information from PDF plans.

## The Problem

Processing architectural PDFs with standard OCR tools results in:
- Mixed graphics and text causing confusion
- Unstructured output - all text dumped together
- Tables and sections losing their semantic meaning
- Complex layouts causing information to be ignored or misread

See `problems.md` for detailed problem analysis (from branch-jedou).

## My Solution

I solved these problems by implementing a **three-stage pipeline** that divides images into focused segments before OCR processing. See `solution.md` for the complete technical explanation.

```
┌─────────────────┐
│   PDF Plan      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  plan_splitter  │ ──► │  Floor Plan      │ (graphics - archived)
│                 │     │  Info Panel      │ (text data - processed)
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────┐
│ info_panel_splitter │ ──► │  ARCHITEKT.png   │
│                     │     │  BAUHERR.png     │
│                     │     │  LEGENDE.png     │
│                     │     │  PROJEKT.png     │
│                     │     │  INDEX.png       │
│                     │     │  ...             │
└─────────┬───────────┘     └──────────────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────┐
│  simple_extractor   │ ──► │  plan_data.json  │
│                     │     │  (structured)    │
└─────────────────────┘     └──────────────────┘
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python plan_splitter.py input.pdf -o output/ --preview
python info_panel_splitter.py output/input_info_panel.png -o output/sections/ --preview
python simple_extractor.py output/sections/ "input"
```

---

## Tools

### 1. Plan Splitter (`plan_splitter.py`)

Splits architectural PDF plans into two parts:
- **Left**: Floor plan drawing (graphics)
- **Right**: Info panel (text metadata)

```bash
# Single PDF file
python plan_splitter.py input.pdf -o output/

# With preview showing split line
python plan_splitter.py input.pdf -o output/ --preview

# Process entire directory
python plan_splitter.py input_folder/ -o output/

# Custom DPI (default: 150)
python plan_splitter.py input.pdf --dpi 300
```

**How it works:**
1. Searches for German keywords (ARCHITEKT, BAUHERR, LEGENDE, etc.) in the right portion
2. Finds vertical separator lines using edge detection
3. Snaps to the nearest separator line for clean cuts

**Output:**
- `{filename}_floor_plan.png`
- `{filename}_info_panel.png`
- `{filename}_preview.png` (optional)

---

### 2. Info Panel Splitter (`info_panel_splitter.py`)

Splits the info panel into individual labeled sections.

```bash
# Single info panel
python info_panel_splitter.py output/input_info_panel.png -o output/sections/

# With preview showing detected groups
python info_panel_splitter.py output/input_info_panel.png -o output/sections/ --preview

# Process all info panels in directory
python info_panel_splitter.py output/ -o output/sections/
```

**How it works:**
1. Detects layout structures (tables, graphics, text regions)
2. Finds section dividers (horizontal lines >80% width)
3. Keeps tables as single units
4. Merges title boxes with content below
5. Labels sections using OCR keyword detection

**Output:**
- `{plan}_group_ARCHITEKT.png`
- `{plan}_group_BAUHERR.png`
- `{plan}_group_LEGENDE.png`
- `{plan}_group_PROJEKT.png`
- `{plan}_group_INDEX.png`
- `{plan}_group_WERKPLANUNG.png`
- `{plan}_groups_preview.png` (optional)

**Detected Section Types:**

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

### 3. Simple Extractor (`simple_extractor.py`)

Extracts text from section images using OCR and outputs structured JSON.

```bash
# Extract text from all sections
python simple_extractor.py output/sections/ "plan_name"

# Custom output path
python simple_extractor.py output/sections/ "plan_name" -o results/data.json
```

**How it works:**
1. Preprocesses images (grayscale, contrast enhancement, thresholding)
2. Runs Tesseract OCR with German language support
3. Cleans up text while preserving structure
4. Saves as structured JSON

**Output Format:**
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

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Tesseract OCR

The tools require Tesseract OCR with German language support.

**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to `C:\Program Files\Tesseract-OCR\`
3. Include German language data during installation

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

1. **High-quality PDFs**: Use original PDFs, not scanned copies
2. **DPI Setting**: Increase `--dpi` for small text (300+ recommended)
3. **Preview First**: Use `--preview` flag to verify detection accuracy
4. **OCR Language**: Ensure Tesseract has German (`deu`) language pack

---

## Troubleshooting

**No sections detected?**
- Check if info panel has clear horizontal divider lines
- Try adjusting image contrast before processing

**Wrong split position?**
- Keywords may be positioned differently; check preview image
- Some plans may need manual adjustment

**OCR errors?**
- Ensure Tesseract is installed with German language support
- Try increasing DPI for better image quality

---

## Documentation

- `problems.md` - Problem analysis from branch-jedou
- `solution.md` - How we solved the problems from branch-jedou
