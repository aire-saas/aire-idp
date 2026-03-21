# AIRE-IDP

## 📋 Project Overview

AIRE-IDP is an advanced Retrieval-Augmented Generation (RAG) system designed to extract, process, and intelligently query architectural floor plan data from PDF documents. The system automatically extracts building information including room layouts, wall materials, thicknesses, and other structural details, then uses LLM-powered retrieval to answer complex architectural queries.

### Key Features

- **PDF to Hierarchical JSON Processing**: Converts floor plan PDFs into structured, hierarchical JSON format
- **Wall Material & Thickness Analysis**: Extracts and tracks wall construction details (material type, thickness)
- **Room Detection & Layout Analysis**: Automatically identifies rooms, boundaries, and spatial relationships
- **Info Panel Extraction**: Extracts metadata from floor plan information panels (project info, dates, scales)
- **Plan Recognition & Prediction**: Detects floor plans and generates enhanced visual representations
- **Multi-Level Data Organization**: Organizes data by pages, floors, rooms, and walls
- **RAG-Powered Query System**: Uses embeddings and LLMs to answer architectural questions
- **Semantic Chunking**: Intelligently chunks data for optimal embedding and retrieval
- **Web-Based UI**: Flask-based interface for PDF upload and interactive querying
- **Production-Ready Architecture**: Modular design with comprehensive testing

---

## 📦 Dependencies

Key packages:
- **llama-index**: RAG framework and document processing
- **openai**: LLM and embedding models
- **groq**: Groq API for alternative LLM
- **chromadb/faiss**: Vector store
- **docling**: Advanced PDF parsing
- **pillow**: Image processing
- **tesseract/easyocr**: OCR for text extraction
- **opencv**: Image analysis
- **flask**: Web UI framework
- **pydantic**: Data validation
- **python-dotenv**: Environment management

---

## 🔧 Installation & Setup

### 1. Prerequisites

- Python 3.11.14
- OpenAI API key
- conda (recommended) or pip

### 2. Environment Setup

```bash
# Clone or navigate to project directory
cd aire-idp

# Create conda environment from requirements
conda create --name aire-idp --file requirements.txt

# Activate environment
conda activate aire-idp

# Or with pip
pip install -r requirements.txt
```

### 3. Configure OpenAI API Key

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your_api_key_here
```

---

## 🚀 Quick Start Guide

### Step 1: Activate the Environment

```bash
# Activate the conda environment
conda activate aire-idp
```

### Step 2: Configure OpenAI API Key

Create a `.env`

**Content**:
```
OPENAI_API_KEY=your_actual_api_key_here
```

⚠️ **IMPORTANT:**
- Replace `your_actual_api_key_here` with your real OpenAI API key

### Step 3: Run the Application

```bash
# From the project root directory
python pipeline/web_ui.py
```

The application will automatically:
- ✅ Load environment variables from `.env` file
- ✅ Verify OpenAI API key is set
- ✅ Initialize Flask server on `http://127.0.0.1:5000`
- ✅ Open your browser automatically
- ✅ Ready to accept PDF uploads and questions

### Verify Everything Works

1. Open browser to `http://127.0.0.1:5000`
2. Upload a floor plan PDF
3. Wait for processing (room detection, wall extraction, etc.)
4. Ask a question like: "What materials are used in the walls?"
5. Get intelligent answers powered by RAG + LLM

---

## 🏗️ Project Architecture

### Complete Data Processing Pipeline Flow

```
PDF Document Upload
    ↓
┌─────────────────────────────────────────┐
│   ArchitecturalPlanPipeline Processing  │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Page-Level Processing          │   │
│  ├─────────────────────────────────┤   │
│  │ • PDF Page Extraction           │   │
│  │ • Image Conversion              │   │
│  └─────────────────────────────────┘   │
│            ↓                            │
│  ┌─────────────────────────────────┐   │
│  │  Info Panel Detection & Extract │   │
│  ├─────────────────────────────────┤   │
│  │ • Panel Localization            │   │
│  │ • OCR Text Extraction           │   │
│  │ • Metadata Parsing              │   │
│  │ • Project Information           │   │
│  └─────────────────────────────────┘   │
│            ↓                            │
│  ┌─────────────────────────────────┐   │
│  │  Plan Detection & Recognition   │   │
│  ├─────────────────────────────────┤   │
│  │ • Floor Plan Localization       │   │
│  │ • Boundary Detection            │   │
│  │ • Plan Enhancement              │   │
│  └─────────────────────────────────┘   │
│            ↓                            │
│  ┌─────────────────────────────────┐   │
│  │  Room Detection & Extraction    │   │
│  ├─────────────────────────────────┤   │
│  │ • Room Boundary Detection       │   │
│  │ • Room Labeling & Naming        │   │
│  │ • Area Calculation              │   │
│  │ • Spatial Relationship Analysis │   │
│  └─────────────────────────────────┘   │
│            ↓                            │
│  ┌─────────────────────────────────┐   │
│  │  Wall Extraction & Analysis     │   │
│  ├─────────────────────────────────┤   │
│  │ • Wall Boundary Detection       │   │
│  │ • Material Recognition          │   │
│  │ • Thickness Measurement         │   │
│  │ • Wall Property Extraction      │   │
│  └─────────────────────────────────┘   │
│            ↓                            │
│  ┌─────────────────────────────────┐   │
│  │  Hierarchical Data Structuring  │   │
│  ├─────────────────────────────────┤   │
│  │ • Floor Organization            │   │
│  │ • Room Grouping                 │   │
│  │ • Wall Association              │   │
│  │ • Metadata Linking              │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
    ↓
Hierarchical JSON Structure
    ↓
┌─────────────────────────────────────────┐
│  Semantic Chunking & Embedding          │
├─────────────────────────────────────────┤
│ • Multi-level chunk creation            │
│ • Metadata tagging                      │
│ • OpenAI embeddings generation          │
│ • Vector store population               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Web UI & RAG Query Engine              │
├─────────────────────────────────────────┤
│ • Flask-based interface                 │
│ • Semantic search & retrieval           │
│ • LLM-powered answer generation         │
│ • Interactive question answering        │
└─────────────────────────────────────────┘
```

### Project Structure

```
aire-idp/
├── pipeline/
│   ├── __init__.py
│   ├── web_ui.py                     # Flask web interface
│   ├── __init__.py (ArchitecturalPlanPipeline)
│   ├── page.py                       # Page class definition
│   │   ├── class Page
│   │   ├── page_number: int
│   │   ├── plan: PIL.Image           # Original floor plan
│   │   ├── predicted_plan: PIL.Image # Enhanced/predicted plan
│   │   ├── info_panel: PIL.Image     # Extracted info panel
│   │   └── rooms: List[Room]         # Detected rooms
│   │
│   ├── detection/
│   │   ├── room_detector.py          # Room detection engine
│   │   ├── info_panel_detector.py    # Info panel extraction
│   │   ├── plan_detector.py          # Floor plan detection
│   │   └── wall_detector.py          # Wall recognition
│   │
│   ├── extraction/
│   │   ├── room_extractor.py         # Room property extraction
│   │   ├── wall_extractor.py         # Wall data extraction
│   │   └── metadata_extractor.py     # Metadata parsing
│   │
│   ├── RAG/
│   │   ├── RAG.py                    # Main RAG class
│   │   ├── data_processing.py        # Hierarchical JSON generation
│   │   ├── embeddings.py             # Embedding generation
│   │   ├── chunking.py               # Semantic chunking
│   │   ├── retrieval.py              # Retrieval engine
│   │   ├── prompts.py                # LLM prompt templates
│   │   ├── test_wall_queries.py      # Test suite
│   │   └── fullPDF (1).json          # Sample data
│   │
│   └── models/
│       ├── room.py                   # Room data model
│       ├── wall.py                   # Wall data model
│       └── page.py                   # Page data model
│
├── templates/
│   └── index.html                    # Web UI frontend
│
├── uploads/                          # Temporary PDF storage
├── requirements.txt                  # Dependencies
├── .env                              # OpenAI API key
└── README.md                         # This file
```

---

## 📊 Data Processing Pipeline

### 1. PDF to Hierarchical JSON Conversion

**File**: `pipeline/RAG/data_processing.py`

The system extracts floor plan data and organizes it hierarchically:

```json
{
  "pages": [
    {
      "page_number": 1,
      "rooms": [
        {
          "room_id": "room_001",
          "room_name": "Living Room",
          "floor": "Ground Floor",
          "area": 45.5,
          "walls": {
            "wall_001": {
              "detected_material": "Mauerwerk KS-L 12",
              "wall_thickness": 0.24,
              "wall_length": 5.2,
              "position": "north",
              "orientation": "horizontal"
            },
            "wall_002": {
              "detected_material": "gipskarton",
              "wall_thickness": 0.12,
              "wall_length": 4.8,
              "position": "east",
              "orientation": "vertical"
            }
          }
        }
      ],
      "infoPanel_Information": {
        "metadata": {
          "projektname": "Project Name",
          "date": "2024-01-15"
        }
      }
    }
  ]
}
```

### 2. Wall Material Detection

The system recognizes and extracts:
- **Common Materials**: Mauerwerk (masonry), gipskarton (drywall), Beton (concrete)
- **Thickness**: Measured in meters (stored as decimal)
- **Position**: north, south, east, west
- **Structural Purpose**: Load-bearing, partition, external, etc.

---

