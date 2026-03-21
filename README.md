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
- **chromadb/faiss**: Vector storefff
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

## 🔍 Core Components Deep Dive

### 1. Web UI Entry Point (`pipeline/web_ui.py`)

The Flask application provides the user interface and orchestrates the entire pipeline:

```python
# Workflow in web_ui.py
┌─────────────────────────────────────────┐
│  User uploads PDF via web_ui.py         │
└──────────────┬──────────────────────────┘
               │
               ↓
   @app.route("/upload", methods=["POST"])
               │
               ├─→ session_id = uuid.uuid4()
               ├─→ file.save(filepath)
               │
               ↓
   pipeline = ArchitecturalPlanPipeline(pdf_path=filepath)
               │
               ├─→ Processes all pages
               ├─→ Detects rooms, panels, plans
               ├─→ Extracts walls & materials
               │
               ↓
   qa = pipeline.run()
               │
               ├─→ Creates RAG engine
               ├─→ Generates embeddings
               ├─→ Populates vector store
               │
               ↓
   qa_engines[session_id] = qa
   pdfDatas[session_id] = pages_data
               │
               ↓
   Return JSON: {pages, rooms, images, session_id}
               │
               ↓
   Frontend displays floor plan images
   & enables question answering
```

**Key Data Structures:**
```python
pdfDatas[session_id] = {
    "success": True,
    "pages": [
        {
            "pageno": 1,
            "has_info_panel": True,
            "has_plan": True,
            "has_predicted_plan": True,
            "plan_base64": "...",           # Display image
            "info_panel_base64": "...",     # Extracted panel
        },
        # ... more pages
    ],
    "total_pages": 5,
    "rooms": [Room, Room, ...]             # Detected rooms
}

qa_engines[session_id] = {
    "qa": FloorPlanQA(),  # RAG engine for querying
    "last_access": datetime
}
```

---

### 2. Page Structure & Processing (`pipeline/page.py`)

Each PDF page is represented as a `Page` object with multiple processing outputs:

```python
class Page:
    page_number: int              # 1-indexed page number
    
    # Original & Enhanced Images
    plan: PIL.Image              # Original floor plan image
    predicted_plan: PIL.Image    # Enhanced/reconstructed plan
    
    # Extracted Information
    info_panel: PIL.Image        # Metadata panel (separate image)
    rooms: List[Room]            # Detected rooms on this page
    
    # Extracted Data
    rooms_data: Dict             # Room details
    walls_data: Dict             # Wall details
    metadata: Dict               # Page metadata from info panel
```

---

### 3. Info Panel Detection & Extraction (`pipeline/detection/info_panel_detector.py`)

**Purpose:** Extract metadata information from floor plan documents

**What it detects:**
- Project name/title
- Date of creation
- Architect/Designer information
- Scale information
- Legend/Symbols explanation
- Floor descriptions
- Room lists with areas
- Revision information

**Processing Steps:**

```
Input: PDF Page Image
    ↓
1. Panel Localization
   - Detect information panel region
   - Separate from floor plan area
   - Extract panel boundaries
    ↓
2. Region Segmentation
   - Identify text regions
   - Identify table/legend regions
   - Separate metadata from graphics
    ↓
3. OCR Text Extraction
   - Extract raw text using Tesseract/EasyOCR
   - Clean and normalize text
   - Handle multi-language content
    ↓
4. Structured Parsing
   - Project information extraction
   - Date/revision parsing
   - Scale factor identification
   - Room list parsing (if present)
    ↓
5. Metadata Assembly
Output: {
    "projektname": "Project Name",
    "date": "2024-01-15",
    "architect": "Name",
    "scale": "1:100",
    "rooms": ["Room1", "Room2"],
    "area_total": 125.5,
    "legend": {...}
}
```

**Output Example:**
```json
{
  "infoPanel_Information": {
    "metadata": {
      "projektname": "Residential Complex Alpha",
      "date": "2024-01-15",
      "architect": "John Doe Architects",
      "scale": "1:100",
      "total_area": 450.75,
      "rooms_list": [
        {"name": "Living Room", "area": 45.5},
        {"name": "Bedroom 1", "area": 32.0},
        {"name": "Kitchen", "area": 25.0}
      ],
      "revision": "A",
      "legend": {
        "door": "D",
        "window": "W",
        "wall": "solid line"
      }
    }
  }
}
```

---

### 4. Plan Detection & Recognition (`pipeline/detection/plan_detector.py`)

**Purpose:** Identify and extract the floor plan from the PDF page

**What it does:**
- Locates the floor plan area within the page
- Separates plan from other elements (title, legend, margins)
- Detects and corrects plan orientation
- Generates enhanced/predicted plan representation

**Processing Steps:**

```
Input: PDF Page Image
    ↓
1. Content Region Detection
   - Find non-white regions
   - Identify main plan area
   - Separate from margins/borders
    ↓
2. Plan Boundary Detection
   - Detect outermost walls
   - Identify plan corners
   - Extract bounding box
    ↓
3. Image Enhancement
   - Contrast adjustment
   - Noise reduction
   - Line detection and strengthening
    ↓
4. Orientation Detection
   - Detect page orientation
   - Correct if rotated
   - Ensure north-up orientation
    ↓
5. Plan Prediction/Reconstruction
   - Use ML models to enhance lines
   - Fill missing segments
   - Generate predicted_plan
    ↓
Output: {
    "plan": original_image,
    "predicted_plan": enhanced_image,
    "boundaries": {x, y, width, height},
    "orientation": "north_up"
}
```

**Detected Plan Features:**
- Wall lines (load-bearing and partition)
- Door openings
- Window placements
- Stairs/ramps
- Special features (fixtures, utilities)

---

### 5. Room Detection & Extraction (`pipeline/detection/room_detector.py`)

**Purpose:** Identify and extract individual rooms from the floor plan

**What it detects:**
- Room boundaries (enclosed spaces)
- Room shapes and areas
- Door/window openings
- Room purposes/types
- Spatial relationships

**Processing Steps:**

```
Input: Extracted Floor Plan Image
    ↓
1. Wall Line Detection
   - Apply edge detection (Canny, Sobel)
   - Extract continuous lines
   - Strengthen wall segments
   - Create wall map/skeleton
    ↓
2. Enclosed Space Detection
   - Flood fill algorithm
   - Connected component analysis
   - Identify closed regions
   - Filter by minimum area (noise removal)
    ↓
3. Room Segmentation
   - Extract each enclosed region
   - Compute centroid
   - Calculate area
   - Identify polygon vertices
    ↓
4. Room Classification
   - Detect doors (openings)
   - Detect windows
   - Infer room type:
     * Living areas (larger, multiple windows)
     * Bedrooms (medium, doors, windows)
     * Bathrooms (small, fixtures)
     * Kitchens (specific layout, fixtures)
    ↓
5. Room Labeling
   - Assign room IDs (room_001, room_002, ...)
   - Extract room names (if present in plan)
   - Default naming if not found
    ↓
Output: List[Room] = [
    Room(id="room_001", name="Living Room", area=45.5, walls=[...]),
    Room(id="room_002", name="Bedroom", area=32.0, walls=[...]),
    ...
]
```

**Room Data Structure:**
```python
class Room:
    room_id: str                    # "room_001"
    room_name: str                  # "Living Room"
    floor: str                      # "Ground Floor"
    area: float                     # 45.5 m²
    
    walls: List[Wall]               # Room walls
    doors: List[Door]               # Door openings
    windows: List[Window]           # Window openings
    
    coordinates: List[Tuple[x,y]]   # Room polygon vertices
    centroid: Tuple[x,y]            # Room center point
    
    room_type: str                  # "living", "bedroom", etc.
    features: Dict                  # Additional properties
```

---

### 6. Wall Extraction & Analysis (`pipeline/extraction/wall_extractor.py`)

**Purpose:** Extract and analyze wall properties including material and thickness

**What it extracts:**
- Wall boundaries and positions
- Wall materials (Mauerwerk, gipskarton, Beton, etc.)
- Wall thickness measurements
- Wall length calculations
- Structural properties

**Processing Steps:**

```
Input: Room Data + Floor Plan Image
    ↓
1. Wall Boundary Detection
   - Identify walls from room polygons
   - Extract wall segments
   - Calculate wall lengths
    ↓
2. Wall Material Recognition
   - Analyze wall patterns/textures
   - OCR detection from legend
   - Cross-reference with standards:
     * Mauerwerk KS-L 12 → 0.24m
     * gipskarton → 0.12m
     * Beton → 0.20-0.30m
   - Pattern matching
    ↓
3. Thickness Measurement
   - Extract from dimension lines
   - OCR measurement text
   - Calculate from pixel ratios
   - Validate against standards
    ↓
4. Wall Property Assignment
   - Position (north/south/east/west)
   - Orientation (horizontal/vertical)
   - Load-bearing status
   - External vs internal
    ↓
Output: {
    "walls": {
        "wall_001": {
            "detected_material": "Mauerwerk KS-L 12",
            "wall_thickness": 0.24,
            "wall_length": 5.2,
            "position": "north",
            "orientation": "horizontal",
            "load_bearing": true
        },
        ...
    }
}
```

---

## 📊 Complete JSON Hierarchical Structure

After all detections and extractions, the pipeline generates a comprehensive hierarchical JSON:

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
          "room_type": "living",
          "walls": {
            "wall_001": {
              "detected_material": "Mauerwerk KS-L 12",
              "wall_thickness": 0.24,
              "wall_length": 5.2,
              "position": "north",
              "orientation": "horizontal",
              "load_bearing": true
            },
            "wall_002": {
              "detected_material": "gipskarton",
              "wall_thickness": 0.12,
              "wall_length": 4.8,
              "position": "east",
              "orientation": "vertical",
              "load_bearing": false
            }
          },
          "doors": [
            {
              "door_id": "door_001",
              "position": "east_wall",
              "width": 0.9,
              "type": "single_swing"
            }
          ],
          "windows": [
            {
              "window_id": "window_001",
              "position": "north_wall",
              "width": 1.2,
              "height": 1.5,
              "type": "double_hung"
            }
          ]
        }
      ],
      "infoPanel_Information": {
        "metadata": {
          "projektname": "Residential Complex Alpha",
          "date": "2024-01-15",
          "architect": "John Doe Architects",
          "scale": "1:100",
          "total_area": 450.75
        }
      }
    }
  ]
}
```

---

## 🔍 RAG Query System

### 1. Main RAG Class (`pipeline/RAG/RAG.py`)

```python
from pipeline.RAG import FloorPlanQA

# Initialize
qa = FloorPlanQA(use_llm=True)

# Ingest floor plan data (called by pipeline.run())
result = qa.ingest('path/to/fullPDF (1).json')

# Query the system
answer = qa.ask("What materials are used in the walls?")
detailed = qa.ask_detailed("What is the wall thickness in the kitchen?")
```

### 2. Query Types Supported

#### Wall Material Queries
- "What materials are used in the walls?"
- "What wall materials are in the living room?"
- "Which walls are made of Mauerwerk KS-L 12?"
- "What is the difference in materials between bedroom walls?"

#### Wall Thickness Queries
- "What is the thickness of the walls in the kitchen?"
- "Compare wall thicknesses between different rooms"
- "Which room has the thickest walls?"
- "What is the wall thickness in the bathroom?"

#### Room-Specific Structural Analysis
- "Describe all walls in the living room"
- "What materials separate the kitchen from the dining area?"
- "Tell me about wall construction in the master bedroom"

#### Combined Queries
- "List all rooms with their wall materials and thicknesses"
- "Which walls are load-bearing and made of Mauerwerk?"
- "Compare the construction of external vs internal walls"

### 3. Semantic Chunking Strategy (`pipeline/RAG/chunking.py`)

Data is chunked at multiple hierarchical levels:

```
Original Hierarchical Data
    ↓
1. Floor-Level Chunks
   - Chunk: "Ground Floor has 5 rooms..."
   - Metadata: {level: "floor", page: 1}
    ↓
2. Room-Level Chunks
   - Chunk: "Living Room (45.5m²) has walls made of..."
   - Metadata: {level: "room", room_id: "room_001"}
    ↓
3. Wall-Specific Chunks
   - Chunk: "North wall of Living Room: Mauerwerk KS-L 12, 0.24m thick..."
   - Metadata: {level: "wall", room_id: "room_001", wall_id: "wall_001"}
    ↓
4. Property Chunks
   - Chunk: "Material specifications: Mauerwerk = 0.24m, gipskarton = 0.12m"
   - Metadata: {level: "property", type: "material_reference"}
    ↓
Vector Store Chunks ← All chunks embedded and stored
```

### 4. Embedding & Retrieval (`pipeline/RAG/embeddings.py`, `pipeline/RAG/retrieval.py`)

```
User Question: "What materials are in the walls?"
    ↓
1. Query Embedding
   - Convert to vector using OpenAI embeddings
    ↓
2. Semantic Search
   - Find similar chunks in vector store
   - Compute similarity scores
   - Return top-K chunks (K=5-10)
    ↓
3. Re-ranking & Filtering
   - Score by relevance
   - Filter by metadata (e.g., only "wall" level)
   - Sort by confidence
    ↓
4. Context Assembly
   - Combine top chunks
   - Preserve hierarchy information
   - Format for LLM
    ↓
5. LLM Answer Generation
   - Feed context + question to Groq LLM
   - Generate comprehensive answer
   - Format response
    ↓
Return: "The walls are constructed using..."
```

### 5. LLM Prompting (`pipeline/RAG/prompts.py`)

System prompts are specialized for architectural analysis:

```python
SYSTEM_PROMPT = """
You are an expert architectural analyst specializing in floor plans, 
building materials, and structural design.

When answering questions about walls:
1. Be specific with material names and measurements
2. Provide measurements in meters and centimeters
3. Explain the structural implications
4. Reference room names and locations
5. Compare materials when relevant

Use the provided floor plan data as your primary source of truth.
"""

WALL_MATERIAL_PROMPT = """
Based on the floor plan data provided:
{context}

Answer this question about wall materials:
{question}

Provide a detailed response that includes:
- Specific material names
- Room locations
- Thickness measurements
- Any structural significance
"""

WALL_THICKNESS_PROMPT = """
Based on the floor plan data provided:
{context}

Answer this question about wall thickness:
{question}

Format your response as:
- Room name: Wall position - Material - Thickness
- Compare thicknesses between rooms if relevant
- Explain structural implications if applicable
"""
```

---

## 💻 Web UI Workflow (`pipeline/web_ui.py`)

```
┌────────────────────────────────────────────────────────────────┐
│  User Interface Workflow                                       │
└────────────────────────────────────────────────────────────────┘

1. USER UPLOADS PDF
   Browser → /upload (POST)
        ↓
   └─→ File validation
   └─→ Secure filename
   └─→ Save to uploads folder
        ↓
2. PIPELINE PROCESSING
   ArchitecturalPlanPipeline(pdf_path)
        ↓
   ├─→ PDF → Pages (page extraction)
   ├─→ Info Panel Detection → page.info_panel
   ├─→ Plan Detection → page.plan, page.predicted_plan
   ├─→ Room Detection → page.rooms
   ├─→ Wall Extraction → walls data
   ├─→ Hierarchical JSON → structured data
        ↓
3. RAG ENGINE CREATION
   pipeline.run()
        ↓
   ├─→ Create FloorPlanQA instance
   ├─→ Ingest hierarchical JSON
   ├─→ Generate semantic chunks
   ├─→ Create embeddings
   ├─→ Populate vector store
        ↓
4. SESSION STORAGE
   qa_engines[session_id] = qa
   pdfDatas[session_id] = {pages, rooms, images}
        ↓
5. RESPONSE TO FRONTEND
   Return JSON:
   {
     "success": true,
     "pages": [...],
     "total_pages": 5,
     "rooms": [...],
     "session_id": "uuid"
   }
        ↓
6. USER ASKS QUESTION
   Browser → /ask (POST)
   {"question": "What materials are in the walls?"}
        ↓
7. RAG QUERY EXECUTION
   qa.ask(question)
        ↓
   ├─→ Question embedding
   ├─→ Semantic search in vector store
   ├─→ Retrieve relevant chunks
   ├─→ LLM answer generation
   ├─→ Post-process response
        ↓
8. RESPONSE TO USER
   Return: {"success": true, "answer": "..."}
        ↓
   Browser displays answer

9. SESSION MANAGEMENT
   └─→ Touch session timestamp
   └─→ Background cleanup of expired sessions (30 min TTL)
```

---

## 🧪 Testing & Validation

### Running Wall Query Tests

```bash
python pipeline/RAG/test_wall_queries.py
```

### Test Workflow

```
Test Execution Flow:
    ↓
1. Validate JSON Structure
   └─→ Check pages exist
   └─→ Check rooms exist
   └─→ Check walls data present
   └─→ Validate required fields
    ↓
2. Initialize RAG System
   └─→ Load FloorPlanQA
   └─→ Verify embeddings model
   └─→ Check vector store
    ↓
3. Ingest Data
   └─→ Load sample JSON
   └─→ Create chunks
   └─→ Generate embeddings
   └─→ Store vectors
    ↓
4. Execute Test Queries
   └─→ Wall material questions
   └─→ Wall thickness questions
   └─→ Room-specific questions
   └─→ Structural analysis questions
    ↓
5. Validate Responses
   └─→ Answer quality
   └─→ Relevance scoring
   └─→ Chunk retrieval verification
    ↓
Report: All tests passed ✅
```

---

## 📈 Usage Examples

### Example 1: Complete PDF Analysis Workflow

```python
from pipeline import ArchitecturalPlanPipeline

# Initialize pipeline with PDF
pipeline = ArchitecturalPlanPipeline(pdf_path="floor_plan.pdf")

# Process all pages
pages = pipeline.pages_content  # List of Page objects

for page in pages:
    print(f"Page {page.page_number}:")
    
    # Access detected info panel
    if page.info_panel:
        print(f"  Project: {page.metadata['projektname']}")
        print(f"  Date: {page.metadata['date']}")
    
    # Access detected plan
    if page.predicted_plan:
        print(f"  Plan detected and enhanced")
    
    # Access detected rooms
    for room in page.rooms:
        print(f"  Room: {room.room_name}")
        print(f"    Area: {room.area} m²")
        print(f"    Walls:")
        for wall_id, wall_data in room.walls.items():
            print(f"      {wall_id}: {wall_data['detected_material']} "
                  f"({wall_data['wall_thickness']}m)")

# Create RAG engine
qa = pipeline.run()

# Query using RAG
answer = qa.ask("What materials are used in the walls?")
print(answer)
```

### Example 2: Room-Specific Analysis

```python
# Query specific room
answer = qa.ask_detailed(
    "What is the wall thickness in the kitchen? List each wall separately."
)

print(f"Chunks retrieved: {answer['chunks_retrieved']}")
print(f"Relevance score: {answer['confidence']}")
print(f"Answer: {answer['answer']}")
```

### Example 3: Material Comparison

```python
# Comparative analysis
answer = qa.ask_detailed(
    "Compare wall materials and thicknesses between "
    "the living room and master bedroom"
)
```

### Example 4: Web UI Integration

```python
# When user uploads PDF via web_ui.py:

# 1. PDF uploaded
# 2. ArchitecturalPlanPipeline processes it
# 3. All detection/extraction happens automatically
# 4. RAG engine created
# 5. Frontend displays:
#    - Floor plan image (predicted_plan)
#    - Info panel (with project metadata)
#    - Room list (detected rooms)
#    - Chat interface (for questions)

# User asks: "What are the wall materials?"
# → RAG retrieves relevant chunks about walls
# → LLM generates comprehensive answer
# → Answer displayed in UI
```

---

## 🔑 Key Components Summary

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **Info Panel Detector** | Extract metadata | Floor plan image | Project info, dates, scales |
| **Plan Detector** | Identify & enhance floor plan | PDF page image | plan, predicted_plan |
| **Room Detector** | Find enclosed spaces | Floor plan image | List of Room objects |
| **Wall Extractor** | Extract wall properties | Room data + image | Wall materials & thickness |
| **Semantic Chunker** | Prepare for embeddings | Hierarchical JSON | Multi-level text chunks |
| **Embeddings** | Convert to vectors | Text chunks | Vector embeddings |
| **Retrieval** | Find relevant chunks | User question + vectors | Top-K similar chunks |
| **LLM Prompter** | Generate answers | Context + question | Comprehensive answers |
| **Web UI** | User interface | PDF files | Images + chat interface |

---