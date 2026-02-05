"""
RAG Pipeline for German Floor Plan Q&A
======================================

A complete pipeline for ingesting floor plan JSONs and answering questions.

Quick Start:
    from RAG_pipeline import FloorPlanQA
    
    qa = FloorPlanQA()
    qa.ingest("path/to/floor_plan.json")
    answer = qa.ask("Compare apartments 17 and 21")
    print(answer)

One-liner:
    from RAG_pipeline import answer_question
    answer = answer_question("floor_plan.json", "Who is the architect?")

Module Structure:
    - config.py: API keys, model configuration
    - models.py: Data classes (RoomData, ApartmentData, etc.)
    - extractor.py: Hierarchy extraction from JSON
    - processor.py: JSON → structured data
    - chunker.py: Create chunks and embeddings
    - rag.py: Vector store and retrieval
    - pipeline.py: Simple FloorPlanQA interface
"""

from .pipeline import FloorPlanQA, answer_question
from .rag import HierarchicalRAG
from .models import (
    RoomData,
    StairsData,
    AufzugData,
    ApartmentData,
    FloorData,
    BuildingData,
    ProjectMetadata,
    ProjectData,
    HierarchicalChunk
)
from .processor import HierarchicalProcessor
from .chunker import HierarchicalChunker
from .extractor import HierarchyExtractor

__all__ = [
    # Main interfaces
    "FloorPlanQA",
    "answer_question",
    "HierarchicalRAG",
    
    # Data models
    "RoomData",
    "StairsData", 
    "AufzugData",
    "ApartmentData",
    "FloorData",
    "BuildingData",
    "ProjectMetadata",
    "ProjectData",
    "HierarchicalChunk",
    
    # Processing classes
    "HierarchicalProcessor",
    "HierarchicalChunker",
    "HierarchyExtractor",
]

__version__ = "2.0.0"
