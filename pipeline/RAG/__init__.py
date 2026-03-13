"""
RAG Pipeline for German Floor Plan Q&A
======================================

A complete pipeline for ingesting floor plan JSONs and answering questions.

Features:
- Hierarchical retrieval from project → building → floor → apartment → room
- Wall material and thickness queries
- Multi-apartment comparisons
- Equipment and spatial analysis

Quick Start:
    from RAG_pipeline import FloorPlanQA
    
    qa = FloorPlanQA()
    qa.ingest("path/to/floor_plan.json")
    
    # Ask about spatial layouts
    answer = qa.ask("Compare apartments 17 and 21")
    
    # Ask about wall materials and thickness
    answer = qa.ask("What materials are used in the walls of apartment 17?")
    answer = qa.ask("What is the wall thickness in the bathroom?")
    print(answer)

One-liner:
    from RAG_pipeline import answer_question
    answer = answer_question("floor_plan.json", "Who is the architect?")

Wall Query Examples:
    - "What materials are used in the walls?"
    - "What is the thickness of walls in apartment 17?"
    - "What is the wall construction in the kitchen?"
    - "Compare wall materials between apartment 10 and 15"
    - "Which rooms have gipskarton walls?"
    - "What is the average wall thickness on floor 2.OG?"

Module Structure:
    - config.py: API keys, model configuration
    - models.py: Data classes (RoomData, ApartmentData, etc.)
    - extractor.py: Hierarchy extraction from JSON, wall data extraction
    - processor.py: JSON → structured data, wall data processing
    - chunker.py: Create chunks with wall materials, embeddings
    - rag.py: Vector store and retrieval with wall query detection
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

__version__ = "2.1.0"
