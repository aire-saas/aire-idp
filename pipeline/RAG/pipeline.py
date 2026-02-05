"""
RAG Pipeline - Main Entry Point
================================

Simple interface for floor plan question answering.

Usage:
    from RAG_pipeline import FloorPlanQA
    
    # Initialize (one time)
    qa = FloorPlanQA()
    
    # Ingest JSONs (one time per project)
    qa.ingest("path/to/floor_plan.json")
    
    # Ask questions
    answer = qa.ask("How many apartments are on 2.OG?")
    print(answer)
"""

from typing import List, Dict, Any, Union
from pathlib import Path

from .rag import HierarchicalRAG


class FloorPlanQA:
    """
    Simple interface for floor plan question answering.
    
    This is the main entry point for the RAG pipeline.
    Input: JSON files from floor plan processing
    Output: Answers to questions about the floor plans
    
    Example:
        qa = FloorPlanQA()
        qa.ingest("fullPDF.json")
        answer = qa.ask("Compare apartments 17 and 21")
    """
    
    def __init__(self, persist_dir: str = None, use_llm: bool = True):
        """
        Initialize the QA system.
        
        Args:
            persist_dir: Optional path to store the vector database.
                        Defaults to ./vector_store in the RAG_pipeline folder.
            use_llm: Whether to use LLM for hierarchy extraction.
                    Set to False to use only regex fallback (faster, no API cost).
        """
        self.rag = HierarchicalRAG(persist_dir=persist_dir, use_llm=use_llm)
    
    def ingest(self, json_paths: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Ingest one or more JSON files into the system.
        
        Args:
            json_paths: Single path or list of paths to JSON files.
            
        Returns:
            Dictionary with ingestion statistics.
        """
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        
        results = {
            "files_processed": 0,
            "total_chunks": 0,
            "projects": [],
            "errors": []
        }
        
        for json_path in json_paths:
            try:
                path = Path(json_path)
                if not path.exists():
                    results["errors"].append(f"File not found: {json_path}")
                    continue
                
                project = self.rag.ingest_full_json(str(path))
                results["files_processed"] += 1
                results["projects"].append(project.project_name)
                
            except Exception as e:
                results["errors"].append(f"Error processing {json_path}: {str(e)}")
        
        stats = self.rag.get_stats()
        results["total_chunks"] = stats["total_chunks"]
        
        return results
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask about the floor plans.
            
        Returns:
            The answer as a string.
        """
        result = self.rag.ask(question)
        return result["answer"]
    
    def ask_detailed(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get detailed response with metadata.
        
        Args:
            question: The question to ask about the floor plans.
            
        Returns:
            Dictionary with answer, chunks used, levels, etc.
        """
        return self.rag.ask(question)
    
    def clear(self):
        """Clear all ingested data."""
        self.rag.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics about ingested data."""
        return self.rag.get_stats()


# Convenience function for one-liner usage
def answer_question(json_path: str, question: str, persist_dir: str = None) -> str:
    """
    One-liner to ingest a JSON and answer a question.
    
    Args:
        json_path: Path to the floor plan JSON file.
        question: The question to ask.
        persist_dir: Optional path for vector store.
        
    Returns:
        The answer as a string.
        
    Example:
        answer = answer_question("fullPDF.json", "Who is the architect?")
    """
    qa = FloorPlanQA(persist_dir=persist_dir)
    qa.ingest(json_path)
    return qa.ask(question)
