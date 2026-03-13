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
import re

from .rag import HierarchicalRAG, HierarchicalChunk


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
    
    def _identify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Identify user intent from query and extract relevant parameters.
        Returns intent type, keywords, and filters for chunk retrieval.
        Special handling for wall material and thickness queries.
        """
        query_lower = query.lower()
        
        intent = {
            'type': 'general',
            'keywords': [],
            'filters': {},
            'preferred_levels': ['room', 'apartment', 'floor'],
            'query_priority': 0
        }
        
        # Wall-related keywords - comprehensive list
        wall_keywords = {
            'material': ['material', 'materialien', 'wandmaterial', 'baumaterial', 'steinart', 
                        'beton', 'mauerwerk', 'gipskarton', 'stahlbeton', 'ks-l', 'ks-licht',
                        'backstein', 'ziegel', 'holz', 'metall', 'stahl', 'glas', 'stein'],
            'thickness': ['dicke', 'stärke', 'thickness', 'wandstärke', 'wanddicke', 'dick', 'stark',
                         'dünn', 'thin', 'cm', 'millimeter', 'mm'],
            'construction': ['construction', 'konstruktion', 'aufbau', 'schichtaufbau', 'wandaufbau',
                           'schicht', 'layer', 'komposition', 'zusammensetzung'],
            'wall': ['wand', 'wall', 'mauer', 'außenwand', 'innenwand', 'trennwand', 'brandschutz',
                    'schalldämmung', 'wärmeschutz', 'feuerschutz']
        }
        
        # Check for wall-related queries FIRST - highest priority
        is_wall_query = False
        wall_aspects = []
        
        for aspect, keywords in wall_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                is_wall_query = True
                wall_aspects.append(aspect)
        
        if is_wall_query:
            intent['type'] = 'wall_query'
            intent['keywords'] = wall_aspects
            intent['preferred_levels'] = ['room', 'apartment', 'metadata', 'floor']
            intent['query_priority'] = 100  # Highest priority
            intent['filters']['content_sections'] = ['wall_materials', 'wall_thickness', 'wall_details']
        
        # Room/Apartment extraction
        apartment_patterns = [
            r'apartment\s*(\d+)',
            r'apt\s*(\d+)',
            r'wohnung\s*(\d+)',
            r'apartment (\d+)',
            r'apart (\d+)',
            r'apt (\d+)',
            r'in apartment (\d+)',
            r'in wohnung (\d+)'
        ]
        
        for pattern in apartment_patterns:
            apartment_match = re.search(pattern, query_lower, re.IGNORECASE)
            if apartment_match:
                apt_num = int(apartment_match.group(1))
                intent['filters']['apartment_id'] = apt_num
                break
        
        # Material type extraction - look for specific material names
        material_types = [
            'gipskarton', 'mauerwerk', 'stahlbeton', 'beton', 'ks-l', 'ks-licht',
            'brandschutz', 'ziegel', 'backstein', 'holz', 'metall', 'stahl', 'glas'
        ]
        materials_found = []
        for material in material_types:
            if material in query_lower:
                materials_found.append(material)
        
        if materials_found:
            intent['filters']['materials'] = materials_found
            if not is_wall_query:  # If not already tagged as wall query
                intent['type'] = 'material_query'
                intent['query_priority'] = 90
        
        # Floor extraction - various patterns
        floor_patterns = [
            r'(\d+)\s*(?:og|stock|floor)',
            r'(?:floor|etage|stockwerk|geschoss)\s*(\d+)',
            r'(\d+)\.og',
            r'(\d+)\. og',
            r'eg|erdgeschoss|ground',
            r'(\d+)\s*ug'
        ]
        
        for pattern in floor_patterns:
            floor_match = re.search(pattern, query_lower)
            if floor_match:
                if floor_match.group(0).lower() in ['eg', 'erdgeschoss', 'ground']:
                    intent['filters']['floor_id'] = 'EG'
                elif 'ug' in floor_match.group(0).lower():
                    intent['filters']['floor_id'] = f"{floor_match.group(1)}.UG"
                elif floor_match.groups()[0] if floor_match.groups() else None:
                    intent['filters']['floor_id'] = f"{floor_match.group(1)}.OG"
                break
        
        # Room type extraction
        room_types = {
            'kitchen': ['küche', 'kitchen', 'cook'],
            'bathroom': ['bad', 'bathroom', 'wc', 'toilet', 'dusche', 'shower'],
            'bedroom': ['zimmer', 'room', 'schlaf', 'bedroom', 'schlafzimmer'],
            'living': ['wohn', 'living', 'wohnzimmer'],
            'hallway': ['flur', 'hallway', 'corridor', 'korridor']
        }
        
        for room_type, keywords in room_types.items():
            if any(kw in query_lower for kw in keywords):
                if 'room_types' not in intent['filters']:
                    intent['filters']['room_types'] = []
                intent['filters']['room_types'].append(room_type)
        
        # Query type classification
        comparison_keywords = ['compare', 'vergleich', 'vs', 'versus', 'between', 'zwischen', 'difference', 'unterschied']
        aggregation_keywords = ['total', 'sum', 'all', 'count', 'how many', 'combined', 'zusammen', 'gesamt', 'insgesamt']
        specific_keywords = ['what', 'which', 'where', 'was', 'welch', 'wo']
        
        if any(kw in query_lower for kw in comparison_keywords):
            intent['query_type'] = 'comparison'
            intent['query_priority'] = max(intent['query_priority'], 70)
        elif any(kw in query_lower for kw in aggregation_keywords):
            intent['query_type'] = 'aggregation'
            intent['query_priority'] = max(intent['query_priority'], 60)
        else:
            intent['query_type'] = 'specific_lookup'
            intent['query_priority'] = max(intent['query_priority'], 50)
        
        return intent
    
    def _filter_chunks_by_intent(self, chunks: List[HierarchicalChunk], intent: Dict[str, Any]) -> List[HierarchicalChunk]:
        """
        Filter and rank chunks based on query intent.
        Prioritizes relevant hierarchy levels and content for wall queries.
        """
        filtered = []
        
        # Filter by preferred levels
        preferred_levels = intent.get('preferred_levels', [])
        for chunk in chunks:
            if chunk.level in preferred_levels or not preferred_levels:
                filtered.append(chunk)
        
        # Apply apartment filter
        if 'apartment_id' in intent['filters']:
            apt_id = intent['filters']['apartment_id']
            filtered = [c for c in filtered if c.apartment_id == apt_id or c.level in ['metadata', 'building', 'floor']]
        
        # Apply material filter - keep chunks that mention the material
        if 'materials' in intent['filters']:
            materials = intent['filters']['materials']
            material_chunks = []
            for chunk in filtered:
                content_lower = chunk.content.lower()
                if any(mat in content_lower for mat in materials):
                    material_chunks.append(chunk)
            # If we found chunks with materials, use them; otherwise keep all
            filtered = material_chunks if material_chunks else filtered
        
        # Apply floor filter
        if 'floor_id' in intent['filters']:
            floor_id = intent['filters']['floor_id']
            filtered = [c for c in filtered if floor_id in (c.floor_id or '').lower() or c.level in ['metadata', 'building', 'project']]
        
        # Apply room type filter
        if 'room_types' in intent['filters']:
            room_types = intent['filters']['room_types']
            room_type_chunks = []
            for chunk in filtered:
                if chunk.level == 'room':
                    metadata = chunk.metadata or {}
                    room_type_en = (metadata.get('room_type_english', '') or '').lower()
                    room_type_de = (metadata.get('room_type', '') or '').lower()
                    if any(rt in room_type_en or rt in room_type_de for rt in room_types):
                        room_type_chunks.append(chunk)
                else:
                    room_type_chunks.append(chunk)  # Keep non-room chunks for context
            filtered = room_type_chunks if room_type_chunks else filtered
        
        return filtered
    
    def _rank_chunks_for_query(self, chunks: List[HierarchicalChunk], query: str, intent: Dict[str, Any]) -> List[HierarchicalChunk]:
        """
        Rank chunks by relevance to query.
        Prioritizes room-level chunks for wall queries, then apartment, then floor.
        """
        query_lower = query.lower()
        
        def calculate_relevance(chunk: HierarchicalChunk) -> tuple:
            """Calculate relevance score for chunk. Higher score = more relevant."""
            score = 0
            
            # Base hierarchical level ranking for different query types
            if intent.get('type') == 'wall_query':
                # For wall queries: room > apartment > metadata > floor
                level_rank = {
                    'room': 200,        # Highest - contains detailed wall info
                    'apartment': 150,   # Context about which apartment
                    'metadata': 140,    # Standard materials info
                    'floor': 100,
                    'building': 50,
                    'stairs': 30,
                    'aufzug': 30,
                    'project': 10
                }
            else:
                # Default ranking
                level_rank = {
                    'room': 100,
                    'apartment': 80,
                    'floor': 60,
                    'building': 40,
                    'metadata': 50,
                    'stairs': 30,
                    'aufzug': 30,
                    'project': 10
                }
            
            score += level_rank.get(chunk.level, 0)
            
            # Boost for intent keywords in content
            if intent.get('keywords'):
                for keyword in intent.get('keywords', []):
                    if keyword.lower() in chunk.content.lower():
                        score += 50
            
            # Boost for exact material names
            if 'materials' in intent.get('filters', {}):
                for material in intent['filters']['materials']:
                    if material in chunk.content.lower():
                        score += 60
            
            # Boost for query terms in content
            for term in query_lower.split():
                if len(term) > 3 and term in chunk.content.lower():
                    score += 10
            
            # Extra boost for room chunks in wall queries
            if intent.get('type') == 'wall_query' and chunk.level == 'room':
                score += 100
            
            # Specific apartment match
            if 'apartment_id' in intent.get('filters', {}):
                if chunk.apartment_id == intent['filters']['apartment_id']:
                    score += 40
            
            return (score, chunk.chunk_id)
        
        sorted_chunks = sorted(chunks, key=calculate_relevance, reverse=True)
        return sorted_chunks


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
