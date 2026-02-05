"""
Hierarchical RAG - Vector store and retrieval.

Main RAG system with hierarchical chunk retrieval for floor plan Q&A.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI
import chromadb
from chromadb.config import Settings

from .config import OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, COLLECTION_NAME, DEFAULT_PERSIST_DIR
from .models import ProjectData, HierarchicalChunk
from .processor import HierarchicalProcessor
from .chunker import HierarchicalChunker


class HierarchicalRAG:
    """
    Main RAG system with hierarchical chunk retrieval.
    
    Usage:
        rag = HierarchicalRAG(persist_dir="./vector_store")
        rag.ingest_full_json("path/to/floor_plan.json")
        result = rag.ask("Compare apartments 17 and 21")
        print(result["answer"])
    """
    
    def __init__(self, persist_dir: str = None, use_llm: bool = True):
        """
        Initialize the RAG system.
        
        Args:
            persist_dir: Directory to store the vector database. 
                        Defaults to ./vector_store in the module directory.
            use_llm: Whether to use LLM for hierarchy extraction.
                    Set to False to use only regex fallback (faster, no API cost).
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.use_llm = use_llm
        
        if persist_dir is None:
            persist_dir = str(Path(__file__).parent / DEFAULT_PERSIST_DIR)
        
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
        except:
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Hierarchical floor plan chunks v2"}
            )
        
        self.processor = HierarchicalProcessor(use_llm=use_llm)
        self.chunker = HierarchicalChunker()
        self.chunk_graph: Dict[str, HierarchicalChunk] = {}
        
        self._rebuild_chunk_graph()
    
    def _rebuild_chunk_graph(self):
        """Rebuild chunk graph from ChromaDB."""
        if self.collection.count() == 0:
            return
        
        results = self.collection.get(include=["metadatas", "documents"])
        
        for i, chunk_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            content = results['documents'][i]
            
            chunk = HierarchicalChunk(
                chunk_id=chunk_id,
                level=metadata.get('level', 'unknown'),
                content=content,
                project_id=metadata.get('project_id'),
                building_id=metadata.get('building_id'),
                floor_id=metadata.get('floor_id'),
                apartment_id=metadata.get('apartment_id'),
                room_id=metadata.get('room_id'),
                stairs_id=metadata.get('stairs_id'),
                aufzug_id=metadata.get('aufzug_id'),
                parent_chunk_id=metadata.get('parent_chunk_id'),
                child_chunk_ids=json.loads(metadata.get('child_chunk_ids', '[]')),
                sibling_chunk_ids=json.loads(metadata.get('sibling_chunk_ids', '[]')),
                metadata=metadata
            )
            self.chunk_graph[chunk_id] = chunk
    
    def ingest_full_json(self, json_path: str) -> ProjectData:
        """
        Ingest a full PDF JSON file with floor plan data.
        
        Args:
            json_path: Path to the JSON file containing floor plan data.
            
        Returns:
            ProjectData object with parsed hierarchy.
        """
        project = self.processor.process_full_json(json_path)
        
        chunks = self.chunker.create_all_chunks(project)
        chunks = self.chunker.generate_embeddings(chunks)
        
        for chunk in chunks:
            self.chunk_graph[chunk.chunk_id] = chunk
            
            metadata = {
                "level": chunk.level,
                "project_id": chunk.project_id or "",
                "building_id": chunk.building_id or "",
                "floor_id": chunk.floor_id or "",
                "apartment_id": chunk.apartment_id if chunk.apartment_id else "",
                "room_id": chunk.room_id if chunk.room_id is not None else "",
                "stairs_id": chunk.stairs_id if chunk.stairs_id is not None else "",
                "aufzug_id": chunk.aufzug_id if chunk.aufzug_id is not None else "",
                "parent_chunk_id": chunk.parent_chunk_id or "",
                "child_chunk_ids": json.dumps(chunk.child_chunk_ids),
                "sibling_chunk_ids": json.dumps(chunk.sibling_chunk_ids),
            }
            metadata.update({k: str(v) if not isinstance(v, (str, int, float, bool)) else v 
                           for k, v in chunk.metadata.items()})
            
            self.collection.add(
                ids=[chunk.chunk_id],
                embeddings=[chunk.embedding],
                documents=[chunk.content],
                metadatas=[metadata]
            )
        
        return project
    
    def _analyze_question_scope(self, question: str) -> Dict[str, Any]:
        """Use LLM to analyze what level of the hierarchy the question targets."""
        prompt = f"""Analyze this question about a German floor plan/building project.
Determine what level of detail and which entities are being asked about.

Question: {question}

Return a JSON object with:
{{
    "target_level": "<primary level: project, metadata, building, floor, apartment, stairs, aufzug, room>",
    "specific_project": "<project name if mentioned, else null>",
    "specific_building": "<building ID like 'H1' if mentioned, else null>",
    "specific_floor": "<floor like '2.OG' if mentioned, else null>",
    "specific_apartment": <apartment number(s) as int or array if comparing multiple, else null>,
    "room_type_filter": "<room type like 'Bad', 'Küche' if asking about specific room types, else null>",
    "query_type": "<one of: overview, comparison, specific_lookup, aggregation, metadata_lookup>",
    "estimated_chunks_needed": <1-20 based on query scope>,
    "needs_metadata": <true if asking about architect, bauherr, materials, legend, changes>,
    "needs_hierarchy_traversal": <true if needs parent/child context>
}}

Query type examples:
- "overview": What apartments are in building H1?
- "comparison": Compare apartments 17 and 18
- "specific_lookup": What is the area of apartment 17?
- "aggregation": Total area of all rooms on 2.OG
- "metadata_lookup": Who is the architect? What materials are used?

Level guidelines:
- project: Questions about the overall project, address, dates
- metadata: Questions about architect, bauherr, legend, materials, changes
- building: Questions about a specific building or comparing buildings
- floor: Questions about a specific floor or all apartments on a floor
- apartment: Questions about specific apartment(s)
- room: Questions about specific rooms or room types
- stairs/aufzug: Questions about staircases or elevators"""

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Analyze questions about German architectural floor plans. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Question analysis failed: {e}")
            return {
                "target_level": "floor",
                "estimated_chunks_needed": 5,
                "query_type": "overview"
            }
    
    def _get_targeted_chunks(self, analysis: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        """Get chunks based on question analysis with smart filtering."""
        target_level = analysis.get("target_level", "floor")
        n_results = analysis.get("estimated_chunks_needed", 5)
        question_lower = question.lower()
        
        chunks = []
        seen_ids = set()
        
        def add_chunk(chunk_id: str):
            if chunk_id not in seen_ids and chunk_id in self.chunk_graph:
                seen_ids.add(chunk_id)
                chunk = self.chunk_graph[chunk_id]
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk.content,
                    "level": chunk.level,
                    "metadata": chunk.metadata
                })
        
        def add_children(chunk_id: str, depth: int = 1):
            if depth <= 0 or chunk_id not in self.chunk_graph:
                return
            chunk = self.chunk_graph[chunk_id]
            for child_id in chunk.child_chunk_ids:
                add_chunk(child_id)
                if depth > 1:
                    add_children(child_id, depth - 1)
        
        def check_stairs_aufzug():
            stairs_keywords = ['trepp', 'stair', 'treppenhaus', 'stufe']
            aufzug_keywords = ['aufzug', 'elevator', 'lift', 'fahrstuhl']
            
            wants_stairs = any(kw in question_lower for kw in stairs_keywords)
            wants_aufzug = any(kw in question_lower for kw in aufzug_keywords)
            
            if wants_stairs or wants_aufzug:
                for chunk_id, chunk in self.chunk_graph.items():
                    if wants_stairs and chunk.level == 'stairs':
                        add_chunk(chunk_id)
                    if wants_aufzug and chunk.level == 'aufzug':
                        add_chunk(chunk_id)
                return True
            return False
        
        # Check for stairs/aufzug first
        if check_stairs_aufzug():
            if chunks:
                return chunks[:n_results]
        
        # Check for project-level queries FIRST - keyword-based detection
        # Project info includes scale, creation date, print date, address, etc.
        project_keywords = ['projekt', 'project', 'adresse', 'address', 'gemarkung', 
                          'flurstück', 'flurstueck', 'erstellt', 'created', 'creation',
                          'maßstab', 'masstab', 'scale', 'gedruckt', 'printed', 'druckdatum',
                          'erstelldatum', 'erstellungsdatum', 'print date', 'creation date',
                          'gedrucktdatum', 'handling', 'working on', 'current project', 
                          'general info', 'project info', 'project name', 'projektname', 'flur ']
        needs_project = target_level == "project" or any(kw in question_lower for kw in project_keywords)
        
        if needs_project:
            for chunk_id, chunk in self.chunk_graph.items():
                if chunk.level == 'project':
                    add_chunk(chunk_id)
            # Return early if ONLY asking about project-level info
            project_only_keywords = ['masstab', 'maßstab', 'scale', 'gedruckt', 'printed', 
                                    'erstelldatum', 'erstellungsdatum', 'gedrucktdatum']
            if chunks and any(kw in question_lower for kw in project_only_keywords):
                return chunks[:n_results]
        
        # Check for building-level queries EARLY - keyword-based detection
        building_keywords = ['building', 'gebäude', 'haus', 'floor', 'etage', 'stockwerk', 
                           'geschoss', 'how many floor', 'building name', 'building number']
        needs_building = target_level == "building" or any(kw in question_lower for kw in building_keywords)
        
        if needs_building:
            for chunk_id, chunk in self.chunk_graph.items():
                if chunk.level == 'building':
                    add_chunk(chunk_id)
            # Return early only if purely about building structure (not space/area/apartments/walls/doors)
            needs_more_context = ['apartment', 'wohnung', 'room', 'zimmer', 'space', 'area', 
                                'fläche', 'total', 'living', 'wohn', 'sqm', 'qm', 'size',
                                'wall', 'wand', 'wide', 'width', 'thick', 'dick', 'external', 'außen',
                                'door', 'tür', 'window', 'fenster', 'equipment', 'ausstattung']
            if target_level == "building" and chunks and not any(kw in question_lower for kw in needs_more_context):
                return chunks[:n_results]
        
        # Check for metadata queries - check keywords regardless of LLM analysis
        # Removed 'datum' and 'date' since those are too broad (project has dates too)
        metadata_keywords = ['architekt', 'architect', 'bauherr', 'client', 'developer',
                           'legende', 'legend', 'material', 'wandaufbau', 'wall construction', 
                           'construction', 'exterior', 'außen', 'thickness', 'layer',
                           'änderung', 'anderung', 'change', 'history', 'revision']
        needs_metadata = analysis.get("needs_metadata") or target_level == "metadata" or any(kw in question_lower for kw in metadata_keywords)
        
        if needs_metadata:
            for chunk_id, chunk in self.chunk_graph.items():
                if chunk.level == 'metadata':
                    add_chunk(chunk_id)
            # If ONLY asking about metadata (not combined with building), return early
            if (target_level == "metadata" or analysis.get("query_type") == "metadata_lookup") and not needs_building:
                if chunks:
                    return chunks[:n_results]
        
        # Check for aggregation queries (total, sum, all, count, how many, list)
        aggregation_keywords = ['total', 'sum', 'all', 'count', 'how many', 'combined', 
                               'zusammen', 'gesamt', 'insgesamt', 'every', 'each', 'describe',
                               'list all', 'list the', 'show all', 'overview', 'enumerate']
        is_aggregation = analysis.get("query_type") == "aggregation" or any(kw in question_lower for kw in aggregation_keywords)
        
        # Check for queries about doors/windows that need room data
        room_aggregation_keywords = ['door', 'tür', 'window', 'fenster']
        needs_room_aggregation = any(kw in question_lower for kw in room_aggregation_keywords) and is_aggregation
        
        # Handle specific apartment queries FIRST (before aggregation)
        # This ensures "how many rooms in apartment 20" gets apartment 20, not all apartments
        specific_apt = analysis.get("specific_apartment")
        if specific_apt is not None:
            apt_list = specific_apt if isinstance(specific_apt, list) else [specific_apt]
            
            for chunk_id, chunk in self.chunk_graph.items():
                if chunk.level == "apartment" and chunk.apartment_id in apt_list:
                    add_chunk(chunk_id)
                    # ALWAYS add room children for specific apartment queries
                    add_children(chunk_id)
            
            # If we found specific apartments, return early (don't get all apartments via aggregation)
            if chunks:
                # Add floor for context
                for chunk_id, chunk in self.chunk_graph.items():
                    if chunk.level == "floor":
                        add_chunk(chunk_id)
                        break
                return chunks[:n_results * 2]
        
        if is_aggregation or needs_room_aggregation:
            # For aggregation queries, get ALL apartments
            for chunk_id, chunk in self.chunk_graph.items():
                if chunk.level == "apartment":
                    add_chunk(chunk_id)
                    # Also add room children for door/window counts
                    if needs_room_aggregation:
                        for child_id in chunk.child_chunk_ids:
                            add_chunk(child_id)
            # Also add floor chunk for context
            for chunk_id, chunk in self.chunk_graph.items():
                if chunk.level == "floor":
                    add_chunk(chunk_id)
        
        # Handle floor-level queries
        specific_floor = analysis.get("specific_floor")
        if specific_floor or target_level == "floor":
            for chunk_id, chunk in self.chunk_graph.items():
                if chunk.level == "floor":
                    if specific_floor and chunk.floor_id != specific_floor:
                        continue
                    add_chunk(chunk_id)
                    add_children(chunk_id)
        
        # Handle specific building queries (when a specific building ID like H1 is mentioned)
        specific_building = analysis.get("specific_building")
        if specific_building:
            for chunk_id, chunk in self.chunk_graph.items():
                if chunk.level == "building" and chunk.building_id == specific_building:
                    add_chunk(chunk_id)
                    add_children(chunk_id)
        
        # If still no chunks, do vector search with OpenAI embeddings
        if not chunks:
            # Generate embedding using OpenAI (same model as stored embeddings)
            embedding_response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[question]
            )
            query_embedding = embedding_response.data[0].embedding
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            for i in range(len(results['ids'][0])):
                chunk_id = results['ids'][0][i]
                if chunk_id not in seen_ids:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "content": results['documents'][0][i],
                        "level": results['metadatas'][0][i].get('level'),
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i]
                    })
        
        # Add parent context for better answers
        if analysis.get("needs_hierarchy_traversal", True):
            parent_ids = set()
            for chunk in chunks[:5]:
                chunk_obj = self.chunk_graph.get(chunk['chunk_id'])
                if chunk_obj and chunk_obj.parent_chunk_id:
                    parent_ids.add(chunk_obj.parent_chunk_id)
            
            for parent_id in list(parent_ids)[:3]:
                add_chunk(parent_id)
        
        # Calculate appropriate return limit
        specific_apt = analysis.get("specific_apartment")
        if specific_apt and isinstance(specific_apt, list):
            return_limit = max(n_results * 2, len(specific_apt) * 4)
        else:
            return_limit = n_results * 2
        
        return chunks[:return_limit]
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer using hierarchical retrieval.
        
        Args:
            question: The question to ask about the floor plan.
            
        Returns:
            Dictionary containing:
                - question: The original question
                - answer: The generated answer
                - chunks_retrieved: Number of chunks used
                - levels_used: List of hierarchy levels used
                - analysis: The question analysis results
        """
        # 1. Analyze question scope
        analysis = self._analyze_question_scope(question)
        
        # 2. Get targeted chunks
        chunks = self._get_targeted_chunks(analysis, question)
        
        if not chunks:
            return {
                "question": question,
                "answer": "I don't have enough information to answer this question.",
                "chunks_retrieved": 0,
                "levels_used": [],
                "analysis": analysis
            }
        
        # 3. Build context
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[{chunk['level'].upper()}]\n{chunk['content']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # 4. Generate answer
        system_prompt = """You are an expert assistant for German architectural floor plans and building projects.
Answer questions based ONLY on the provided context.
If the context doesn't contain the answer, say so clearly.
Use German terms when appropriate but explain in the user's language.
For measurements, use metric units (sqm, meters).
Be precise with numbers and apartment/room identifications."""
        
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.2
        )
        
        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "chunks_retrieved": len(chunks),
            "levels_used": list(set(c['level'] for c in chunks)),
            "analysis": {
                "target_level": analysis.get("target_level"),
                "estimated_chunks": analysis.get("estimated_chunks_needed"),
                "specific_apartment": analysis.get("specific_apartment"),
                "query_type": analysis.get("query_type"),
                "needs_metadata": analysis.get("needs_metadata")
            }
        }
    
    def clear(self):
        """Clear all data from the vector store."""
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Hierarchical floor plan chunks v2"}
        )
        self.chunk_graph.clear()
        self.processor.projects.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current data."""
        levels = {}
        projects = set()
        buildings = set()
        floors = set()
        
        for chunk in self.chunk_graph.values():
            levels[chunk.level] = levels.get(chunk.level, 0) + 1
            if chunk.project_id:
                projects.add(chunk.project_id)
            if chunk.building_id:
                buildings.add(chunk.building_id)
            if chunk.floor_id:
                floors.add(chunk.floor_id)
        
        return {
            "total_chunks": len(self.chunk_graph),
            "chunk_breakdown": levels,
            "projects": list(projects),
            "buildings": list(buildings),
            "floors": list(floors)
        }
