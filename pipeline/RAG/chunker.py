"""
Hierarchical Chunker for creating embeddings.

Creates hierarchical chunks for embedding and retrieval.
Levels: project → metadata → building → floor → apartment/stairs/aufzug → room
"""

from typing import List, Dict

from openai import OpenAI

from .config import OPENAI_API_KEY, EMBEDDING_MODEL
from .models import (
    ProjectData, RoomData, StairsData, AufzugData,
    ApartmentData, FloorData, BuildingData, HierarchicalChunk
)


class HierarchicalChunker:
    """
    Creates hierarchical chunks for embedding and retrieval.
    Levels: project → metadata → building → floor → apartment/stairs/aufzug → room
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def create_all_chunks(self, project: ProjectData) -> List[HierarchicalChunk]:
        """Create chunks for all levels of the hierarchy."""
        chunks = []
        
        # 1. Project-level chunk
        project_chunk = self._create_project_chunk(project)
        chunks.append(project_chunk)
        
        # 2. Metadata chunk
        metadata_chunk = self._create_metadata_chunk(project)
        metadata_chunk.parent_chunk_id = project_chunk.chunk_id
        project_chunk.child_chunk_ids.append(metadata_chunk.chunk_id)
        chunks.append(metadata_chunk)
        
        # 3. Building-level chunks
        building_chunks = []
        for building_id, building in project.buildings.items():
            building_chunk = self._create_building_chunk(building, project.project_id)
            building_chunk.parent_chunk_id = project_chunk.chunk_id
            project_chunk.child_chunk_ids.append(building_chunk.chunk_id)
            building_chunks.append(building_chunk)
            chunks.append(building_chunk)
            
            # 4. Floor-level chunks
            floor_chunks = []
            for floor_id, floor in building.floors.items():
                floor_chunk = self._create_floor_chunk(floor, project.project_id)
                floor_chunk.parent_chunk_id = building_chunk.chunk_id
                building_chunk.child_chunk_ids.append(floor_chunk.chunk_id)
                floor_chunks.append(floor_chunk)
                chunks.append(floor_chunk)
                
                # 5. Apartment chunks
                apt_chunks = []
                for apt_num, apartment in floor.apartments.items():
                    apt_chunk = self._create_apartment_chunk(apartment, project.project_id)
                    apt_chunk.parent_chunk_id = floor_chunk.chunk_id
                    floor_chunk.child_chunk_ids.append(apt_chunk.chunk_id)
                    apt_chunks.append(apt_chunk)
                    chunks.append(apt_chunk)
                    
                    # 6. Room chunks
                    room_chunks = []
                    for room in apartment.rooms:
                        room_chunk = self._create_room_chunk(room, project.project_id, apt_num)
                        room_chunk.parent_chunk_id = apt_chunk.chunk_id
                        apt_chunk.child_chunk_ids.append(room_chunk.chunk_id)
                        room_chunks.append(room_chunk)
                        chunks.append(room_chunk)
                    
                    for rc in room_chunks:
                        rc.sibling_chunk_ids = [c.chunk_id for c in room_chunks if c.chunk_id != rc.chunk_id]
                
                for ac in apt_chunks:
                    ac.sibling_chunk_ids = [c.chunk_id for c in apt_chunks if c.chunk_id != ac.chunk_id]
                
                # 5b. Stairs chunks
                for stair in floor.stairs:
                    stairs_chunk = self._create_stairs_chunk(stair, project.project_id)
                    stairs_chunk.parent_chunk_id = floor_chunk.chunk_id
                    floor_chunk.child_chunk_ids.append(stairs_chunk.chunk_id)
                    chunks.append(stairs_chunk)
                
                # 5c. Aufzug chunks
                for aufzug in floor.aufzugs:
                    aufzug_chunk = self._create_aufzug_chunk(aufzug, project.project_id)
                    aufzug_chunk.parent_chunk_id = floor_chunk.chunk_id
                    floor_chunk.child_chunk_ids.append(aufzug_chunk.chunk_id)
                    chunks.append(aufzug_chunk)
            
            for fc in floor_chunks:
                fc.sibling_chunk_ids = [c.chunk_id for c in floor_chunks if c.chunk_id != fc.chunk_id]
        
        for bc in building_chunks:
            bc.sibling_chunk_ids = [c.chunk_id for c in building_chunks if c.chunk_id != bc.chunk_id]
        
        return chunks
    
    def _create_project_chunk(self, project: ProjectData) -> HierarchicalChunk:
        """Create project-level chunk."""
        content = f"""PROJECT (Projekt): {project.project_name}
Address (Adresse): {project.address}, {project.city}
Gemarkung: {project.gemarkung}, Flur: {project.flur}, Flurstück: {project.flurstueck}
Creation Date (Erstellt/Erstelldatum): {project.erstellt_date}
Print Date (Gedruckt/Druckdatum): {project.gedruckt_date}
Scale (Maßstab): {project.massstab}
Total Buildings: {len(project.buildings)}"""
        
        return HierarchicalChunk(
            chunk_id=f"project_{project.project_id}",
            level="project",
            content=content,
            project_id=project.project_id,
            metadata={
                "project_name": project.project_name,
                "address": project.address,
                "city": project.city,
                "erstellt_date": project.erstellt_date,
                "gedruckt_date": project.gedruckt_date,
                "massstab": project.massstab,
                "building_count": len(project.buildings)
            }
        )
    
    def _create_metadata_chunk(self, project: ProjectData) -> HierarchicalChunk:
        """Create metadata chunk with architect, bauherr, legende."""
        meta = project.metadata
        
        content = f"""PROJECT METADATA for {project.project_name}

ARCHITECT (Architekt):
  Name: {meta.architect_name}
  Address: {meta.architect_address}
  Phone: {meta.architect_phone}
  Email: {meta.architect_email}
  Website: {meta.architect_website}

CLIENT/DEVELOPER (Bauherr):
  Name: {meta.bauherr_name}
  Contact: {meta.bauherr_contact}
  Address: {meta.bauherr_address}

LEGEND (Legende):
{self._format_legende(meta.legende)}

WALL CONSTRUCTION (Wandaufbau):
{self._format_wall_construction(meta.wall_construction)}

CHANGE HISTORY (Änderungen):
{self._format_aenderungen(meta.aenderungen)}"""
        
        return HierarchicalChunk(
            chunk_id=f"metadata_{project.project_id}",
            level="metadata",
            content=content,
            project_id=project.project_id,
            metadata={
                "architect": meta.architect_name,
                "bauherr": meta.bauherr_name,
                "has_legende": bool(meta.legende),
                "changes_count": len(meta.aenderungen)
            }
        )
    
    def _format_legende(self, legende: Dict) -> str:
        """Format legend entries."""
        lines = []
        for category, items in legende.items():
            lines.append(f"  {category}:")
            if isinstance(items, dict):
                for abbr, meaning in items.items():
                    lines.append(f"    {abbr}: {meaning}")
            else:
                lines.append(f"    {items}")
        return '\n'.join(lines) if lines else "  No legend available"
    
    def _format_wall_construction(self, wandaufbau: Dict) -> str:
        """Format wall construction info."""
        lines = []
        for wall_type, layers in wandaufbau.items():
            lines.append(f"  {wall_type}:")
            if isinstance(layers, dict):
                for layer, thickness in layers.items():
                    lines.append(f"    {layer}: {thickness}")
        return '\n'.join(lines) if lines else "  No wall construction info"
    
    def _format_aenderungen(self, aenderungen: list) -> str:
        """Format change history."""
        lines = []
        for change in aenderungen:
            date = change.get('datum', '')
            name = change.get('name', '')
            # Try both 'änderungen' and 'beschreibung' keys for the description
            desc = change.get('änderungen') or change.get('beschreibung', '')
            index = change.get('index', '')
            if index:
                lines.append(f"  [{index}] {date} ({name}): {desc}")
            else:
                lines.append(f"  {date} ({name}): {desc}")
        return '\n'.join(lines) if lines else "  No changes recorded"
    
    def _format_wall_materials_and_thickness(self, room: RoomData) -> str:
        """Format wall materials and thickness information from room data."""
        if not hasattr(room, 'walls') or not room.walls:
            return "  No wall data available"
        
        lines = []
        materials_summary = {}
        total_wall_length = 0
        
        # Aggregate data by material
        for wall in room.walls:
            material = wall.get('detected_material', 'Unknown')
            thickness = wall.get('wall_thickness', 0)
            length = wall.get('wall_length', 0)
            position = wall.get('position', 'unknown')
            
            if material not in materials_summary:
                materials_summary[material] = {
                    'thicknesses': [],
                    'total_length': 0,
                    'count': 0,
                    'positions': []
                }
            
            materials_summary[material]['thicknesses'].append(thickness)
            materials_summary[material]['total_length'] += length
            materials_summary[material]['count'] += 1
            materials_summary[material]['positions'].append(position)
            total_wall_length += length
        
        # Format summary by material type
        lines.append("WALL MATERIALS AND THICKNESS (WANDMATERIAL UND WANDDICKE):")
        for material in sorted(materials_summary.keys()):
            data = materials_summary[material]
            min_thickness = min(data['thicknesses']) if data['thicknesses'] else 0
            max_thickness = max(data['thicknesses']) if data['thicknesses'] else 0
            avg_thickness = sum(data['thicknesses']) / len(data['thicknesses']) if data['thicknesses'] else 0
            
            # Convert to cm for readability if needed
            min_cm = min_thickness * 100
            max_cm = max_thickness * 100
            avg_cm = avg_thickness * 100
            
            lines.append(f"  {material}:")
            lines.append(f"    Thickness (Wanddicke): {min_cm:.1f}cm - {max_cm:.1f}cm (avg: {avg_cm:.1f}cm)")
            lines.append(f"    Total Length (Gesamtlänge): {data['total_length']:.2f}m ({data['count']} wall segments)")
            
            # Add position info if available
            unique_positions = set(p for p in data['positions'] if p)
            if unique_positions:
                pos_str = ', '.join(sorted(unique_positions))
                lines.append(f"    Position: {pos_str}")
        
        lines.append(f"\n  Total Wall Perimeter (Gesamter Wandumfang): {total_wall_length:.2f}m")
        
        return '\n'.join(lines)
    
    def _format_room_walls_detail(self, room: RoomData) -> str:
        """Format detailed wall information for room."""
        if not hasattr(room, 'walls') or not room.walls:
            return ""
        
        lines = []
        lines.append("DETAILED WALL INFORMATION (DETAILLIERTE WANDINFORMATIONEN):")
        
        for idx, wall in enumerate(room.walls, 1):
            material = wall.get('detected_material', 'Unknown')
            thickness = wall.get('wall_thickness', 0)
            length = wall.get('wall_length', 0)
            position = wall.get('position', 'unknown')
            wall_id = wall.get('wall_id', f'wall_{idx}')
            
            # Convert thickness to cm for better readability
            thickness_cm = thickness * 100
            
            lines.append(f"  Wall {idx} ({wall_id}):")
            lines.append(f"    Material: {material} | Thickness: {thickness_cm:.1f}cm | Length: {length:.2f}m | Position: {position}")
        
        return '\n'.join(lines)
    
    def _create_building_chunk(self, building: BuildingData, project_id: str) -> HierarchicalChunk:
        """Create building-level chunk."""
        floor_ids = list(building.floors.keys())
        total_apts = sum(len(f.apartments) for f in building.floors.values())
        
        # Collect all apartment numbers across all floors
        all_apt_numbers = []
        for floor in building.floors.values():
            all_apt_numbers.extend(list(floor.apartments.keys()))
        apt_list_str = ', '.join(map(str, sorted(all_apt_numbers))) if all_apt_numbers else 'None'
        
        content = f"""BUILDING (Gebäude): {building.building_id}
Project: {project_id}
Address: {building.address}
Total Floors: {len(building.floors)}
Floor IDs: {', '.join(floor_ids)}
Total Apartments: {total_apts}
Apartment Numbers: {apt_list_str}"""
        
        return HierarchicalChunk(
            chunk_id=f"building_{project_id}_{building.building_id}",
            level="building",
            content=content,
            project_id=project_id,
            building_id=building.building_id,
            metadata={
                "floor_count": len(building.floors),
                "apartment_count": total_apts,
                "floor_ids": floor_ids,
                "apartment_numbers": all_apt_numbers
            }
        )
    
    def _create_floor_chunk(self, floor: FloorData, project_id: str) -> HierarchicalChunk:
        """Create floor-level chunk."""
        apt_numbers = list(floor.apartments.keys())
        total_rooms = sum(len(a.rooms) for a in floor.apartments.values())
        total_area = sum(a.total_area_sqm for a in floor.apartments.values())
        
        content = f"""FLOOR: {floor.floor_id} (Level {floor.floor_number})
Building: {floor.building_id}
Project: {project_id}
Source: {floor.source_file}
Apartments: {len(floor.apartments)} ({', '.join(map(str, apt_numbers))})
Total Rooms: {total_rooms}
Total Area: {total_area:.1f} sqm
Stairs: {len(floor.stairs)}
Elevators: {len(floor.aufzugs)}
Common Areas: {len(floor.common_rooms)}"""
        
        return HierarchicalChunk(
            chunk_id=f"floor_{project_id}_{floor.building_id}_{floor.floor_id}",
            level="floor",
            content=content,
            project_id=project_id,
            building_id=floor.building_id,
            floor_id=floor.floor_id,
            metadata={
                "floor_number": floor.floor_number,
                "apartment_count": len(floor.apartments),
                "apartment_numbers": apt_numbers,
                "total_rooms": total_rooms,
                "total_area_sqm": total_area,
                "has_stairs": len(floor.stairs) > 0,
                "has_elevator": len(floor.aufzugs) > 0
            }
        )
    
    def _create_apartment_chunk(self, apartment: ApartmentData, project_id: str) -> HierarchicalChunk:
        """Create apartment-level chunk."""
        room_types = [r.room_type for r in apartment.rooms]
        room_types_str = ', '.join(room_types)
        
        all_equipment = []
        for room in apartment.rooms:
            all_equipment.extend(room.equipment)
        equipment_str = ', '.join(set(all_equipment)) if all_equipment else "None listed"
        
        content = f"""APARTMENT: {apartment.apartment_number}
Floor: {apartment.floor_id}
Building: {apartment.building_id}
Project: {project_id}
Total Rooms: {apartment.room_count}
Total Area: {apartment.total_area_sqm:.1f} sqm
Room Types: {room_types_str}
Equipment: {equipment_str}"""
        
        return HierarchicalChunk(
            chunk_id=f"apt_{project_id}_{apartment.building_id}_{apartment.floor_id}_{apartment.apartment_number}",
            level="apartment",
            content=content,
            project_id=project_id,
            building_id=apartment.building_id,
            floor_id=apartment.floor_id,
            apartment_id=apartment.apartment_number,
            metadata={
                "room_count": apartment.room_count,
                "total_area_sqm": apartment.total_area_sqm,
                "room_types": room_types,
                "equipment": list(set(all_equipment))
            }
        )
    
    def _create_room_chunk(self, room: RoomData, project_id: str, apt_num: int) -> HierarchicalChunk:
        """Create room-level chunk."""
        equipment_str = ', '.join(room.equipment) if room.equipment else "None"
        subrooms_str = ', '.join(room.subroom_ids) if room.subroom_ids else ""
        
        content = f"""ROOM: {room.room_type} ({room.room_type_english})
Apartment: {apt_num}
Floor: {room.floor_id}
Building: {room.building_id}
Project: {project_id}
Area: {room.area_sqm:.2f} sqm
Perimeter: {room.perimeter_m:.2f} m
Doors: {room.doors_count}
Windows: {room.windows_count}
Equipment: {equipment_str}"""
        
        if subrooms_str:
            content += f"\nSubrooms: {subrooms_str}"
        
        content += f"\n\n{self._format_wall_materials_and_thickness(room)}"
        content += f"\n{self._format_room_walls_detail(room)}"
        
        return HierarchicalChunk(
            chunk_id=f"room_{project_id}_{room.building_id}_{room.floor_id}_{apt_num}_{room.room_id}",
            level="room",
            content=content,
            project_id=project_id,
            building_id=room.building_id,
            floor_id=room.floor_id,
            apartment_id=apt_num,
            room_id=room.room_id,
            metadata={
                "room_type": room.room_type,
                "room_type_english": room.room_type_english,
                "area_sqm": room.area_sqm,
                "doors": room.doors_count,
                "windows": room.windows_count,
                "equipment": room.equipment,
                "subroom_ids": room.subroom_ids
            }
        )
    
    def _create_stairs_chunk(self, stairs: StairsData, project_id: str) -> HierarchicalChunk:
        """Create stairs chunk."""
        content = f"""STAIRS (Treppenhaus): #{stairs.stairs_id}
Floor: {stairs.floor_id}
Building: {stairs.building_id}
Project: {project_id}
Total Area: {stairs.total_area_sqm:.2f} sqm
Number of Parts: {stairs.parts_count}
Detection Confidence: {stairs.confidence:.1%}"""
        
        return HierarchicalChunk(
            chunk_id=f"stairs_{project_id}_{stairs.building_id}_{stairs.floor_id}_{stairs.stairs_id}",
            level="stairs",
            content=content,
            project_id=project_id,
            building_id=stairs.building_id,
            floor_id=stairs.floor_id,
            stairs_id=stairs.stairs_id,
            metadata={
                "area_sqm": stairs.total_area_sqm,
                "parts_count": stairs.parts_count,
                "confidence": stairs.confidence
            }
        )
    
    def _create_aufzug_chunk(self, aufzug: AufzugData, project_id: str) -> HierarchicalChunk:
        """Create elevator chunk."""
        content = f"""ELEVATOR (Aufzug): #{aufzug.aufzug_id}
Floor: {aufzug.floor_id}
Building: {aufzug.building_id}
Project: {project_id}
Area: {aufzug.area_sqm:.2f} sqm
Detection Confidence: {aufzug.confidence:.1%}"""
        
        return HierarchicalChunk(
            chunk_id=f"aufzug_{project_id}_{aufzug.building_id}_{aufzug.floor_id}_{aufzug.aufzug_id}",
            level="aufzug",
            content=content,
            project_id=project_id,
            building_id=aufzug.building_id,
            floor_id=aufzug.floor_id,
            aufzug_id=aufzug.aufzug_id,
            metadata={
                "area_sqm": aufzug.area_sqm,
                "confidence": aufzug.confidence
            }
        )
    
    def generate_embeddings(self, chunks: List[HierarchicalChunk]) -> List[HierarchicalChunk]:
        """Generate embeddings for all chunks."""
        contents = [c.content for c in chunks]
        
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            all_embeddings.extend([e.embedding for e in response.data])
        
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk.embedding = embedding
        
        return chunks
