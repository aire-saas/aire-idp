"""
Hierarchical Processor for floor plan JSONs.

Handles full project info, metadata, buildings, floors, apartments, rooms.
"""

import json
from typing import Dict

from .models import (
    RoomData, StairsData, AufzugData, ApartmentData,
    FloorData, BuildingData, ProjectData
)
from .extractor import HierarchyExtractor


class HierarchicalProcessor:
    """
    Processes floor plan JSONs into hierarchical structure.
    Handles full project info, metadata, buildings, floors, apartments, rooms.
    """
    
    def __init__(self, use_llm: bool = True):
        self.extractor = HierarchyExtractor(use_llm=use_llm)
        self.projects: Dict[str, ProjectData] = {}
    
    def process_full_json(self, json_path: str) -> ProjectData:
        """
        Process a full PDF JSON file with all pages.
        Each page represents one floor of a building.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pages = data.get('pages', [])
        if not pages:
            raise ValueError("No pages found in JSON")
        
        first_page = pages[0]
        info_panel = first_page.get('infoPanel_Information', {})
        
        project = self.extractor.extract_project_info(info_panel)
        
        for page in pages:
            floor_data = self._process_page(page, project.project_id)
            
            building_id = floor_data.building_id
            if building_id not in project.buildings:
                project.buildings[building_id] = BuildingData(
                    building_id=building_id,
                    address=project.address
                )
            
            project.buildings[building_id].floors[floor_data.floor_id] = floor_data
        
        for building in project.buildings.values():
            building.total_apartments = sum(
                len(floor.apartments) for floor in building.floors.values()
            )
        
        self.projects[project.project_id] = project
        return project
    
    def _process_page(self, page: Dict, project_id: str) -> FloorData:
        """Process a single page (floor) from the JSON."""
        rooms_data = page.get('rooms', {})
        rooms = rooms_data.get('rooms', []) if isinstance(rooms_data, dict) else rooms_data if isinstance(rooms_data, list) else []
        stairs = rooms_data.get('stairs', []) if isinstance(rooms_data, dict) else []
        aufzugs = rooms_data.get('aufzugs', []) if isinstance(rooms_data, dict) else []
        
        page_num = page.get('page_number', 1)
        info_panel = page.get('infoPanel_Information', {})
        metadata = info_panel.get('metadata', {})
        projektname = metadata.get('projektname', f'Page_{page_num}')
        zeichnung = info_panel.get('zeichnung', {})
        floor_from_drawing = zeichnung.get('zeichnung', '').lower()
        
        hierarchy = self.extractor.extract_hierarchy(rooms, projektname)
        
        building_id = hierarchy.get('building_id') or 'Unknown'
        floor_id = hierarchy.get('floor_id') or 'Unknown'
        floor_number = hierarchy.get('floor_number', 0)
        
        # Try to extract floor from drawing title if not found in hierarchy
        if floor_id == 'Unknown' and floor_from_drawing:
            if '2.obergeschoss' in floor_from_drawing or '2.og' in floor_from_drawing:
                floor_id = '2.OG'
                floor_number = 2
            elif '1.obergeschoss' in floor_from_drawing or '1.og' in floor_from_drawing:
                floor_id = '1.OG'
                floor_number = 1
            elif 'erdgeschoss' in floor_from_drawing or 'eg' in floor_from_drawing:
                floor_id = 'EG'
                floor_number = 0
        
        floor_data = FloorData(
            floor_id=floor_id,
            floor_number=floor_number,
            source_file=projektname,
            building_id=building_id
        )
        
        room_hierarchy = {r['room_number']: r for r in hierarchy.get('rooms', [])}
        
        for room in rooms:
            room_num = room.get('room_id', 0) + 1
            room_info = room_hierarchy.get(room_num, {})
            
            # Extract wall data from room - handle dict format with string keys
            walls_data = room.get('walls', {})
            walls_list = []
            if isinstance(walls_data, dict):
                for wall_id, wall_info in walls_data.items():
                    if isinstance(wall_info, dict):
                        wall_entry = {
                            'wall_id': str(wall_id),
                            'detected_material': wall_info.get('detected_material', 'Unknown'),
                            'wall_thickness': float(wall_info.get('wall_thickness', 0)),
                            'wall_length': float(wall_info.get('wall_length', 0)),
                            'position': self._infer_wall_position(
                                wall_info.get('detected_material', ''),
                                wall_info.get('wall_thickness', 0)
                            )
                        }
                        walls_list.append(wall_entry)
            
            room_data = RoomData(
                room_id=room.get('room_id', 0),
                room_number=room_num,
                room_type=room_info.get('room_type', 'Unknown'),
                room_type_english=room_info.get('room_type_english', 'Unknown'),
                area_sqm=float(room.get('area_meters_sq', 0)),
                perimeter_m=float(room.get('perimeter_meters', 0)),
                doors_count=int(room.get('doors_count', 0)),
                windows_count=int(room.get('windows_count', 0)),
                equipment=room_info.get('equipment', []),
                words=room.get('words', []),
                floor_id=floor_id,
                building_id=building_id,
                subroom_ids=room_info.get('subroom_ids', []),
                walls=walls_list
            )
            
            apt_num = room_info.get('apartment_number')
            is_common = room_info.get('is_common_area', False)
            
            if apt_num and not is_common:
                room_data.apartment_id = apt_num
                if apt_num not in floor_data.apartments:
                    floor_data.apartments[apt_num] = ApartmentData(
                        apartment_number=apt_num,
                        floor_id=floor_id,
                        building_id=building_id
                    )
                floor_data.apartments[apt_num].rooms.append(room_data)
            else:
                floor_data.common_rooms.append(room_data)
        
        for apt in floor_data.apartments.values():
            apt.calculate_totals()
        
        for stair in stairs:
            stairs_data = StairsData(
                stairs_id=stair.get('stairs_id', 0),
                total_area_sqm=stair.get('total_area_meters_sq', 0),
                parts_count=len(stair.get('parts', [])),
                confidence=stair.get('confidence', 0),
                bounding_box=stair.get('bounding_box', {}),
                floor_id=floor_id,
                building_id=building_id
            )
            floor_data.stairs.append(stairs_data)
        
        for aufzug in aufzugs:
            aufzug_data = AufzugData(
                aufzug_id=aufzug.get('aufzug_id', 0),
                area_sqm=aufzug.get('area_meters_sq', 0),
                confidence=aufzug.get('confidence', 0),
                bounding_box=aufzug.get('bounding_box', {}),
                floor_id=floor_id,
                building_id=building_id
            )
            floor_data.aufzugs.append(aufzug_data)
        
        return floor_data
    
    def _infer_wall_position(self, material: str, thickness: float) -> str:
        """
        Infer wall position (external/internal) based on material and thickness.
        
        Args:
            material: Wall material name
            thickness: Wall thickness in meters
            
        Returns:
            Position: 'external', 'internal', or 'partition'
        """
        material_lower = material.lower()
        
        # Thick materials are typically external walls
        if thickness >= 0.20:  # >= 20cm
            if 'gipskarton' in material_lower or 'ständerwand' in material_lower:
                return 'partition'
            elif 'stahlbeton' in material_lower or 'brandschutz' in material_lower:
                return 'external'
            else:  # Mauerwerk, KS-L, etc.
                return 'external'
        
        # Thin materials are typically partition walls
        if thickness <= 0.15:  # <= 15cm
            if 'gipskarton' in material_lower or 'ständerwand' in material_lower:
                return 'partition'
            elif 'stahlbeton' in material_lower:
                return 'external'
            else:
                return 'internal'
        
        # Medium thickness - context dependent
        if 'gipskarton' in material_lower or 'ständerwand' in material_lower:
            return 'partition'
        elif 'stahlbeton' in material_lower or 'brandschutz' in material_lower:
            return 'external'
        else:
            return 'internal'
