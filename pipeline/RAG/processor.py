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
        rooms = rooms_data.get('rooms', [])
        stairs = rooms_data.get('stairs', [])
        aufzugs = rooms_data.get('aufzugs', [])
        
        page_num = page.get('page_number', 1)
        info_panel = page.get('infoPanel_Information', {})
        metadata = info_panel.get('metadata', {})
        projektname = metadata.get('projektname', f'Page_{page_num}')
        
        hierarchy = self.extractor.extract_hierarchy(rooms, projektname)
        
        building_id = hierarchy.get('building_id') or 'Unknown'
        floor_id = hierarchy.get('floor_id') or 'Unknown'
        floor_number = hierarchy.get('floor_number', 0)
        
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
            
            room_data = RoomData(
                room_id=room.get('room_id', 0),
                room_number=room_num,
                room_type=room_info.get('room_type', 'Unknown'),
                room_type_english=room_info.get('room_type_english', 'Unknown'),
                area_sqm=room.get('area_meters_sq', 0),
                perimeter_m=room.get('perimeter_meters', 0),
                doors_count=room.get('doors_count', 0),
                windows_count=room.get('windows_count', 0),
                equipment=room_info.get('equipment', []),
                words=room.get('words', []),
                floor_id=floor_id,
                building_id=building_id,
                subroom_ids=room_info.get('subroom_ids', [])
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
