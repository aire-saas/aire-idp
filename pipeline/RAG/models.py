"""
Data models for the Hierarchical RAG system.

Hierarchy: Project → Metadata → Building → Floor → Apartment/Stairs/Aufzug → Room
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RoomData:
    """Individual room within an apartment."""
    room_id: int
    room_number: int
    room_type: str  # German: Zimmer, Bad, Küche, etc.
    room_type_english: str  # English: Room, Bathroom, Kitchen, etc.
    area_sqm: float
    perimeter_m: float
    doors_count: int
    windows_count: int
    equipment: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    dimensions: Dict[str, Any] = field(default_factory=dict)
    apartment_id: Optional[int] = None
    floor_id: str = ""
    building_id: str = ""
    subroom_ids: List[str] = field(default_factory=list)  # IDs of subrooms if this is a combined space
    walls: List[Dict[str, Any]] = field(default_factory=list)  # Wall material and thickness data


@dataclass
class StairsData:
    """Staircase in a building."""
    stairs_id: int
    total_area_sqm: float
    parts_count: int
    confidence: float
    bounding_box: Dict[str, int] = field(default_factory=dict)
    floor_id: str = ""
    building_id: str = ""


@dataclass
class AufzugData:
    """Elevator in a building."""
    aufzug_id: int
    area_sqm: float
    confidence: float
    bounding_box: Dict[str, int] = field(default_factory=dict)
    floor_id: str = ""
    building_id: str = ""


@dataclass
class ApartmentData:
    """Apartment containing multiple rooms."""
    apartment_number: int
    rooms: List[RoomData] = field(default_factory=list)
    total_area_sqm: float = 0
    room_count: int = 0
    floor_id: str = ""
    building_id: str = ""
    
    def calculate_totals(self):
        """Calculate total area and room count from rooms list."""
        self.total_area_sqm = sum(r.area_sqm for r in self.rooms)
        self.room_count = len(self.rooms)


@dataclass
class FloorData:
    """Floor of a building containing apartments, stairs, elevators."""
    floor_id: str  # "2.OG", "EG", etc.
    floor_number: int  # Numeric: 0=EG, 1=1.OG, -1=1.UG
    source_file: str
    building_id: str
    apartments: Dict[int, ApartmentData] = field(default_factory=dict)
    stairs: List[StairsData] = field(default_factory=list)
    aufzugs: List[AufzugData] = field(default_factory=list)
    common_rooms: List[RoomData] = field(default_factory=list)


@dataclass
class BuildingData:
    """Building containing multiple floors."""
    building_id: str  # "H1", "Haus 1", etc.
    floors: Dict[str, FloorData] = field(default_factory=dict)
    address: str = ""
    total_apartments: int = 0


@dataclass
class ProjectMetadata:
    """Project-level metadata - shared across all buildings/floors."""
    architect_name: str = ""
    architect_address: str = ""
    architect_phone: str = ""
    architect_email: str = ""
    architect_website: str = ""
    bauherr_name: str = ""
    bauherr_contact: str = ""
    bauherr_address: str = ""
    legende: Dict[str, Dict[str, str]] = field(default_factory=dict)
    wall_construction: Dict[str, Dict[str, str]] = field(default_factory=dict)
    aenderungen: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ProjectData:
    """Top-level project containing all data."""
    project_id: str
    project_name: str
    address: str = ""
    city: str = ""
    gemarkung: str = ""
    flur: str = ""
    flurstueck: str = ""
    erstellt_date: str = ""
    gedruckt_date: str = ""
    massstab: str = ""
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    buildings: Dict[str, BuildingData] = field(default_factory=dict)


@dataclass 
class HierarchicalChunk:
    """A chunk at any level of the hierarchy with relationships."""
    chunk_id: str
    level: str  # project, metadata, building, floor, apartment, stairs, aufzug, room
    content: str
    embedding: List[float] = field(default_factory=list)
    project_id: Optional[str] = None
    building_id: Optional[str] = None
    floor_id: Optional[str] = None
    apartment_id: Optional[int] = None
    room_id: Optional[int] = None
    stairs_id: Optional[int] = None
    aufzug_id: Optional[int] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    sibling_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
