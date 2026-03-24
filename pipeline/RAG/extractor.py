"""
Hierarchy Extractor for German floor plan data.

Uses Azure OpenAI LLM for intelligent extraction with regex fallback.
"""

import re
import json
import os
from typing import List, Dict, Any

from openai import AzureOpenAI

from .config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, LLM_DEPLOYMENT_NAME
from .models import ProjectData, ProjectMetadata


class HierarchyExtractor:
    """
    Extracts hierarchy information from floor plan JSON.
    Uses Azure OpenAI LLM for intelligent extraction with regex fallback.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm:
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
        else:
            self.client = None
    
    def extract_project_info(self, info_panel: Dict) -> ProjectData:
        """Extract project-level information from infoPanel."""
        projekt = info_panel.get('projekt', {})
        architekt = info_panel.get('architekt', {})
        bauherr = info_panel.get('bauherr', {})
        legende = info_panel.get('legende', {})
        wandaufbau = info_panel.get('wandaufbau_außenwände', {})
        aenderungen = info_panel.get('änderungen', [])
        
        # Handle nested address objects
        def format_address(addr):
            if isinstance(addr, dict):
                parts = []
                if addr.get('straße'): parts.append(addr['straße'])
                if addr.get('plz'): parts.append(addr['plz'])
                if addr.get('stadt'): parts.append(addr['stadt'])
                return ', '.join(parts)
            return str(addr) if addr else ''
        
        metadata = ProjectMetadata(
            architect_name=architekt.get('name', ''),
            architect_address=format_address(architekt.get('adresse', '')),
            architect_phone=architekt.get('telefon', ''),
            architect_email=architekt.get('email', ''),
            architect_website=architekt.get('website', ''),
            bauherr_name=bauherr.get('name', ''),
            bauherr_contact=bauherr.get('kontakt', ''),
            bauherr_address=format_address(bauherr.get('adresse', '')),
            legende=legende,
            wall_construction=wandaufbau,
            aenderungen=aenderungen
        )
        
        # Handle project name - try 'bezeichnung' first, then 'name'
        project_name = projekt.get('bezeichnung') or projekt.get('name') or 'Unknown'
        
        # Handle project address which may be nested
        project_address = format_address(projekt.get('adresse', ''))
        
        # Extract city from nested address or separate field
        project_city = ''
        if isinstance(projekt.get('adresse'), dict):
            addr = projekt['adresse']
            project_city = f"{addr.get('plz', '')} {addr.get('stadt', '')}".strip()
        else:
            project_city = projekt.get('plz_stadt', '')
        
        project = ProjectData(
            project_id=project_name,
            project_name=project_name,
            address=project_address,
            city=project_city,
            gemarkung=projekt.get('gemarkung', ''),
            flur=projekt.get('flur', ''),
            flurstueck=projekt.get('flurstück', ''),
            erstellt_date=projekt.get('erstellt', ''),
            gedruckt_date=projekt.get('gedruckt', ''),
            massstab=projekt.get('maßstab', ''),
            metadata=metadata
        )
        
        return project
    
    def extract_hierarchy(self, rooms: List[Dict], file_name: str) -> Dict[str, Any]:
        """
        Use Azure OpenAI LLM to analyze all rooms and extract hierarchy structure.
        Returns structured data about building, floor, apartments, and rooms.
        Includes wall materials and thickness information.
        """
        all_room_data = []
        for room in rooms:
            # Extract walls data from the room - handle dict format with string keys
            walls_dict = room.get('walls', {})
            walls_list = []
            if isinstance(walls_dict, dict):
                for wall_id, wall_data in walls_dict.items():
                    if isinstance(wall_data, dict):
                        walls_list.append({
                            'wall_id': wall_id,
                            'material': wall_data.get('detected_material', 'Unknown'),
                            'thickness_m': wall_data.get('wall_thickness', 0),
                            'length_m': wall_data.get('wall_length', 0)
                        })
            
            room_entry = {
                "room_number": room.get('room_id', 0) + 1,
                "words": room.get('words', []),
                "area_sqm": room.get('area_meters_sq', 0),
                "doors": room.get('doors_count', 0),
                "windows": room.get('windows_count', 0),
                "walls": walls_list  # Pass extracted walls
            }
            all_room_data.append(room_entry)
        
        prompt = f"""Analyze this German floor plan data and extract the hierarchy.

File name: {file_name}

Room data (including walls):
{json.dumps(all_room_data, indent=2, ensure_ascii=False)}

Extract and return a JSON object with this structure:
{{
    "building_id": "<building identifier like 'H1', 'H2', 'Haus 1', 'Haus 5', or extracted from codes, or null if not found>",
    "floor_id": "<floor like '2.OG', 'EG', '1.UG', or null if not found>",
    "floor_number": <numeric: 0 for EG, 1 for 1.OG, 2 for 2.OG, -1 for 1.UG, etc.>,
    "rooms": [
        {{
            "room_number": <int>,
            "apartment_number": <int or null if common area>,
            "room_type": "<German type: Zimmer, Bad, Küche, Flur, WC, etc.>",
            "room_type_english": "<English: Room, Bathroom, Kitchen, Hallway, Toilet>",
            "equipment": ["<detected equipment items>"],
            "is_common_area": <true if hallway/staircase/elevator, false otherwise>,
            "subroom_ids": [<list of additional room IDs if this is a combined/large room, else empty>],
            "walls": [
                {{
                    "wall_id": "<identifier>",
                    "detected_material": "<material type: e.g., Gipskarton, KS-L, Stahlbeton, Mauerwerk, etc.>",
                    "wall_thickness": <thickness in meters as float>,
                    "wall_length": <length in meters as float>,
                    "position": "<wall position: external/exterior/außen, internal/interior/innen, partition/trennwand>"
                }}
            ]
        }}
    ]
}}

WALL DATA EXTRACTION:
- Extract wall materials and thicknesses from the 'walls' field in each room
- detected_material: The construction material (e.g., "Gipskarton", "KS-L", "Stahlbeton", "Mauerwerk")
- wall_thickness: Thickness value in meters (convert from cm if needed: 10cm = 0.1m)
- wall_length: Length of wall segment in meters
- position: Classify wall position (external/outside walls vs internal/partition walls)
- If wall data exists, extract ALL wall segments for the room

IMPORTANT - Room/Apartment Notation Formats:
Different floor plans use different notation systems. Be flexible and detect ANY of these patterns:

1. Standard format: "17.Zimmer", "17.Bad" → Apartment 17, Room type Zimmer/Bad
   - Pattern: <apartment_number>.<room_type>
   
2. Compact numeric code: "50501" → Haus 5, Apartment 05, Zimmer 01
   - Pattern: <haus><apartment><room> as concatenated digits
   - Example: 50501 = Haus 5, Apartment 5, Room 1
   - Example: 60203 = Haus 6, Apartment 2, Room 3
   
3. Dash-separated format: "6.2.3-15" → Haus 6, Floor 2, Apartment 3, Room 15
   - Pattern: <haus>.<floor>.<apartment>-<room>
   - Example: 5.1.2-03 = Haus 5, Floor 1, Apartment 2, Room 3
   
4. Building-Floor prefix: "H1 - 2.OG" with "17.Zimmer" → Building H1, Floor 2.OG, Apartment 17
   - Combines building/floor info with apartment.room notation

5. Any other numeric patterns in words that indicate room/apartment groupings

SUBROOMS: If a room's words section contains MULTIPLE room identifiers (e.g., both "50501" and "50502"), 
this indicates a large room that encompasses multiple sub-rooms. List all detected room IDs in subroom_ids.

German floor notation:
- EG = Erdgeschoss (ground floor) = 0
- OG = Obergeschoss (upper floor): 1.OG=1, 2.OG=2, etc.
- UG = Untergeschoss (basement): 1.UG=-1, 2.UG=-2, etc.
- DG = Dachgeschoss (attic)

Equipment: Kühlschrank (fridge), Backofen (oven), Dusche (shower), Badewanne (bathtub), etc.
Common areas: Flur (hallway), Treppenhaus (stairwell), Aufzug (elevator)

Be flexible and extract meaningful hierarchy even if the notation format is unfamiliar."""

        # Skip LLM if disabled
        if not self.use_llm:
            return self._fallback_extraction(rooms, file_name)

        try:
            response = self.client.chat.completions.create(
                model=LLM_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert in German architectural floor plans. Extract hierarchy information and wall materials. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Azure OpenAI LLM extraction failed: {e}")
            return self._fallback_extraction(rooms, file_name)
    
    def _fallback_extraction(self, rooms: List[Dict], file_name: str) -> Dict[str, Any]:
        """Fallback extraction using regex patterns if LLM fails."""
        building_id = None
        floor_id = None
        floor_number = 0
        
        all_words = []
        for room in rooms:
            all_words.extend(room.get('words', []))
        
        all_text = ' '.join(all_words)
        
        # Building pattern: H1, H2, Haus 1, etc.
        building_match = re.search(r'(H\d+|Haus\s*\d+)', all_text, re.IGNORECASE)
        if building_match:
            building_id = building_match.group(1)
        
        # Floor pattern: 2.OG, EG, 1.UG, etc.
        floor_match = re.search(r'(\d+\.?[OU]G|EG|DG)', all_text, re.IGNORECASE)
        if floor_match:
            floor_id = floor_match.group(1)
            if 'EG' in floor_id.upper():
                floor_number = 0
            elif 'OG' in floor_id.upper():
                num_match = re.search(r'(\d+)', floor_id)
                floor_number = int(num_match.group(1)) if num_match else 1
            elif 'UG' in floor_id.upper():
                num_match = re.search(r'(\d+)', floor_id)
                floor_number = -int(num_match.group(1)) if num_match else -1
        
        room_info = []
        for room in rooms:
            room_num = room.get('room_id', 0) + 1
            words = room.get('words', [])
            word_text = ' '.join(words).lower()
            
            apt_num = None
            subroom_ids = []
            
            # Try multiple apartment notation patterns:
            
            # Pattern 1: Standard format "17.Zimmer", "17.Bad" -> apartment 17
            apt_match = re.search(r'(\d+)\.(zimmer|bad|küche|wc|flur)', word_text)
            if apt_match:
                apt_num = int(apt_match.group(1))
            
            # Pattern 2: Compact numeric code "50501" -> Haus 5, Apt 05, Room 01
            # Look for 5-6 digit codes
            if apt_num is None:
                compact_matches = re.findall(r'\b(\d{5,6})\b', word_text)
                if compact_matches:
                    # Parse first match: HAARR or HAARRR format
                    code = compact_matches[0]
                    if len(code) == 5:  # HAARR: Haus, Apt(2), Room(2)
                        building_id = building_id or f"H{code[0]}"
                        apt_num = int(code[1:3])
                    elif len(code) == 6:  # HAARRR or HHARR
                        building_id = building_id or f"H{code[0]}"
                        apt_num = int(code[1:3])
                    # Track all codes as potential subrooms
                    subroom_ids = compact_matches
            
            # Pattern 3: Dash-separated "6.2.3-15" -> Haus 6, Floor 2, Apt 3, Room 15
            # Only extract floor from dash notation if not already detected from standard pattern
            if apt_num is None:
                dash_match = re.search(r'(\d+)\.(\d+)\.(\d+)-(\d+)', word_text)
                if dash_match:
                    building_id = building_id or f"H{dash_match.group(1)}"
                    # Only use dash floor if we didn't find standard floor pattern
                    if floor_id is None:
                        dash_floor = int(dash_match.group(2))
                        floor_number = dash_floor
                        floor_id = f"{dash_floor}.OG" if dash_floor > 0 else "EG"
                    apt_num = int(dash_match.group(3))
            
            # Room type detection
            room_type = "Zimmer"
            room_type_en = "Room"
            if 'bad' in word_text:
                room_type = "Bad"
                room_type_en = "Bathroom"
            elif 'küche' in word_text:
                room_type = "Küche"
                room_type_en = "Kitchen"
            elif 'flur' in word_text:
                room_type = "Flur"
                room_type_en = "Hallway"
            elif 'wc' in word_text:
                room_type = "WC"
                room_type_en = "Toilet"
            elif 'balkon' in word_text:
                room_type = "Balkon"
                room_type_en = "Balcony"
            elif 'terrasse' in word_text:
                room_type = "Terrasse"
                room_type_en = "Terrace"
            elif 'abstellraum' in word_text or 'abst' in word_text:
                room_type = "Abstellraum"
                room_type_en = "Storage"
            
            # Equipment detection
            equipment = []
            if 'kühlschrank' in word_text or 'kühl-' in word_text:
                equipment.append("Kühlschrank")
            if 'backofen' in word_text:
                equipment.append("Backofen")
            if 'ceranfeld' in word_text:
                equipment.append("Ceranfeld")
            if 'spülbecken' in word_text or 'spül-' in word_text:
                equipment.append("Spülbecken")
            if 'dusche' in word_text:
                equipment.append("Dusche")
            if 'badewanne' in word_text or 'wanne' in word_text:
                equipment.append("Badewanne")
            if 'geschirr' in word_text:
                equipment.append("Geschirrspüler")
            
            is_common = room_type in ["Flur", "Treppenhaus", "Aufzug"] and apt_num is None
            
            room_info.append({
                "room_number": room_num,
                "apartment_number": apt_num,
                "room_type": room_type,
                "room_type_english": room_type_en,
                "equipment": equipment,
                "is_common_area": is_common,
                "subroom_ids": subroom_ids if len(subroom_ids) > 1 else []
            })
        
        return {
            "building_id": building_id,
            "floor_id": floor_id,
            "floor_number": floor_number,
            "rooms": room_info
        }
    
    def _extract_walls(self, room_data: Dict) -> List[Dict]:
        """Extract wall materials and thickness data from room."""
        walls = []
        
        if 'walls' not in room_data:
            return walls
        
        room_walls = room_data.get('walls', {})
        for wall_id, wall_info in room_walls.items():
            wall = {
                'wall_id': wall_id,
                'detected_material': wall_info.get('detected_material', 'Unknown'),
                'wall_thickness': wall_info.get('wall_thickness', 0),
                'wall_length': wall_info.get('wall_length', 0),
                'xmin': wall_info.get('xmin'),
                'ymin': wall_info.get('ymin'),
                'xmax': wall_info.get('xmax'),
                'ymax': wall_info.get('ymax')
            }
            walls.append(wall)
        
        return walls
    
    def _extract_wall_materials_summary(self, walls: List[Dict]) -> Dict:
        """Create summary of wall materials and thickness."""
        summary = {}
        
        for wall in walls:
            material = wall.get('detected_material', 'Unknown')
            thickness = wall.get('wall_thickness', 0)
            length = wall.get('wall_length', 0)
            
            if material not in summary:
                summary[material] = {
                    'count': 0,
                    'thicknesses': [],
                    'total_length': 0,
                    'min_thickness': thickness,
                    'max_thickness': thickness
                }
            
            summary[material]['count'] += 1
            summary[material]['thicknesses'].append(thickness)
            summary[material]['total_length'] += length
            summary[material]['min_thickness'] = min(summary[material]['min_thickness'], thickness)
            summary[material]['max_thickness'] = max(summary[material]['max_thickness'], thickness)
        
        # Calculate averages
        for material in summary:
            if summary[material]['thicknesses']:
                summary[material]['avg_thickness'] = (
                    sum(summary[material]['thicknesses']) / len(summary[material]['thicknesses'])
                )
            del summary[material]['thicknesses']
        
        return summary
