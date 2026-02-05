"""
Unified Pipeline for PDF Processing and Room Detection

Flow:
1. Process PDF → Extract text, images, dimensions
2. Split plan → Separate floor plan from info panel
3. Detect rooms → Identify walls, doors, stairs, rooms
4. Save results → Output annotated images and JSON data
"""

import logging
import json
from pathlib import Path
from typing import List
import re

from infoGroupseExtractor import InfoGroupExtractor
import planSplitter
from process import PDFProcessor, PageContent
from planSplitter import PlanSplitter
from full_iter_one_model import full_iter_one_model
from RAG.pipeline import FloorPlanQA

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ArchitecturalPlanPipeline:
    """Complete pipeline for processing architectural PDFs and detecting rooms."""
    
    def __init__(self, 
                 pdf_path: str,
                 output_dir: str = "pipeline_output",
                 model_path: str = 'best.pt',
                 stairs_model_path: str = 'best_stairs.pt',
                 dpi: int = 150):
        """
        Initialize the pipeline.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory for output files
            model_path: Path to YOLO detection model
            stairs_model_path: Path to YOLO stairs model
            dpi: DPI for PDF rendering
        """
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = model_path
        self.stairs_model_path = stairs_model_path
        self.dpi = dpi
        
        self.pdf_processor = PDFProcessor(pdf_path)
        self.plan_splitter = PlanSplitter(self.pdf_processor)
        self.info_extractor = InfoGroupExtractor(pdfProcessor= self.pdf_processor)
    
        
        self.pages_content: List[PageContent] = []
    
    def run(self) -> FloorPlanQA:
        """
        Execute the full pipeline.
        
        Returns:
            List of processed PageContent objects
        """
        logger.info(f"Starting pipeline for {self.pdf_path}")
        
        # Step 1: Process PDF
        logger.info("Step 1: Processing PDF - extracting text and images...")
        self.pages_content = self._process_pdf()
        
        # Step 2: Split plans and info panels 
        logger.info("Step 2: Splitting plans - separating floor plan from info panel...")
        self.pages_content = self._split_plans()
        
        logger.info("Step 2.5.1: OpenAI grouping - extracting structured info from info panels...")
        self.pages_content = self._extract_info_panels()

        logger.info("Step 2.5.2: Scale the image to Maßstab")
        scale = self._extract_scale()
        logger.info(f"Extracted scale: 1:{scale}")
        if (scale is not None and scale > 1):
            self.pages_content = self._scale_plans(scale)
            self.pages_content = self._scale_Text(scale)
            


        # Step 3: Detect rooms
        logger.info("Step 3: Detecting rooms and architectural elements...")
        self.pages_content = self._detect_rooms()

        # Step 4 : Link text to rooms
        logger.info("Step 4: Linking text to detected rooms...")
        self.linkTextToRooms()

        # Step 5 : Clean JSON
        logger.info("Step 5: Cleaning JSON outputs...")
        self.clean_json()
        
        # Step 4: Create full json :
        full_pdf = {
            "pdf_path": self.pdf_path,
            "pages": [
                {
                    "page_number": page.page_number,
                    "infoPanel_Information": getattr(page, 'openAIGrouping', None),
                    "number_of_rooms": len(page.rooms.get('rooms', [])) if page.rooms else 0,
                    "rooms": page.rooms        
                }
                for page in self.pages_content
            ]
            }
        with open(self.output_dir / "fullPDF.json", "w", encoding="utf-8") as f:
            json.dump(full_pdf, f, indent=2, ensure_ascii=False)        
        # Step 4: Save results
        logger.info("Step 4: Saving results...")
        self._save_results()
        
        # Step 5 : ASK
        qa = FloorPlanQA()
        #qa.ingestD([full_pdf])
     #   qa.ingest([self.output_dir / "fullPDF.json"])
        '''
        while True:
            question = input("enter question:" )
            answer = qa.ask(question=question)
            print("answer " + answer )
        '''

        '''
        print("type of " , type(full_pdf))

        answer = qa.ask("Who is the architect?")
        print("answering the question is " + answer)

        # 3. Ask questions about your floor plans
        #answer = qa.ask("Who is the architect?")
        print(answer)'''
        logger.info("Pipeline complete!")
        return qa
    
    def _process_pdf(self) -> List[PageContent]:
        """Step 1: Process PDF to extract text and images."""
        self.pdf_processor.process()
        return self.pdf_processor.pages_content
    
    def _split_plans(self) -> List[PageContent]:
        """Step 2: Split floor plans from info panels."""
        for page in self.pages_content:
            try:
                self.plan_splitter.split_plan()
         
            except Exception as e:
                logger.warning(f"Page {page.page_number}: Failed to split plan - {e}")
                # Fall back to using the full page image as plan
                page.plan = page.page_image.copy()
        
        return self.pages_content

    def _extract_info_panels(self) -> List[PageContent]:
        """Step 2.5: Extract structured information from info panels using OpenAI."""
        self.info_extractor.processAllPages()
        return self.pages_content
    
    def find_key(self, data, target_key):
        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() == target_key:
                    return value
                result = self.find_key(value, target_key)
                if result is not None:
                    return result

        elif isinstance(data, list):
            for item in data:
                result = self.find_key(item, target_key)
                if result is not None:
                    return result

        return None
    
    def _extract_scale(self):
        for page_content in self.pages_content: 
            val = self.find_key(page_content.openAIGrouping, "maßstab")
            if (val is not None):
                if ("100" in val): return 2 
                return 1

    def _scale_plans(self, scale: int):
        for page_content in self.pages_content:
            if page_content.plan:
                width, height = page_content.plan.size
                new_width = int(width * scale)
                new_height = int(height * scale)
                page_content.plan = page_content.plan.resize((new_width, new_height))
        return self.pages_content

    def _scale_Text(self, scale: int):
        for page_content in self.pages_content:
            page_content.offset = (page_content.offset[0]*scale, page_content.offset[1]*scale)
            if page_content.text:
                self.scale_image_coordinates(page_content.text, scale)
        return self.pages_content

    def scale_image_coordinates(self, data, scale):
        if isinstance(data, dict):
            for key, value in data.items():

                # Found the target block
                if key == "image_coordinates" and isinstance(value, dict):
                    for coord in ("x0", "y0", "x1", "y1"):
                        if coord in value and isinstance(value[coord], (int, float)):
                            value[coord] *= scale

                # Recurse into nested structures
                self.scale_image_coordinates(value, scale)

        elif isinstance(data, list):
            for item in data:
                self.scale_image_coordinates(item, scale)
    
    def _detect_rooms(self) -> List[PageContent]:
        """Step 3: Detect rooms and architectural elements."""
        self.pages_content = full_iter_one_model(
            self.pages_content,
            model_path=self.model_path,
            stairs_model_path=self.stairs_model_path,
            conf= 0.25,
            device='cpu'
        )
        return self.pages_content
    
    
    def linkTextToRooms(self):
        for page in self.pages_content:
            if page.rooms and page.text:
                self.linkPage(page.text, page.rooms, page.offset)

    def linkPage(self, doclingJson, roomsJson, offset):   
        
        def filter_words(words: list) -> list:
            words_filtered = []
            surface = 0.0  
            # Rule 1: only digits and length <= 2
            for s in words:
                s = s.strip()
                if (("m²" in s) or ("m2" in s) or ("qm" in s)):
                    match = re.search(r"\d+(\.\d+)?", s)
                    if match:
                        number = float(match.group())
                        #print(number)
                        surface += number
                if (s.isdigit() and len(s) <= 3) or (re.fullmatch(r"\d+\.\d+", s) or (s == "")):
                        continue
                words_filtered.append(s)

            return words_filtered , surface
        # we test the surface here
        
        for roomEntry in roomsJson["rooms"]:
            surfaceT = 0.0
            room_words = []
            rect_count = roomEntry["rectangles_count"]
            print(f"Room ID: {roomEntry['room_id']} has {rect_count} rectangles.")
            for rect in roomEntry.get("rectangles", []):
                filtered_words, surface = filter_words(self.search_text(doclingJson, rect["bbox"], offset))
                surfaceT += surface
                room_words.extend(filtered_words)   
                #room_words.extend(self.search_text(doclingJson, rect["bbox"], offset))
            computedSurface = roomEntry.get("area_meters_sq", 0)
            print(f"Computed surface area for Room ID {roomEntry['room_id']}: {computedSurface}")
            print(f"Extracted surface area for Room ID {roomEntry['room_id']}: {surfaceT}")
            if (abs(surfaceT - computedSurface) <= 2.5):
                roomEntry["area_meters_sq"] = surfaceT
                print(f"Warning: Surface area mismatch for Room ID {roomEntry['room_id']}: extracted {surfaceT} vs computed {computedSurface}")

            #print(f"Found words for Room ID {roomEntry['room_id']}: {len(room_words)} words.")
            roomEntry["words"] = room_words

    def search_text(self, TextJson, bbox , offset):
        words = []

        x_offset, y_offset = offset

        target = {
            'x0': bbox['x'] + x_offset,
            'y0': bbox['y'] + y_offset,
            'x1': bbox['x'] + bbox['width'] + x_offset,
            'y1': bbox['y'] + bbox['height'] + y_offset,
        }

        new_text_regions = []

        for entry in TextJson["text_regions"]:
            eb = entry.get('image_coordinates', {})

            word_box = {
                'x0': int(eb['x0']),
                'y0': int(eb['y0']),
                'x1': int(eb['x1']),
                'y1': int(eb['y1']),
            }

            overlap = not (
                word_box['x1'] <= target['x0'] or
                word_box['x0'] >= target['x1'] or
                word_box['y1'] <= target['y0'] or
                word_box['y0'] >= target['y1']
            )

            if overlap:
                words.append(entry.get('text', ''))
            else:
                new_text_regions.append(entry)

        TextJson["text_regions"] = new_text_regions
        return words

    def clean_json(self):
        for pageContent in self.pages_content:
            if pageContent.rooms:
                if (pageContent.rooms):
                    self.clean_jsonRooms(pageContent.rooms)

    def clean_jsonRooms(self,data):
        FIELDS_TO_REMOVE = {
                        "rectangles",
                        "area_pixels",
                        "perimeter_pixels",
                        "centroid",
                        "rectangles_count",
                    }
        if isinstance(data, dict):
            # Remove unwanted keys at this level
            for key in list(data.keys()):
                if key in FIELDS_TO_REMOVE:
                    del data[key]
                else:
                    self.clean_jsonRooms(data[key])

        elif isinstance(data, list):
            for item in data:
                self.clean_jsonRooms(item)

    def _save_results(self):
        """Step 4: Save all results to output directory."""
        import json
        from datetime import datetime
        
        for page in self.pages_content:
            page_dir = self.output_dir / f"page_{page.page_number:03d}"
            page_dir.mkdir(exist_ok=True)
            
            # Save page image
            if page.page_image:
                page.page_image.save(page_dir / "page_original.png")
            
            # Save plan
            if page.plan:
                page.plan.save(page_dir / "plan.png")
            
            # Save info panel
            if page.info_panel:
                page.info_panel.save(page_dir / "info_panel.png")
            
            # Save predicted plan with detections
            if page.predicted_plan:
                page.predicted_plan.save(page_dir / "plan_detected.png")
            
            # Save extracted images
            if page.images:
                for idx, img in enumerate(page.images):
                    img.save(page_dir / f"extracted_image_{idx + 1:02d}.png")
            
            # Save text data
            if page.text:
                with open(page_dir / "text_data.json", "w", encoding="utf-8") as f:
                    json.dump(page.text, f, indent=2, ensure_ascii=False)
            # Save OpenAI grouping
            if hasattr(page, 'openAIGrouping') and page.openAIGrouping:
                with open(page_dir / "InfoPanelGrouping.json", "w", encoding="utf-8") as f:
                    json.dump(page.openAIGrouping, f, indent=2, ensure_ascii=False)
            
            # Save rooms data
            if page.rooms:
                with open(page_dir / "rooms_detected.json", "w", encoding="utf-8") as f:
                    # Convert any non-serializable objects
                    rooms_data = self._make_serializable(page.rooms)
                    json.dump(rooms_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved page {page.page_number} results to {page_dir}")
        
        # Save pipeline summary
        summary = {
            "pdf_path": str(self.pdf_path),
            "output_dir": str(self.output_dir),
            "pages_processed": len(self.pages_content),
            "timestamp": datetime.now().isoformat(),
            "pages": [
                {
                    "page_number": page.page_number,
                    "has_text": page.text is not None,
                    "has_plan": page.plan is not None,
                    "has_info_panel": page.info_panel is not None,
                    "has_predictions": page.predicted_plan is not None,
                    "rooms_count": len(page.rooms.get('rooms', [])) if page.rooms else 0,
                    "images_extracted": len(page.images) if page.images else 0
                }
                for page in self.pages_content
            ]
        }
        
        with open(self.output_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pipeline summary saved to {self.output_dir / 'pipeline_summary.json'}")
    
    @staticmethod
    def _make_serializable(obj):
        """Convert non-serializable objects to JSON-serializable format."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: ArchitecturalPlanPipeline._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ArchitecturalPlanPipeline._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process architectural PDFs and detect rooms"
    )
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("-o", "--output", default="pipeline_output", help="Output directory")
    parser.add_argument("--model", default="best.pt", help="Path to detection model")
    parser.add_argument("--stairs-model", default="best_stairs.pt", help="Path to stairs model")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PDF rendering")
    
    args = parser.parse_args()
    
    pipeline = ArchitecturalPlanPipeline(
        pdf_path=args.pdf,
        output_dir=args.output,
        model_path=args.model,
        stairs_model_path=args.stairs_model,
        dpi=args.dpi
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
