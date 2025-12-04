"""
Simple Section Text Extractor - Extracts raw text from info panel sections.

Output: Clean JSON with section names and their text content.
No manual parsing - let the LLM do the understanding.
"""

import cv2
import numpy as np
import pytesseract
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class SimpleExtractor:
    """Simple extractor - just OCR text per section, no parsing."""
    
    def __init__(self):
        pass
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using OCR."""
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not load image: {image_path}")
            return ""
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        # Threshold to clean up
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR - try German first, fallback to default
        try:
            text = pytesseract.image_to_string(binary, lang='deu', config='--psm 6')
        except:
            text = pytesseract.image_to_string(binary, config='--psm 6')
        
        # Clean up text
        text = text.strip()
        
        # Remove excessive whitespace while preserving structure
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        return '\n'.join(lines)
    
    def process_plan_sections(self, sections_dir: str, plan_name: str) -> Dict[str, Any]:
        """
        Process all section images for a single plan.
        
        Returns simple structure:
        {
            "plan_name": "...",
            "extraction_date": "...",
            "sections": {
                "ARCHITEKT": "text content...",
                "BAUHERR": "text content...",
                ...
            }
        }
        """
        sections_path = Path(sections_dir)
        
        result = {
            'plan_name': plan_name,
            'extraction_date': datetime.now().isoformat(),
            'sections': {}
        }
        
        # Find all section images for this plan
        pattern = f"{plan_name}_group_*.png"
        section_files = list(sections_path.glob(pattern))
        
        if not section_files:
            # Try without underscore
            pattern = f"{plan_name}*_group_*.png"
            section_files = list(sections_path.glob(pattern))
        
        logger.info(f"Found {len(section_files)} sections for plan: {plan_name}")
        
        for section_file in sorted(section_files):
            # Extract section type from filename
            # e.g., "input1_group_ARCHITEKT.png" -> "ARCHITEKT"
            name_parts = section_file.stem.split('_group_')
            if len(name_parts) >= 2:
                section_type = name_parts[-1]
            else:
                section_type = "UNKNOWN"
            
            # Extract text
            text = self.extract_text_from_image(str(section_file))
            
            if text:
                # If section already exists (e.g., ARCHITEKT_2), merge or add numbered
                if section_type in result['sections']:
                    # Append with separator
                    result['sections'][section_type] += f"\n---\n{text}"
                else:
                    result['sections'][section_type] = text
                
                logger.info(f"  ✓ {section_type}: {len(text)} chars")
            else:
                logger.info(f"  ✗ {section_type}: no text extracted")
        
        return result
    
    def save_to_json(self, data: Dict[str, Any], output_path: str):
        """Save extracted data to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved to: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple text extraction from sections')
    parser.add_argument('sections_dir', help='Directory containing section images')
    parser.add_argument('plan_name', help='Base name of the plan')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    
    args = parser.parse_args()
    
    extractor = SimpleExtractor()
    result = extractor.process_plan_sections(args.sections_dir, args.plan_name)
    
    # Print summary
    print("\n" + "="*60)
    print(f"EXTRACTED: {result['plan_name']}")
    print("="*60)
    
    for section_name, text in result['sections'].items():
        preview = text[:100].replace('\n', ' ')
        if len(text) > 100:
            preview += "..."
        print(f"\n📄 {section_name}:")
        print(f"   {preview}")
    
    # Save to JSON
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.sections_dir) / f"{args.plan_name}_simple.json"
    
    extractor.save_to_json(result, str(output_path))


if __name__ == "__main__":
    main()

