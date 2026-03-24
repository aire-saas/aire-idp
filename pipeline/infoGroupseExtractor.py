"""
Azure OpenAI Vision API Image Text Extractor

Uses Azure OpenAI's Vision API to extract and structure text from images into JSON format.
All information from the image is preserved, and no external information is added.
"""

import json
import logging
import base64
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from openai import AzureOpenAI
from PIL.Image import Image as PILImage
import io
import base64
from process import PDFProcessor, PageContent


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)



class InfoGroupExtractor:
    """Extract and structure text from images using Azure OpenAI Vision API."""
    
    def __init__(self, api_key: Optional[str] = None, deployment_name: str = "gpt-4o", pdfProcessor: PDFProcessor | None = None):
        """
        Initialize the Azure OpenAI Image Extractor.
        
        Args:
            api_key: Azure OpenAI API key. If None, will try to get from AZURE_OPENAI_API_KEY env var.
            deployment_name: Azure OpenAI deployment name. Default is "gpt-4o".
            pdfProcessor: PDF processor instance for processing pages.
        """
        # Get Azure OpenAI credentials from environment or parameters
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        print("Azure OpenAI key: " + str(bool(self.api_key)))
        print("Azure OpenAI endpoint: " + str(self.endpoint))
        print("Azure OpenAI API version: " + str(self.api_version))
        
        if not self.api_key or not self.endpoint:
            raise ValueError(
                "Azure OpenAI credentials not found. "
                "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
            )
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
        self.deployment_name = deployment_name
        logger.info(f"Initialized Azure OpenAI Image Extractor with deployment: {deployment_name}")

        self.pdfProcessor = pdfProcessor
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_image_mime_type(self, image_path: str) -> str:
        """
        Get MIME type from image file extension.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MIME type string
        """
        extension = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')
    
    def extract_and_structure2(self, image: PILImage) -> Dict[str, Any]:
        """
        Extract text from a PIL image and structure it into JSON using Azure OpenAI Vision API.

        Args:
            image: PIL Image object

        Returns:
            Dictionary containing structured JSON data extracted from the image
        """
        if image is None:
            raise ValueError("Input image is None")

        logger.info("Processing PIL image")

        # -------------------------
        # Encode PIL image to base64
        # -------------------------
        buffered = io.BytesIO()

        # PNG is safe and lossless for OCR / vision
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mime_type = "image/png"

        # Build prompt
        prompt = self._build_extraction_prompt()

        try:
            # Call Azure OpenAI Vision API
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )

            # Extract response text
            response_text = response.choices[0].message.content

            # Parse JSON from response
            structured_data = self._parse_json_response(response_text)

            # Extract metadata
            project_name = self._extract_project_name(structured_data)
            plan_number = self._extract_plan_number(structured_data)

            structured_data["_metadata"] = {
                "project_name": project_name if project_name else None,
                "plan_number": plan_number if plan_number else None
            }

            logger.info("✓ Successfully extracted and structured text from image")
            if project_name:
                logger.info(f"  Project name: {project_name}")
            if plan_number:
                logger.info(f"  Plan number: {plan_number}")

            return structured_data, response_text

        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise
    
    def extract_and_structure(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image and structure it into JSON using Azure OpenAI Vision API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing structured JSON data extracted from the image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Processing image: {image_path}")
        
        # Encode image
        base64_image = self.encode_image(image_path)
        mime_type = self.get_image_mime_type(image_path)
        
        # Build prompt
        prompt = self._build_extraction_prompt()
        
        try:
            # Call Azure OpenAI Vision API
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Parse JSON from response
            structured_data = self._parse_json_response(response_text)
            
            # Extract project name and plan number from the structured data
            project_name = self._extract_project_name(structured_data)
            plan_number = self._extract_plan_number(structured_data)
            
            # Add metadata
            structured_data["_metadata"] = {
                "project_name": project_name if project_name else None,
                "plan_number": plan_number if plan_number else None
            }
            
            logger.info("✓ Successfully extracted and structured text from image")
            if project_name:
                logger.info(f"  Project name: {project_name}")
            if plan_number:
                logger.info(f"  Plan number: {plan_number}")
            return structured_data
            
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise
    
    def _build_extraction_prompt(self) -> str:
        """
        Build the prompt for Azure OpenAI Vision API.
        
        Returns:
            Prompt string with instructions for extraction
        """
        return """You are an expert at extracting and structuring text from images.

LANGUAGE CONTEXT:
- ALL text in this image is in GERMAN language
- Field names, labels, and keys are in German (e.g., "Name", "Adresse", "Telefon", "E-Mail", "PLZ", "Stadt")
- Dates follow German format (e.g., DD.MM.YYYY)
- Addresses follow German format (e.g., Straße, PLZ Stadt)
- Use German field names in the structured_fields (e.g., "name", "adresse", "telefon", "email", "plz", "stadt", "strasse")

TEXT EXTRACTION - MAXIMUM EFFORT REQUIRED:
- Pay EXTREME attention to text extraction - this is the most critical part
- Read EVERY single character, word, number, and symbol carefully
- Scan the ENTIRE image systematically - do not miss any text areas
- Look carefully at small text, fine print, and text in corners or margins
- Pay special attention to text that might be partially obscured, faded, or in unusual fonts
- Read text in all orientations (horizontal, vertical, rotated)
- Extract text from headers, footers, watermarks, stamps, and annotations
- Be very careful with similar-looking characters (0 vs O, 1 vs I vs l, rn vs m, etc.)
- Double-check numbers, dates, phone numbers, and postal codes for accuracy
- Read text in tables cell by cell - do not skip any cells
- Extract text from labels, captions, legends, and all annotations
- If text is unclear, try your best to read it - include your best interpretation
- Do multiple passes: scan once for structure, then again for all text details

Your task is:
1. Read ALL text visible in this image with MAXIMUM CARE and ATTENTION
2. Extract EVERY piece of information present in the image - leave NOTHING out
3. Structure the information into a well-organized JSON format
4. Preserve ALL information - nothing should be lost

CRITICAL RULES:
- Extract ONLY information that is actually visible in the image
- DO NOT add any information that is not present in the image
- DO NOT make assumptions or infer information not shown
- Include ALL text, numbers, dates, addresses, names, etc. that appear in the image
- If something is unclear or partially visible, include it as-is with a note
- Preserve the exact text as it appears (including typos if visible)
- Maintain the structure and organization visible in the image

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown code blocks, no explanations before or after).
The JSON should be well-structured and organized based on what you see in the image.

For structured documents (forms, tables, lists), organize the data accordingly:
- Forms: Extract field labels and their values
- Tables: Extract headers and all rows
- Lists: Extract all items
- Addresses: Structure into street, city, postal code, country (keep within the section where they appear)
- Contact info: Keep phone, email, website, etc. within the section where they appear
- Dates: Extract in readable format (keep within the section where they appear)
- Numbers: Preserve exact values

IMPORTANT: Keep all information within its original section. If an email appears in a "Contact" section, keep it in that section's structured_fields. Do NOT extract emails, addresses, phone numbers, etc. into separate global arrays - they must remain in their respective sections.

STRUCTURE THE JSON EXACTLY AS THE INFORMATION IS STRUCTURED IN THE IMAGE:
- Follow the visual layout and organization of the image
- If the image has sections, create sections in the JSON
- If the image has tables, structure them as tables
- If the image has forms with labels and values, structure them accordingly
- Use the field names and labels exactly as they appear in the image (in German)
- Preserve the hierarchy and relationships visible in the image
- Structure the JSON to match how the information is organized visually in the image

PROJECT NAME EXTRACTION:
- Look for project name fields in the image (e.g., "Projekt", "Projektname", "Projekt-Nr", "Projektnummer", "Projektbezeichnung")
- Extract the project name value if it appears in the image
- The project name should be included in the appropriate section where it appears
-   The JSON file must start with a metadata object that contains the project name and the project number. This metadata should appear at the very top of the JSON structure, before any other fields.

Remember:
- TEXT EXTRACTION IS CRITICAL - spend extra time and effort reading every word carefully
- ALL text is in GERMAN - use German field names and understand German context
- Include EVERYTHING visible in the image - no text should be missed
- Do NOT add information not present in the image
- Keep emails, addresses, phone numbers, dates, etc. within their original sections - do NOT extract them into separate global arrays
- Each section should contain all its information in structured_fields
- Use German field names (e.g., "telefon" not "phone", "adresse" not "address", "plz" not "postal_code")
- Be especially careful with text extraction - read slowly and thoroughly
- If you're unsure about a character, include your best interpretation rather than skipping it
- Return ONLY valid JSON"""
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from Azure OpenAI response.
        
        Args:
            response_text: Raw response text from Azure OpenAI
            
        Returns:
            Parsed JSON dictionary
        """
        response_text = response_text.strip()
        
        # Try to find JSON in the response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                logger.warning(f"Response text: {response_text[:500]}")
                # Return fallback structure
                return {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_text,
                    "extracted_text": response_text
                }
        
        # If no JSON found, return the raw text
        logger.warning("No JSON found in response, returning raw text")
        return {
            "extracted_text": response_text,
            "note": "Response was not in JSON format"
        }
    
    def _extract_project_name(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Try to extract project name from the structured data.
        Searches for common German project name fields.
        
        Args:
            data: Structured data dictionary
            
        Returns:
            Project name if found, None otherwise
        """
        project_fields = [
            "projekt_name", "projektname", "projekt", "projekt_nr", "projektnummer",
            "projektbezeichnung", "projekt_bezeichnung", "projektbezeichnung",
            "project_name", "project", "name"
        ]
        
        def search_dict(d: Any, depth: int = 0) -> Optional[str]:
            """Recursively search for project name in dictionary."""
            if depth > 5:  # Limit recursion depth
                return None
            
            if isinstance(d, dict):
                # Check direct keys
                for key in project_fields:
                    if key.lower() in str(d.keys()).lower():
                        # Check if any key contains the project field name
                        for dict_key, dict_value in d.items():
                            if any(pf in dict_key.lower() for pf in project_fields):
                                if isinstance(dict_value, str) and dict_value.strip():
                                    return dict_value.strip()
                                elif isinstance(dict_value, dict):
                                    # Check nested dict for name/value
                                    if "name" in dict_value and isinstance(dict_value["name"], str):
                                        return dict_value["name"].strip()
                                    if "value" in dict_value and isinstance(dict_value["value"], str):
                                        return dict_value["value"].strip()
                
                # Check for direct matches
                for key, value in d.items():
                    key_lower = key.lower()
                    if any(pf in key_lower for pf in project_fields):
                        if isinstance(value, str) and value.strip():
                            return value.strip()
                        elif isinstance(value, dict):
                            # Check nested dict
                            result = search_dict(value, depth + 1)
                            if result:
                                return result
                
                # Recursively search nested dictionaries
                for value in d.values():
                    result = search_dict(value, depth + 1)
                    if result:
                        return result
            
            elif isinstance(d, list):
                for item in d:
                    result = search_dict(item, depth + 1)
                    if result:
                        return result
            
            return None
        
        return search_dict(data)
    
    def _extract_plan_number(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Try to extract plan number from the structured data.
        Prioritizes plan number fields over project number fields.
        Looks specifically in WERKPLANUNG/PLANINFO sections.
        
        Args:
            data: Structured data dictionary
            
        Returns:
            Plan number if found, None otherwise
        """
        plan_number_fields = [
            "plan_nr", "plan-nr", "plan_nummer", "plan-nummer",
            "plan_nr.", "plan_nummer.", "plan_number", "plan_no",
            "plannummer", "plannr", "nummer", "nr"
        ]
        
        plan_info_fields = [
            "werkplanung", "planinfo", "plan_info", "plan", "zeichnung"
        ]
        
        def is_valid_plan_number(value: str) -> bool:
            """Check if value looks like a valid plan number (not a legend entry)."""
            if not value or not isinstance(value, str):
                return False
            
            value_clean = value.strip()
            
            # Reject if it looks like a legend entry (contains "=" or ":" with explanation)
            if "=" in value_clean or ":" in value_clean:
                # Check if it's a simple assignment (e.g., "05 = Fensternummerierung")
                parts = re.split(r'[=:]', value_clean)
                if len(parts) == 2 and len(parts[0].strip()) <= 5:
                    return False
            
            # Reject single digits or very short numeric values (likely legend codes)
            if len(value_clean) <= 2 and value_clean.isdigit():
                return False
            
            # Accept patterns that look like plan numbers:
            # - Contains letters and numbers (e.g., "ARC 50G01", "APN-WP-01")
            # - Contains dashes (e.g., "1503-656", "70675-ARC-50001")
            # - Alphanumeric with spaces (e.g., "ARC 50G01")
            # - Minimum 3 characters
            if len(value_clean) >= 3:
                # Must contain at least one letter (to distinguish from pure numbers)
                if re.search(r'[A-Za-z]', value_clean):
                    # Accept alphanumeric with dashes, spaces, slashes
                    if re.match(r'^[A-Za-z0-9\-\s/]+$', value_clean):
                        return True
                # Also accept numeric patterns with dashes (like "1503-656")
                elif re.match(r'^\d+[\-\s]\d+', value_clean):
                    return True
            
            return False
        
        def search_dict(d: Any, depth: int = 0, in_plan_section: bool = False) -> Optional[str]:
            """Recursively search for plan number in dictionary."""
            if depth > 5:  # Limit recursion depth
                return None
            
            if isinstance(d, dict):
                # First, check if we're in a plan-related section
                for key, value in d.items():
                    key_lower = key.lower()
                    if any(pif in key_lower for pif in plan_info_fields):
                        in_plan_section = True
                
                # Priority 1: Look for explicit plan number fields
                for key, value in d.items():
                    key_lower = key.lower()
                    
                    # Check for explicit plan number fields
                    if any(pnf in key_lower for pnf in plan_number_fields):
                        if isinstance(value, str) and is_valid_plan_number(value):
                            return value.strip()
                        elif isinstance(value, dict):
                            result = search_dict(value, depth + 1, in_plan_section)
                            if result:
                                return result
                    
                    # If in plan section, look for "plan_nr" or similar in nested content
                    if in_plan_section:
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                nested_key_lower = nested_key.lower()
                                if any(pnf in nested_key_lower for pnf in plan_number_fields):
                                    if isinstance(nested_value, str) and is_valid_plan_number(nested_value):
                                        return nested_value.strip()
                
                # Priority 2: Look in plan info sections for plan_nr fields
                plan_section_candidates = []
                for key, value in d.items():
                    key_lower = key.lower()
                    if any(pif in key_lower for pif in plan_info_fields):
                        if isinstance(value, dict):
                            # Priority: First look for "arc" field (most reliable plan identifier)
                            if "arc" in value and isinstance(value["arc"], str):
                                if is_valid_plan_number(value["arc"]):
                                    plan_section_candidates.append(value["arc"].strip())
                            
                            # Then look for plan number fields
                            for nested_key, nested_value in value.items():
                                nested_key_lower = nested_key.lower()
                                # Skip "arc" if we already found it above
                                if nested_key_lower == "arc":
                                    continue
                                # Check for plan number fields
                                if any(pnf in nested_key_lower for pnf in plan_number_fields):
                                    if isinstance(nested_value, str) and is_valid_plan_number(nested_value):
                                        plan_section_candidates.append(nested_value.strip())
                                elif isinstance(nested_value, dict):
                                    # Check nested content
                                    if "content" in nested_value or isinstance(nested_value, dict):
                                        for sub_key, sub_value in nested_value.items():
                                            sub_key_lower = sub_key.lower()
                                            if sub_key_lower == "arc" and isinstance(sub_value, str):
                                                if is_valid_plan_number(sub_value):
                                                    plan_section_candidates.append(sub_value.strip())
                                            elif any(pnf in sub_key_lower for pnf in plan_number_fields):
                                                if isinstance(sub_value, str) and is_valid_plan_number(sub_value):
                                                    plan_section_candidates.append(sub_value.strip())
                
                # Return the first valid candidate (prioritized by order above)
                if plan_section_candidates:
                    return plan_section_candidates[0]
                
                # Recursively search nested dictionaries
                for value in d.values():
                    result = search_dict(value, depth + 1, in_plan_section)
                    if result:
                        return result
            
            elif isinstance(d, list):
                for item in d:
                    result = search_dict(item, depth + 1, in_plan_section)
                    if result:
                        return result
            
            return None
        
        return search_dict(data)
    
    def save_to_json(self, data: Dict[str, Any], output_path: str) -> None:
        """
        Save extracted data to JSON file.
        
        Args:
            data: Dictionary containing extracted data
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved extracted data to: {output_path}")

    def processAllPages(self):
        count = 0
        for page_content in self.pdfProcessor.pages_content:
                try:
                    result, response_text = self.extract_and_structure2(image= page_content.info_panel)
                    page_content.openAIGrouping = result
                    count += 1
                except Exception:
                    continue
        logger.info("successfully processed info panel textual elements:" + str(count) + "/" + str(len(self.pdfProcessor.pages_content)))

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract and structure text from images using Azure OpenAI Vision API'
    )
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('-o', '--output', help='Output JSON file path (optional)')
    parser.add_argument(
        '--api-key',
        help='Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var)',
        default=None
    )
    parser.add_argument(
        '--deployment',
        help='Azure OpenAI deployment name (default: gpt-4o)',
        default='gpt-4o'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = InfoGroupExtractor(api_key=args.api_key, deployment_name=args.deployment)
    
    # Extract and structure
    result = extractor.extract_and_structure(args.image_path)
    
    # Save to file if output path provided
    if args.output:
        extractor.save_to_json(result, args.output)
    else:
        # Print to stdout
        print("INFO PANEL GROUPING DONE")
        #print(json.dumps(result, ensure_ascii=False, indent=2))
    
    logger.info("✓ Extraction complete")


if __name__ == "__main__":
    main()
