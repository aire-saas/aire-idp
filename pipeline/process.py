import logging
import json
import ast
import base64
from pathlib import Path
from typing import List, Any
from io import BytesIO

import fitz
from PIL import Image

# ------------------------
# Config
# ------------------------
RENDER_DPI = 150
PDF_DPI = 72
SCALE = RENDER_DPI / PDF_DPI

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ------------------------
# Helper functions
# ------------------------


# ------------------------
# Data container
# ------------------------
class PageContent:
    def __init__(self, page_image: Image.Image,page_number: int, text: Any, images: List[Image.Image] | None = None, info_panel: Image.Image | None = None,
        plan: Image.Image | None = None, offset: tuple | None = (0, 0), predicted_plan: Image.Image | None = None, rooms: Any  | None = None, openAIGrouping: Any | None = None):
        self.page_image = page_image
        self.page_number = page_number
        self.text = text
        self.images = images
        self.info_panel = info_panel
        self.plan = plan
        self.offset = offset
        self.predicted_plan = predicted_plan
        self.rooms = rooms
        self.openAIGrouping = openAIGrouping

# ------------------------
# PDF Processor
# ------------------------
class PDFProcessor:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.pages_content: List[PageContent] = []

    def make_json_serializable(self, obj):
        """
        Convert non-JSON-serializable objects (like bytes) to JSON-serializable format.
        Recursively processes dictionaries and lists.
        """
        if isinstance(obj, bytes):
            # Convert bytes to base64 string
            return base64.b64encode(obj).decode('utf-8')
        elif isinstance(obj, dict):
            # Recursively process dictionary values
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            # Recursively process list items
            return [self.make_json_serializable(item) for item in obj]
        else:
            # Return other types as-is
            return obj

    def extract_text_regions(self, obj, regions=None):
        """
        Extract text regions from PyMuPDF dict structure in a clean format.
        Handles nested blocks and lines to extract all text with coordinates.
        """
        if regions is None:
            regions = []
        
        if isinstance(obj, dict):
            # Check if this is a text block
            if "lines" in obj and obj.get("type") == 0:
                # Process lines in this block
                for line in obj.get("lines", []):
                    if isinstance(line, dict) and "spans" in line:
                        for span in line.get("spans", []):
                            if isinstance(span, dict) and "text" in span:
                                bbox = span.get("bbox", [0, 0, 0, 0])
                                if len(bbox) == 4:
                                    x0, y0, x1, y1 = bbox
                                    regions.append({
                                        "text": span.get("text", ""),
                                        "image_coordinates": {
                                            "x0": round(x0 * SCALE, 2),
                                            "y0": round(y0 * SCALE, 2),
                                            "x1": round(x1 * SCALE, 2),
                                            "y1": round(y1 * SCALE, 2)
                                        }
                                    })
            
            # Recursively process nested dictionaries
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    self.extract_text_regions(v, regions)
                    
        elif isinstance(obj, list):
            # Process list items
            for item in obj:
                if isinstance(item, (dict, list)):
                    self.extract_text_regions(item, regions)
        
        return regions

    def convert_geometry(self, obj):
        # Scale bbox and origin coordinates to match rendered image
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "bbox":
                    x0, y0, x1, y1 = v
                    # Store both original and scaled coordinates
                    out["bbox"] = [x0, y0, x1, y1]  # Original PDF coordinates
                    out["bbox_scaled"] = {
                        "x0": round(x0 * SCALE, 2), 
                        "y0": round(y0 * SCALE, 2), 
                        "x1": round(x1 * SCALE, 2), 
                        "y1": round(y1 * SCALE, 2)
                    }
                elif k == "origin":
                    x, y = v
                    out[k] = [x, y]  # Keep original values
                    out["origin_scaled"] = {
                        "x": round(x * SCALE, 2), 
                        "y": round(y * SCALE, 2)
                    }
                elif k == "dir":
                    dx, dy = v
                    out[k] = [dx, dy]  # Keep original values
                    out["dir_scaled"] = {
                        "dx": round(dx * SCALE, 2), 
                        "dy": round(dy * SCALE, 2)
                    }
                elif k == "lines" and len(v) > 0:
                    # Process lines recursively
                    out[k] = self.convert_geometry(v)
                
                elif isinstance(v, (dict, list)):
                    # Recursively process other nested structures
                    out[k] = self.convert_geometry(v)
                else:
                    # Keep primitive values as-is
                    out[k] = v
            return out
        elif isinstance(obj, list):
            return [self.convert_geometry(i) for i in obj if i]  # Filter out empty dicts
        return obj

    def process(self):
        doc = fitz.open(self.input_path)
        matrix = fitz.Matrix(SCALE, SCALE)
        pages_content = []

        for page_index, page in enumerate(doc):
            # ---- Render image with scaling matrix
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            page_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # ---- Extract text WITHOUT matrix first to get original coordinates
            raw_text = page.get_text("dict")
            
            # ---- Extract clean text regions
            text_regions = self.extract_text_regions(raw_text)
            
            # ---- Create clean structured output
            page_data = {
                "pdf_path": str(self.input_path),
                "page_number": page_index + 1,
                "image_dimensions": {
                    "width": pix.width,
                    "height": pix.height
                },
                "pdf_dimensions": {
                    "width": raw_text.get("width", 0),
                    "height": raw_text.get("height", 0)
                },
                "scale_factors": {
                    "x": SCALE,
                    "y": SCALE
                },
                "rendering_info": {
                    "dpi": RENDER_DPI,
                    "pdf_dpi": PDF_DPI,
                    "scale_factor": SCALE,
                    "note": "Image coordinates match the rendered image at the specified DPI"
                },
                "text_regions": text_regions
            }

            # ---- Extract embedded images
            images: List[Image.Image] = []
            for img_index, img_ref in enumerate(page.get_images(full=True)):
                try:
                    xref = img_ref[0]
                    pix_img = fitz.Pixmap(doc, xref)

                    # Skip images with NULL colorspace (can't be reliably converted)
                    if pix_img.colorspace is None or pix_img.colorspace.name == "null":
                        logger.info(f"Skipping image {img_index + 1} from page {page_index + 1} (NULL colorspace)")
                        continue

                    # Ensure image is RGB without alpha channel
                    if pix_img.alpha:  # Has alpha channel
                        pix_img = fitz.Pixmap(pix_img, 0)  # Remove alpha
                    
                    # Convert CMYK → RGB if needed
                    if pix_img.n >= 4:  # CMYK or other multi-channel
                        pix_img = fitz.Pixmap(fitz.csRGB, pix_img)

                    # Use PNG format which handles all cases better
                    img_data = pix_img.tobytes("png")
                    
                    # Create image from PNG data
                    img = Image.open(BytesIO(img_data))
                    img = img.convert("RGB")  # Ensure RGB format
                    images.append(img)
                    logger.info(f"Extracted image {img_index + 1} from page {page_index + 1}")
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index + 1} from page {page_index + 1}: {e}")

            pages_content.append(
                PageContent(
                    page_image=page_image,
                    page_number=page_index + 1,
                    text=page_data,
                    images=images
                )
            )
            logger.info(f"Processed page {page_index + 1}: extracted {len(text_regions)} text regions")

        doc.close()
        self.pages_content = pages_content
        return pages_content

    def visualize_output(self, output_folder="output"):
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        for page in self.pages_content:
            page_dir = output_dir / f"page_{page.page_number:03d}"
            page_dir.mkdir(exist_ok=True)

            page.page_image.save(page_dir / "page_image.png")
            
            # Convert to JSON-serializable format (handles bytes objects)
            serializable_text = self.make_json_serializable(page.text)
            
            with open(page_dir / "page_text.json", "w", encoding="utf-8") as f:
                json.dump(serializable_text, f, indent=4, ensure_ascii=False)

            logger.info(f"Saved page {page.page_number}")
            
            # Save extracted images
            for idx, img in enumerate(page.images):
                img.save(page_dir / f"extracted_image_{idx + 1:02d}.png")


if __name__ == "__main__":
    pdf_path = "Dataset/4.pdf"  # adjust path
    output_folder = "Dataset4ScaledOutput"

    processor = PDFProcessor(pdf_path)
    processor.process()
    processor.visualize_output(output_folder)
    logger.info("Processing complete")
