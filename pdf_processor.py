import logging
import time
from pathlib import Path
from typing import Dict, Any

from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from paddleocr import PaddleOCRVL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_log = logging.getLogger(__name__)

# Constants
IMAGE_RESOLUTION_SCALE = 5.0


class PDFProcessor:
    """PDF processing pipeline using Docling and PaddleOCR-VL."""

    def __init__(self, input_pdf: str, output_base_dir: str = "output"):
        """Initialize the PDF processor.
        
        Args:
            input_pdf: Path to input PDF file
            output_base_dir: Base output directory for results
        """
        self.input_pdf = Path(input_pdf)
        self.output_base_dir = Path(output_base_dir)
        self.doc_filename = self.input_pdf.stem
        
        # Create output directories for Docling and PaddleOCR
        self.output_docling = self.output_base_dir / "output_docling"
        self.output_ocr = self.output_base_dir / "output_ocr"
        
        self.output_docling.mkdir(parents=True, exist_ok=True)
        self.output_ocr.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR pipeline
        self.ocr_pipeline = PaddleOCRVL()

    def setup_docling_pipeline(self) -> DocumentConverter:
        """Setup Docling document converter with optimal settings.
        
        Returns:
            Configured DocumentConverter instance
        """
        # Configure PDF processing options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        
        # Create and return converter with PDF options
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process_docling(self, conv_res) -> list:
        """Process PDF with Docling and save outputs (JSON, MD, PNG).
        
        Args:
            conv_res: Docling conversion result
            
        Returns:
            List of extracted page image paths
        """
        _log.info("Processing with Docling...")
        
        # Save JSON representation
        docling_json_file = self.output_docling / f"{self.doc_filename}-docling.json"
        conv_res.document.save_as_json(docling_json_file)
        _log.info(f"Saved JSON: {docling_json_file}")
        
        # Save markdown with embedded images
        md_embedded = self.output_docling / f"{self.doc_filename}-embedded-images.md"
        conv_res.document.save_as_markdown(md_embedded, image_mode=ImageRefMode.EMBEDDED)
        _log.info(f"Saved embedded markdown: {md_embedded}")
        
        # Save markdown with referenced images
        md_referenced = self.output_docling / f"{self.doc_filename}-referenced-images.md"
        conv_res.document.save_as_markdown(md_referenced, image_mode=ImageRefMode.REFERENCED)
        _log.info(f"Saved referenced markdown: {md_referenced}")
        
        # Extract and save page images as PNG files
        page_images = []
        for page_no, page in conv_res.document.pages.items():
            page_image_filename = self.output_docling / f"{self.doc_filename}-page-{page_no}.png"
            with page_image_filename.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")
            page_images.append(str(page_image_filename))
            _log.info(f"Saved page {page_no} image: {page_image_filename}")
        
        _log.info(f"Total pages extracted: {len(page_images)}")
        return page_images

    def process_with_paddleocr(self, image_path: str) -> None:
        """Process image with PaddleOCR-VL and save results.
        
        Args:
            image_path: Path to image file
        """
        _log.info(f"Processing with PaddleOCR-VL: {image_path}")
        
        try:
            # Run OCR predictions on image
            output = self.ocr_pipeline.predict(image_path)
            _log.info(f"OCR predictions completed for: {image_path}")
            
            # Create output directory for OCR results
            image_name = Path(image_path).stem
            image_ocr_dir = self.output_ocr / f"{image_name}_ocr"
            image_ocr_dir.mkdir(parents=True, exist_ok=True)
            _log.info(f"Created OCR output directory: {image_ocr_dir}")
            
            # Save OCR results if available
            if output:
                for res in output:
                    if hasattr(res, 'save_to_json'):
                        res.save_to_json(save_path=str(image_ocr_dir))
                        _log.info(f"Saved OCR JSON results for: {image_name}")
                    if hasattr(res, 'save_to_markdown'):
                        res.save_to_markdown(save_path=str(image_ocr_dir))
                        _log.info(f"Saved OCR markdown results for: {image_name}")
            else:
                _log.warning(f"No OCR output generated for: {image_path}")
            
            _log.info(f"PaddleOCR results saved to: {image_ocr_dir}")
            
        except Exception as e:
            _log.error(f"Error processing with PaddleOCR: {e}", exc_info=True)

    def process(self) -> Dict[str, Any]:
        """Execute the complete PDF processing pipeline.
        
        Returns:
            Dictionary with processing status and results
        """
        _log.info(f"Starting PDF processing: {self.input_pdf}")
        _log.info(f"Output directories: {self.output_docling} | {self.output_ocr}")
        start_time = time.time()
        
        try:
            # Step 1: Convert PDF with Docling
            _log.info("Converting PDF with Docling...")
            doc_converter = self.setup_docling_pipeline()
            conv_res = doc_converter.convert(str(self.input_pdf))
            _log.info(f"PDF conversion completed. Document has {len(conv_res.document.pages)} pages")
            
            # Step 2: Process with Docling to generate outputs
            _log.info("Extracting pages and generating outputs...")
            page_images = self.process_docling(conv_res)
            
            # Step 3: Process extracted pages with PaddleOCR
            _log.info(f"Processing {len(page_images)} images with PaddleOCR-VL...")
            for idx, image_path in enumerate(page_images, 1):
                _log.info(f"[{idx}/{len(page_images)}] Processing image: {image_path}")
                self.process_with_paddleocr(image_path)
            
            processing_time = time.time() - start_time
            
            # Prepare success result
            result = {
                "status": "success",
                "filename": self.doc_filename,
                "output_docling_directory": str(self.output_docling),
                "output_ocr_directory": str(self.output_ocr),
                "processing_time_seconds": round(processing_time, 2)
            }
            
            _log.info(f"PDF processing completed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            _log.error(f"Error during PDF processing: {e}", exc_info=True)
            # Return error result
            return {
                "status": "failed",
                "filename": self.doc_filename,
                "error": str(e)
            }


def main():
    """Main entry point for PDF processing."""
    input_pdf = "input.pdf"
    output_base_dir = "output"
    
    # Initialize processor and run pipeline
    processor = PDFProcessor(input_pdf, output_base_dir)
    result = processor.process()
    
    # Display results
    print("\n" + "="*60)
    print("PDF PROCESSING COMPLETE")
    print("="*60)
    print(f"Status: {result['status']}")
    print(f"Output directory: {result.get('output_docling_directory', 'N/A')}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
