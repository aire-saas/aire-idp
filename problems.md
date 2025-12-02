# Known Issues and Problems

## 1. OCR Output Limitations
- The OCR engine may **ignore some information** present in the PDF document, particularly:
  - Complex formatting and layout structures
  - Special characters and symbols that are not in the OCR model's training data
  - Text embedded in images or graphics

- **Tables are treated as images**: Tables within the PDF are often extracted as image elements rather than structured data. This means:
  - Table content is not converted to structured formats (e.g., CSV, structured JSON)
  - Table data cannot be easily parsed or searched
  - Requires additional post-processing to extract tabular information from the OCR'd images

## 2. Docling Output Characteristics
- **Docling output is stable and consistent**: The conversion process reliably extracts content from PDFs with predictable results
- **All text is captured**: Docling successfully retrieves all text content present in the PDF
- **Unstructured data format**: 
  - Docling outputs markdown and JSON formats that contain all text but in an **unstructured manner**
  - Requires additional processing and treatment to organize and structure the data meaningfully
  - No inherent separation of tables, headers, or semantic elements

## 3. Possible Solutions

### Image Division Strategy
When dividing extracted images into smaller sections focusing on specific content areas (e.g., tables, sections):
- **PaddleOCR performs significantly better** with partial images containing focused content
- **Improved structure**: Results in more organized and structured JSON/markdown outputs
- **Better handling of special elements**: Successfully processes micro-images such as material images, diagrams, and embedded graphics
- **Structured output**: Generates properly formatted JSON and markdown files with better semantic organization

### Implementation Approach
1. Detect sections, tables, or logical content areas within the full page images
2. Divide large images into smaller, focused image segments
3. Process each segment independently with PaddleOCR
4. Aggregate results while maintaining structure and hierarchy
5. Merge the structured outputs into a cohesive document

### Bounding Box Reconstruction (Alternative Solution)
The Docling JSON output contains valuable structural information that can be leveraged:
- **Complete word positions**: Each extracted word includes its bounding box (bbox) coordinates
- **Spatial information**: Bbox data reveals the precise location and dimensions of text elements
- **Reconstruction potential**: By analyzing bbox relationships, we can:
  - Detect and reconstruct table structures based on aligned bounding boxes
  - Identify logical sections and reading order
  - Group related text elements into semantic units
  - Rebuild the document with proper hierarchical structure

**Advantages of this approach**:
- No need to re-process images or run additional OCR
- Leverages existing, stable Docling output
- Preserves all extracted text and metadata
- Can be implemented as a post-processing step on the JSON data
- More computationally efficient than image division
