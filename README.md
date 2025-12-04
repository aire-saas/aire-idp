# Traceable Digital Twin Extraction from German Architectural Plans (PDF/Images)

## Team Members

| Name | Matrikel-No |
|------|-------------|
| Feres Ben Frej | 03770610 |
| Oussema Belkhiria | 03774160 |
| Oussama Jeddou | 037688191 |
| Ayoub Albouchi | 03745252 |

## Administrative Information

- **Examiner/Professor:** Prof. Holger Patzelt (TUM)
- **Supervisor:** Dr. Alex Christian
- **Host:** AI-RE UG (haftungsbeschränkt), Munich
- **Duration:** 14 weeks (Winter Semester 2025/26)
- **Team size:** 4 students (B.Sc./M.Sc. Informatics) working jointly on one codebase
- **Confidentiality:** Project uses non-public real-estate documents; NDA / data-processing agreement available on request.

---

## 1. Executive Summary

This interdisciplinary project (IDP) focuses on developing a production-grade spatial and geometric recognition system for German architectural floor plans, with the primary goal of extracting room boundaries, coordinates and accurate area calculations. The system combines computer vision (object detection, segmentation), geometric analysis (wireframe extraction, topology) and document AI (OCR, layout understanding) to create machine-readable digital twins of construction documents.

### Primary Deliverable: Question-Answering System

The AI models must be able to answer questions about plans, not merely extract data. The system will be evaluated on its ability to correctly answer questions such as:

- "How big is apartment Nr. 3?" → Answer: 85.6 m² (with provenance).
- "How many 2-room apartments are planned?" → Answer: 8 apartments (with list).
- "What is the total living space in building A?" → Answer: 1,245 m² (with calculation breakdown).

### Key Priorities (in Order of Importance)

#### 1. Room identification with accurate measurements (CRITICAL)

This is the most important deliverable.

- Detect all rooms with boundaries (polygons).
- Calculate accurate areas (median error ≤ 2.5%).
- Extract coordinates for each room.
- **Testable output:** the system can answer "How big is room X?" with a correct value in m².

#### 2. Wall/door/window detection with measurements

- Detect and classify openings.
- Measure wall thickness.
- Count doors and windows by type.
- **Testable output:** the system can answer "How many doors are in the building?".

#### 3. Question-answering capability with provenance

- Parse natural language questions.
- Query the extracted data.
- Return answers with source references.
- **Testable output:** demonstrator UI showing question answering with clickable sources.

**Scope adjustments:** given the workload, lower-priority tasks (e.g. material taxonomy, advanced legend parsing, compliance extraction) may be skipped if time-constrained. The final two weeks (weeks 13–14) are dedicated to report writing and presentation preparation.

---

## 2. Background and Motivation

Real-estate development workflows still rely on floor plans and construction drawings in which critical metrics (areas, apertures, wall specifications) are contained in PDF documents. Existing academic baselines extract objects from raster plans but do not deliver production-grade, DIN-aware outputs with traceability and provenance. AI-RE's SaaS platform requires a module that turns each plan page into a machine-readable digital twin that downstream systems and an assistant can query with auditability.

### Core Focus: Spatial Recognition over Text Recognition

The primary challenge and value of this project lies in geometric and spatial recognition (room boundaries, coordinates, areas) rather than pure text extraction. While text recognition (OCR, labels) supports the solution, the critical technical work is:

1. **Room polygonization with accurate coordinates:** detecting closed boundaries and computing areas.
2. **Wall centerline and thickness extraction:** geometric analysis of the plan structure.
3. **Topology and spatial relationships:** connecting rooms, walls and openings consistently.
4. **Coordinate system and scale resolution:** mapping pixels to real-world metres.

Text extraction (OCR, labels) is a supporting capability rather than the primary focus. The system must be able to perceive and understand the geometric structure of plans, which is significantly more complex than reading text.

### Evaluation Strategy: Question-Answering Test Battery

The system will be evaluated on its ability to correctly answer questions, not only on raw extraction metrics.

#### Test Battery Structure

- At least 100 test questions covering room measurements, counts and geometry.
- Questions categorized by priority: CRITICAL (rooms and measurements), HIGH (walls and doors), LOWER (materials and compliance).
- Each question has a ground-truth answer derived from a manually annotated gold set.

#### Example test questions include:

1. "How big is apartment Nr. 3?" → Expected: "85.6 m²" (source: page 5, polygon ID).
2. "How many 2-room apartments are in building A?" → Expected: "8 apartments" (with list of IDs).
3. "How wide are the external walls?" → Expected: "24 cm" (source: wall measurements).
4. "What is the total living space in building B?" → Expected: "1,245 m²" (derived from room polygons).

#### Success Criteria

- At least 80% of CRITICAL questions answered correctly.
- At least 95% of answers include provenance (page / bounding box / polygon ID).
- A live demonstration successfully answering questions on previously unseen plans.

#### Evaluation Deliverable

An automated test suite that scores the system and generates a report summarizing:

- Percentage of questions answered correctly (overall and by category).
- Average error for numerical answers (e.g. area measurements).
- Provenance coverage (percentage of answers with source links).
- Representative examples of correct and incorrect answers.

---

## 3. Goal

From German architectural plans (PDF/images), the system shall output a traceable, machine-readable digital twin per page.

### Primary Focus

- Rooms (polygons, area, coordinates) as the most critical task.
- Walls (centerlines, thickness, type), doors and windows (geometry, attributes).
- Dimensions, legends and materials, floors/levels and global metrics.
- Provenance: page index and coordinates for every numeric or textual fact; versioned model metadata.
- APIs: extraction endpoint and query interface for an assistant.

### Success KPIs (Semester Target)

#### Critical (Must Achieve) — Evaluated via Question-Answering Test Set

**1. Room identification with measurements (most important)**

- Room detection recall: at least 95%.
- Area accuracy: median error at most 2.5%.
- Coordinate extraction: all rooms have valid polygon coordinates.
- **Test:** the system correctly answers "How big is apartment/room X?" for at least 90% of test questions.

**2. Apertures (doors and windows) with counts**

- Detection recall: at least 92% for doors and windows.
- **Test:** the system correctly answers "How many doors/windows are there?" for at least 85% of test questions.

**3. Wall geometry with measurements**

- Thickness accuracy: mean absolute error (MAE) at most 10 mm for wall thickness.
- Stable wall centerlines for topological reasoning.
- **Test:** the system correctly answers "How wide are the walls?" for at least 80% of test questions.

#### High (Should Achieve)

- Locale/DIN compliance: decimal commas, metric units, level codes (EG/OG/DG).
- Traceability: at least 95% of returned facts are clickable back to a page/span/polygon ID.
- **Test:** the system provides a source page / bounding box for at least 95% of answers.

#### Evaluation Method

- Gold set: 200 annotated pages with ground-truth measurements.
- Test battery: 100+ questions covering room sizes, counts and measurements.
- Success metric: at least 80% of CRITICAL questions answered correctly with provenance.

---

## 4. Scope and Deliverables

Core deliverables shall be implemented (open-source friendly where feasible, otherwise in a company-internal repository).

### Critical Deliverables (Must Deliver)

#### 1. Extraction pipeline (on-premises, no third-party APIs) — focused on room recognition with measurements

- PDF/vector pre-parsing (Docling) to extract paths, text spans and layers; raster fallback at 300–400 dpi.
- Dual geometry backbone: (a) wireframe (HAWP/L-CNN with vector fusion) and (b) semantics (YOLOv8/Detectron2 masks and symbols).
- Topology and units engine: room polygonization with coordinates and area (primary focus), scale resolution, area and length computation, wall ribbons (centerline and thickness), opening clipping.
- Post-processing and quality assurance: polygon repair, duplicate removal, confidence calibration.

#### 2. Question-answering interface (testable, demonstration-ready system)

- Natural language question parser.
- Query engine that searches the extracted data.
- Answer generator with provenance links.
- Demonstration UI: interactive interface showing questions, answers and clickable sources.

#### 3. Versioned JSON/GeoJSON schema with page anchors and model provenance

- All room measurements include source coordinates.
- Every fact traceable to page / bounding box / polygon ID.

#### 4. /extract REST API and /query API for assistant integration

- `POST /extract`: process PDF, return structured data.
- `POST /query`: natural language question to answer with provenance.

#### 5. Evaluation suite with test battery

- 200-page gold set (DIN-German) with ground-truth measurements.
- 100+ test questions covering CRITICAL categories.
- Automated scoring: percentage of questions answered correctly.
- Regression tests for KPI tracking.

#### 6. Engineering report (10–15 pages)

Design, data, metrics, ablations, limitations and next steps:
- Section 1: question-answering results (test battery scores).
- Section 2: room detection and measurement accuracy.
- Section 3: technical approach (computer vision models, topology engine).
- Section 4: demonstration screenshots showing example question answering.

### High-Priority Deliverables (If Time Permits)

- OCR (PaddleOCR German) and LayoutLMv3 linker for labels, dimensions and legends bound to geometry.
- Alignment of door and window swing direction.

### Lower-Priority Deliverables (Optional)

- Full material taxonomy and abbreviations parsing.
- Advanced legend parsing.

### Out-of-Scope in this Semester (Optional Stretch Goals)

- Compact vision-language model head.
- 3D lifting.
- Comprehensive material parsing.

---

## 5. Methods and Reading (Starter Pack)

Relevant literature and tools include:

- **Floor-plan parsing:** CubiCasa5K; Deep Floor Plan Recognition (ICCV 2019); PolyRoom (2024); raster-to-graph methods.
- **Wireframe and vectorization:** HAWP, L-CNN.
- **Document AI:** LayoutLMv3; Docling (PDF); PaddleOCR/PP-Structure (German).
- **Toolkits:** Detectron2, YOLOv8.

Detailed links will be provided in an internal document.

---

## 6. Data and Annotation Plan

- **Weak-label bootstrapping** from vector PDFs (walls, doors, windows, dimension strings, leader lines).
- **Active learning for scans:** tiles with high uncertainty are targeted for annotation (doors, windows, wall thickness, room closure corrections).
- **Gold set:** 200 fully verified pages; mixed vendors and styles; tracked via DVC or similar data versioning.

---

## 7. Architecture (High Level)

```
PDF -> Docling (vectors+text)
    |-> Semantic heads (YOLOv8/Detectron2)
    |-> Wireframe (HAWP/L-CNN) + vector fusion
    '-> OCR (PaddleOCR-DE) -> LayoutLMv3 linker

[Topology & Units Engine]
- Scale resolution (1:100 / explicit dimensions)
- Wall ribbons (centerline + thickness + type)
- Room polygonization and area computation
- Openings clipped to walls
- Legend / material parsing (e.g., "KS 24")

JSON/GeoJSON (+ provenance) -> /extract API -> Assistant queries
```

---

## 8. Milestones and Timeline (14 Weeks)

The timeline is adjusted to allocate two weeks for final report and presentation preparation (weeks 13–14), following the professor's recommendation. Weeks may shift with the TUM calendar; milestones are acceptance-tested against KPIs where applicable.

| Week | Milestone and Acceptance Criteria | Priority |
|------|-----------------------------------|----------|
| 1 | Kick-off and setup: repository access, dataset inventory; baseline Docling and PaddleOCR demonstration on 10 pages. | Critical |
| 2 | PDF intake and raster fallback: vector/text extraction; at least 95% page load success on a batch of 100 documents. | Critical |
| 3 | Semantic baseline: YOLOv8 symbols (doors, windows, stairs) trained on weak labels; mAP50 ≥ 0.75 on development set. | Critical |
| 4 | Wall/void masks: Detectron2 (Mask R-CNN/Mask2Former) baseline; qualitative visual QA on 30 pages. | Critical |
| 5 | Wireframe: HAWP/L-CNN inference with vector fusion; stable line deduplication and snapping; junction recall qualitatively acceptable. | Critical |
| 6 | Topology engine v1 (CRITICAL): room polygonization with coordinates and area from edges/masks; initial area calculation (median error ≤ 5% on development set); tested on 10 sample plans. | Critical |
| 7 | Linker v1 (if time): LayoutLMv3 fine-tuning; correct binding of at least 85% room labels and dimensions on development set. | High |
| 8 | Walls v2: centerline and thickness estimation; thickness MAE ≤ 15 mm on an annotated subset; test door and window counts on sample plans. | Critical |
| 9 | Openings v2: door and window clipping to walls; at least 90% recall and correct wall alignment on development set. | Critical |
| 10 | Localisation and standards: level codes (EG/OG/DG), decimal commas, unit parsing; legend abbreviations optional. | High |
| 11 | Question parser and query engine MVP (CRITICAL): parse questions and query extracted data; tested on 20 questions across 5 plans; 200-page gold set frozen; evaluation harness in place. | Critical |
| 12 | Question-battery evaluation (CRITICAL): automated test suite with 100+ questions; at least 80% of CRITICAL questions answered correctly; live question-answering demonstration on unseen plans; KPI report. | Critical |
| 13 | Report and presentation preparation (week 1): draft report with question-answering results; prepare demonstration UI; initial presentation outline. | Critical |
| 14 | Final delivery (week 2): complete report (10–15 pages) with test battery results; final presentation with live demonstration; code handover; roadmap for next steps. | Critical |

### Testing milestones are integrated throughout:

- **Week 6:** manual testing of room extraction on 10 plans.
- **Week 8:** manual testing of door/window counts.
- **Week 11:** 20 test questions on 5 plans (MVP validation).
- **Week 12:** 100+ questions on the 200-page gold set (full evaluation).

---

## 9. Roles and Responsibilities (Four Students)

With four students on the team, roles and responsibilities will be finalised during the kick-off meeting. Initial role suggestions:

| Role | Responsibilities |
|------|------------------|
| **Computer Vision Lead** | Detectors/segmenters, wall/void masks, YOLOv8/Detectron2 |
| **Geometry/Topology Lead** | Wireframe fusion, room polygonization (critical), metrology, coordinates and area calculation |
| **Document AI Lead** | OCR, LayoutLMv3 linker, legend parsing (lower priority) |
| **MLOps/Backend** | Data curation (DVC), evaluation dashboard, /extract API, CI tests, evaluation harness |

### Team and Roles (Current Proposal)

| Name | Role |
|------|------|
| Feres Ben Frej | Document AI Lead |
| Oussema Belkhiria | MLOps/Backend |
| Oussama Jeddou | Computer Vision Lead |
| Ayoub Albouchi | Geometry/Topology Lead |

Students may rotate tasks; final responsibilities per person will be documented in the kick-off minutes after the team redistributes the work among all four members.

---

## 10. Infrastructure and Tools

- On-premises only (no third-party cloud APIs); Dockerised stack; Python 3.11.
- Repositories: private ai-re/*; issue and pull request workflows; CI with unit and integration tests.
- Data governance: GDPR-compliant storage; document access via secure shares; NDA where required.

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Heterogeneous plan styles | Active learning and diversity in the gold set |
| Vector-poor scans | Combining wireframe and mask fusion; manual seed annotations where necessary |
| Locale parsing edge cases | Combination of regular expressions, learned linkers and unit tests for critical strings |
| Time constraints | Milestone gates with acceptance criteria; de-scoping of lower-priority stretch goals (e.g. VLM, 3D) |

---

## 12. Ethics, Legal and Intellectual Property

Documents may contain personal data (names, addresses). Processing is conducted for research and engineering purposes under NDA; outputs exclude personal data where it is not required.

---

## 13. Deliverables Checklist

### Technical Deliverables

- [ ] Working extraction pipeline (Docker) with trained models
- [ ] `/extract` API endpoint (process PDF to JSON)
- [ ] `/query` API endpoint (question to answer with provenance)
- [ ] Demonstration UI showing live question answering on sample plans
- [ ] 200-page gold set with ground-truth measurements
- [ ] At least 100 test questions with expected answers

### Evaluation and Results

- [ ] Automated test battery showing the percentage of questions answered correctly
- [ ] Evaluation report with KPIs:
  - Room detection recall (target: ≥ 95%)
  - Area measurement accuracy (target: ≤ 2.5% error)
  - Question-answering accuracy (target: ≥ 80% of CRITICAL questions)
  - Provenance coverage (target: ≥ 95% facts traceable)
- [ ] Confusion matrices and error analysis where applicable

### Documentation

- [ ] Engineering report (10–15 pages) with:
  - Question-answering results (test battery scores and examples)
  - Room and measurement accuracy (detection metrics, area errors)
  - Technical approach (models, pipeline, topology)
  - Demonstration screenshots (question answering with provenance visualisation)
- [ ] Demonstration video (up to 8 minutes) showing:
  - Upload of a plan
  - System extraction of rooms and measurements
  - Question answering with answers and clickable sources
- [ ] README with setup instructions, API usage and example question formats

---

## 14. Question Set the System Must Answer

Questions are prioritised based on project focus. Critical questions concern room and space recognition; high-priority questions cover complementary configuration and geometry aspects; lower-priority questions are optional and may be omitted if time-constrained.

### General Questions (Metadata Extraction)

| Question | Source | JSON Field |
|----------|--------|------------|
| Who is the architect of the planning? | Title block / legend text (Docling text spans and OCR) | `meta.project.architect { name, company, bbox, page }` |
| What was the last change or modification in the planning? | Revision table, stamps | `meta.revisions[]` and `meta.revision_current` |
| What is the exact planning number? | Title block | `meta.project.planning_number { value, page, bbox }` |
| When was the last change or modification of the planning? | Revision table | `meta.revision_current.date` |
| Name all changes or modifications mentioned in the planning. | Revision table rows | `meta.revisions[]` |
| What is the date of the planning? | Title block | `meta.project.date` |

### Configuration Questions (Counts and Overviews) — CRITICAL

| Question | Source | JSON Field |
|----------|--------|------------|
| How many apartments are planned in building Nr. xy? | Apartment polygons and labels (LayoutLMv3 binding) | `apartments[] { building_id }` |
| How many stairs are planned in the building? | Symbol detection (stairs) and room labels | `installations.stairs[]` |
| How many apartments with two rooms are planned? | Apartment room counts via room-to-apartment mapping | `apartments[].stats.room_count` |
| Overview of all apartments (size and number of rooms). | Aggregation of room polygons per apartment | `apartments[] { id, area_m2, room_count, rooms[], page_refs[] }` |
| How many staircases are in building Nr. xy? | Grouped by building | `installations.staircases[] { building_id }` |
| How many lifts are in building Nr. xy? | Lift symbol detection ("Aufzug") | `installations.lifts[] { building_id }` |

### Dimension and Calculation Questions — CRITICAL

| Question | Source | JSON Field |
|----------|--------|------------|
| Total living space in building Nr. xy. | Sum of apartment areas per building | `metrics.buildings[].living_area_total_m2` |
| Living space in apartment Nr. xy. | Sum of polygons tagged as living space | `apartments[].area_living_m2` |
| Width of external walls (e.g. 22 cm). | Wall centerline and thickness; cross-check with legend | `walls[] { is_external, thickness_mm }` |
| Total space of all windows in the project. | Window geometry aggregated | `metrics.windows.total_glazed_area_m2` |
| Outer diameter or footprint perimeter of the building. | Building footprint polygon | `buildings[].footprint { perimeter_m, area_m2, bbox, page }` |
| Number of doors in the building by type. | Door detections with type classification and room adjacency | `doors[] { type, connected_rooms[], page, bbox }` |
| Surface of all external walls. | External wall lengths and storey height | `metrics.walls.external_surface_area_m2` |
| Width of corridors in staircases. | Corridor polygons inside staircases | `circulation.staircase[].corridor_widths { min_m, avg_m }` |
| Width of walls between different flats. | Walls with flag `between_apartments = true` | `walls[] { between_apartments, thickness_mm }` |
| Width of internal walls inside flats. | Internal walls per apartment | `walls[] { apartment_id, thickness_mm }` |
| Dimensions of the front door. | Door symbol and label | `doors[] { type = "entrance", width_m, height_m }` |
| Ceiling height inside flats. | Section/elevation notes or standard floor height | `levels[] { ceiling_height_m, page, bbox }` |
| Floor space (Grundfläche) of the building in total. | Building footprint area | `metrics.buildings[].gross_footprint_area_m2` |

Each answer should be accompanied by provenance: `{ page, bbox, model_version, element_id }` and a confidence score, as well as calculation inputs for derived quantities.

---

## 15. JSON Schema Additions

New or expanded fields include:

- `meta.revisions[]`, `meta.project.*` (architect, planning number, dates)
- `apartments[].finishes.floor`, `apartments[].area_living_m2`, `apartments[].stats.room_count`
- `envelope.insulation { product, thickness_mm }`
- `compliance.fire[]`, `compliance.acoustic[]`, `compliance.egress.paths[]`
- `metrics.*` such as `windows.total_glazed_area_m2`, `walls.external_surface_area_m2`, `buildings.gross_footprint_area_m2`

---

## 16. Milestone Add-Ons (Answerability Focus)

Critical milestones to ensure answerability:

- **Week 6:** room polygonization with coordinates and area (primary focus). Output: system can extract room polygons and calculate areas; tested on 10 sample plans.
- **Weeks 8–9:** door/window typing, wall thickness and apartment room-count audit. Output: system detects and counts doors/windows and measures walls; verified on sample plans.
- **Week 11:** question parser and query engine MVP. Output: system can parse questions such as "How big is room X?" and query the extracted data; tested on 20 questions across 5 plans.
- **Week 12:** full question-battery evaluation. Output: automated test suite with at least 100 questions on the 200-page gold set; target: at least 80% of CRITICAL questions answered correctly with provenance; live demonstration on unseen plans.

---

## 17. Project Management and Reporting

### Progress Tracking

- Weekly check-ins with the supervisor.
- Internal tracking of progress against milestones and KPIs.
- If significant changes occur, an updated project description will be submitted with the final report.

---

## License

This project is developed as part of an IDP at TUM in collaboration with AI-RE UG.

---

*Last updated: Winter Semester 2025/26*
