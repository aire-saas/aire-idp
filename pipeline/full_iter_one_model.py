import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os
import argparse
from process import PDFProcessor, PageContent
from planSplitter import PlanSplitter
from pathlib import Path
import logging
import detect_rooms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoomDetectionPipeline:
    """Pipeline for detecting rooms and architectural elements in floor plans."""
    
    def __init__(self, 
                 model_path='best.pt',
                 stairs_model_path='best_stairs.pt',
                 tile_size=1024,
                 conf=0.25,
                 device='cpu'):
        """
        Initialize the detection pipeline.
        
        Args:
            model_path: Path to YOLO model for walls, doors, windows
            stairs_model_path: Path to YOLO model for stairs/elevators
            tile_size: Size of tiles for processing large images
            conf: Confidence threshold for detections
            device: Device to use ('cpu' or 'cuda')
        """
        self.model = YOLO(model_path)
        self.stairs_model = YOLO(stairs_model_path)
        self.tile_size = tile_size
        self.conf = conf
        self.device = device
        
        # Font for annotations
        try:
            self.font = ImageFont.truetype('arial.ttf', size=16)
        except Exception:
            self.font = ImageFont.load_default()
    
    def process_pages(self, pages_content: list) -> list:
        """
        Process a list of PageContent objects to detect rooms and elements.
        Updates each PageContent with predictions and room data.
        
        Args:
            pages_content: List of PageContent objects
            
        Returns:
            List of updated PageContent objects
        """
        for page in pages_content:
            if page.plan is None:
                logger.warning(f"Page {page.page_number} has no plan image, skipping")
                continue
            
            logger.info(f"Processing page {page.page_number} for room detection")
            
            # Run detection on the plan
            detections, annotated_img = self._detect_elements(page.plan)
            
            # Store predicted plan with annotations
            page.predicted_plan = annotated_img
            
            # Detect rooms based on detections
            drawn_rooms, rooms_data = detect_rooms.detect_rooms_from_image(
                page.predicted_plan, 
                detections
            )
            page.rooms = rooms_data
            page.predicted_plan = Image.fromarray(drawn_rooms)
            
            #logger.info(f"Page {page.page_number}: Detected {len(detections)} elements, {len(rooms_data.get('rooms', []))} rooms")
        
        return pages_content
    
    def _detect_elements(self, image: Image.Image) -> tuple:
        """
        Detect walls, doors, windows, stairs in the image.
        
        Args:
            image: PIL Image of floor plan
            
        Returns:
            Tuple of (detections_list, annotated_image)
        """
        #image = Image.open("pipeline_output\\page_001\\plan.png")
        img = image.convert('RGB')
        #img = image
        W, H = img.size
        print("Image size:", W, H)
        
        detections = []
        tiles = []
        coords = []
        
        # Create tiles for processing
        for y in range(0, H, self.tile_size):
            for x in range(0, W, self.tile_size):
                x2 = min(x + self.tile_size, W)
                y2 = min(y + self.tile_size, H)
                tile = img.crop((x, y, x2, y2))
                if tile.size != (self.tile_size, self.tile_size):
                    pad = Image.new('RGB', (self.tile_size, self.tile_size), (0, 0, 0))
                    pad.paste(tile, (0, 0))
                    tile = pad
                tiles.append(np.array(tile))
                coords.append((x, y, x2 - x, y2 - y))
        
        print("Number of tiles:", len(tiles))
        # Process each tile with main model
        for idx, tile_arr in enumerate(tiles):
            x_off, y_off, w_orig, h_orig = coords[idx]
            try:
                results = self.model.predict(
                    source=tile_arr, 
                    imgsz=self.tile_size, 
                    conf=self.conf, 
                    device=self.device, 
                    save=False
                )
                print("results:", len(results))
            except Exception as e:
                logger.warning(f"Error processing tile {idx}: {e}")
                results = []
            
            for r in results:
                try:
                    xyxy = r.boxes.xyxy.cpu().numpy().tolist() if hasattr(r.boxes, 'xyxy') else []
                    confs = r.boxes.conf.cpu().numpy().tolist() if hasattr(r.boxes, 'conf') else []
                    cls = r.boxes.cls.cpu().numpy().astype(int).tolist() if hasattr(r.boxes, 'cls') else []
                except Exception:
                    xyxy, confs, cls = [], [], []
                print("xyxy:", len(xyxy))
                for b_i, box in enumerate(xyxy):
                    xmin, ymin, xmax, ymax = box
                    xmin_o = float(xmin) + x_off
                    ymin_o = float(ymin) + y_off
                    xmax_o = float(xmax) + x_off
                    ymax_o = float(ymax) + y_off
                    
                    # Clamp to image bounds
                    xmin_o = max(0.0, min(float(W), xmin_o))
                    ymin_o = max(0.0, min(float(H), ymin_o))
                    xmax_o = max(0.0, min(float(W), xmax_o))
                    ymax_o = max(0.0, min(float(H), ymax_o))
                    
                    c = confs[b_i] if b_i < len(confs) else None
                    class_id = int(cls[b_i]) if b_i < len(cls) else None
                    class_name = self.model.names[class_id] if (class_id is not None and class_id in self. model.names) else None

                    # Determine source type based on class name
                    cname_l = str(class_name).lower() if class_name else ''
                    if 'door' in cname_l:
                        source = 'door'
                    elif 'window' in cname_l:
                        source = 'window'
                    elif 'wall' in cname_l:
                        source = 'wall'
                    else:
                        source = 'other'

                    det = {
                        'source': source,
                        'xmin': float(xmin_o),
                        'ymin': float(ymin_o),
                        'xmax': float(xmax_o),
                        'ymax': float(ymax_o),
                        'confidence': float(c) if c is not None else None,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(det)
        
        # Process tiles with stairs model
        for idx, tile_arr in enumerate(tiles):
            x_off, y_off, w_orig, h_orig = coords[idx]
            
            try:
                results = self.stairs_model.predict(
                    source=tile_arr, 
                    imgsz=self.tile_size, 
                    conf=self.conf, 
                    device=self.device, 
                    save=False
                )
            except Exception as e:
                logger.warning(f"Error processing tile {idx} with stairs model: {e}")
                results = []
            
            for r in results:
                try:
                    xyxy = r.boxes.xyxy.cpu().numpy().tolist() if hasattr(r.boxes, 'xyxy') else []
                    confs = r.boxes.conf.cpu().numpy().tolist() if hasattr(r.boxes, 'conf') else []
                    cls = r.boxes.cls.cpu().numpy().astype(int).tolist() if hasattr(r.boxes, 'cls') else []
                except Exception:
                    xyxy, confs, cls = [], [], []
                
                for b_i, box in enumerate(xyxy):
                    xmin, ymin, xmax, ymax = box
                    xmin_o = float(xmin) + x_off
                    ymin_o = float(ymin) + y_off
                    xmax_o = float(xmax) + x_off
                    ymax_o = float(ymax) + y_off
                    
                    xmin_o = max(0.0, min(float(W), xmin_o))
                    ymin_o = max(0.0, min(float(H), ymin_o))
                    xmax_o = max(0.0, min(float(W), xmax_o))
                    ymax_o = max(0.0, min(float(H), ymax_o))
                        
                    c = confs[b_i] if b_i < len(confs) else None
                    class_id = int(cls[b_i]) if b_i < len(cls) else None
                    class_name = self.stairs_model.names[class_id] if (class_id is not None and class_id in self.stairs_model.names) else None

                    # Determine source type based on class name
                    cname_l = str(class_name).lower() if class_name else ''
                    if 'stair' in cname_l or 'treppe' in cname_l:
                        source = 'stairs'
                    elif 'aufzug' in cname_l or 'elevator' in cname_l or 'lift' in cname_l:
                        source = 'aufzug'
                    else:
                        source = 'stairs_other'

                    det = {
                        'source': source,
                        'xmin': float(xmin_o),
                        'ymin': float(ymin_o),
                        'xmax': float(xmax_o),
                        'ymax': float(ymax_o),
                        'confidence': float(c) if c is not None else None,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(det)
        
        # Create annotated image
        annotated_img = self._annotate_detections(img, detections)
        
        return detections, annotated_img
    
    def _annotate_detections(self, image: Image.Image, detections: list) -> Image.Image:
        
        page = Image.new('RGBA', (image.width, image.height), (255, 255, 255, 255))
        page_draw = ImageDraw.Draw(page)

        # Draw walls filled black (opaque) first
        for det in detections:
            if det['source'] == 'wall':
                xmin = int(round(det['xmin']))
                ymin = int(round(det['ymin']))
                xmax = int(round(det['xmax']))
                ymax = int(round(det['ymax']))
                page_draw.rectangle([xmin, ymin, xmax, ymax], fill=(0, 0, 0, 255))

        # Draw doors and windows on top
        for det in detections:
            if det['source'] in ('door', 'window'):
                xmin = int(round(det['xmin']))
                ymin = int(round(det['ymin']))
                xmax = int(round(det['xmax']))
                ymax = int(round(det['ymax']))
                
                if det['source'] == 'door':
                    color = (0, 0, 255, 255)  # blue
                else:
                    color = (0, 0, 0, 255)  # black for windows
                
                # outline
                page_draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
                # filled translucent
                fill_col = (color[0], color[1], color[2], 140)
                page_draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_col)

        # Save the new page as the annotated image with same resolution as original
        annotated = page.convert('RGB')
        return annotated    


def full_iter_one_model(pages_content: list,
                        model_path='best.pt',
                        stairs_model_path='best_stairs.pt',
                        tile_size=1024,
                        conf=0.25,
                        device='cpu') -> list:
    """
    Process pages with room detection pipeline.
    
    Args:
        pages_content: List of PageContent objects
        model_path: Path to main detection model
        stairs_model_path: Path to stairs detection model
        tile_size: Tile size for processing
        conf: Confidence threshold
        device: Device to use
        
    Returns:
        Updated list of PageContent objects
    """
    pipeline = RoomDetectionPipeline(
        model_path=model_path,
        stairs_model_path=stairs_model_path,
        tile_size=tile_size,
        conf=conf,
        device=device
    )
    
    return pipeline.process_pages(pages_content)

