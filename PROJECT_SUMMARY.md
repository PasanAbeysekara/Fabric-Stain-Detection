# Fabric Stain Detection System - Project Summary

## Project Overview
This is an **Automatic Textile Stain Detection System** designed for manufacturing facilities, based on an improved YOLO algorithm. The system is designed to detect stains and defects in fabric images automatically, replacing manual inspection processes.

## Current Status
✅ **Successfully Explored and Executed**
- All dependencies installed and configured
- Command-line interface created and working
- Demo mode functional with sample images
- Image processing pipeline operational

⚠️ **Missing Component**: YOLO Custom Weights File
- The custom trained weights file needs to be downloaded separately
- Demo mode works without weights for testing purposes

## Project Structure
```
/workspaces/Fabric-Stain-Detection/
├── main.py                    # Original GUI-based application
├── fabric_detection_cli.py    # New command-line interface (recommended)
├── main_demo.py              # GUI demo version (requires display)
├── README.md                 # Project documentation
├── LICENSE                   # Project license
├── images/                   # Sample images
│   ├── defect_free/          # Clean fabric samples (4 images)
│   ├── stain/               # Stained fabric samples (4 images)
│   ├── 23.jpg               # Additional sample
│   └── output*.png          # Example outputs
├── yolo_model/              # YOLO model files
│   ├── classes.names        # Class labels (contains "stain")
│   ├── custom.cfg           # YOLO configuration
│   └── CustomWeights Link   # Link to download weights
└── crop/                    # Generated crop images
```

## Features Implemented

### ✅ Working Features (Demo Mode)
1. **Image Loading and Processing**
   - Supports multiple image formats (JPG, PNG, etc.)
   - Image dimension analysis
   - Area calculation in square meters

2. **Image Splitting**
   - Automatic splitting into top and bottom halves
   - Saves crop images for further analysis

3. **Command-Line Interface**
   - Easy-to-use CLI with multiple options
   - Demo mode for testing without weights
   - Batch processing of sample images

4. **Output Generation**
   - Annotated images with processing information
   - Structured output with measurements
   - Saved crop images

### 🔄 Full Features (Requires YOLO Weights)
1. **YOLO-based Stain Detection**
   - Real-time stain detection and classification
   - Confidence scoring for detections
   - Bounding box visualization

2. **Advanced Analysis**
   - Stain location mapping in meters
   - Total stain area calculation
   - Multiple detection threshold options

3. **Database Integration**
   - MySQL database logging (in original code)
   - Detection result storage

## How to Run the Project

### Prerequisites
All dependencies are already installed:
- Python 3.12
- OpenCV 4.12.0
- NumPy
- Additional system libraries for GUI support

### Running the System

#### 1. View Available Options
```bash
cd /workspaces/Fabric-Stain-Detection
python3 fabric_detection_cli.py
```

#### 2. Run Demo Mode (Recommended)
```bash
python3 fabric_detection_cli.py --demo
```

#### 3. Process Specific Image
```bash
python3 fabric_detection_cli.py -i images/stain/3.jpg
python3 fabric_detection_cli.py -i images/defect_free/1.jpg
```

#### 4. With Custom Parameters
```bash
python3 fabric_detection_cli.py -i images/stain/3.jpg -c 0.5 -t 0.4
```

### Expected Output
```
============================================================
FABRIC STAIN DETECTION SYSTEM
============================================================
[INFO] Checking YOLO files...
[ERROR] Weights file not found: yolo_model/custom.weights
[INFO] Please download the weights file from:
https://www.mediafire.com/file/k7eh9v107de1z2x/custom.weights/file
[INFO] Running in DEMO mode without YOLO detection...

[DEMO] Processing image: 3.jpg
[DEMO] Image dimensions: 1984x1488
[DEMO] Total area: 0.781101 m²
[DEMO] Image split into top and bottom halves
[DEMO] Saved: crop/top.jpg and crop/bottom.jpg
[DEMO] Annotated image saved: output_3.jpg
```

## Generated Files
- `output_*.jpg`: Annotated images with processing information
- `crop/top.jpg`: Top half of the processed image
- `crop/bottom.jpg`: Bottom half of the processed image

## To Enable Full Functionality

### Download YOLO Weights
1. Visit: https://www.mediafire.com/file/k7eh9v107de1z2x/custom.weights/file
2. Download the `custom.weights` file
3. Place it in the `yolo_model/` directory
4. Rerun the system - it will automatically detect the weights and enable full YOLO detection

### Alternative: Train Your Own Model
The system uses a custom YOLO configuration optimized for fabric stain detection. You can train your own model using the provided `custom.cfg` configuration.

## Technical Details

### Image Processing Pipeline
1. **Input**: Fabric image (any standard format)
2. **Preprocessing**: Resize, normalize for YOLO input
3. **Detection**: YOLO network inference (if weights available)
4. **Post-processing**: Non-maximum suppression, confidence filtering
5. **Analysis**: Area calculation, coordinate mapping
6. **Output**: Annotated images, crop files, detection results

### Measurements
- **Area Conversion Factor**: 0.0000002645833 (pixels to square meters)
- **Image Split**: Horizontal and vertical halving
- **Detection Confidence**: Configurable threshold (default: 0.3)
- **NMS Threshold**: Configurable (default: 0.3)

## Algorithm Information
- **Base**: YOLO (You Only Look Once) object detection
- **Framework**: Darknet/OpenCV DNN
- **Input Size**: 416x416 pixels
- **Classes**: Single class ("stain")
- **Architecture**: Custom configuration with 823 layers

## Summary
The project has been successfully explored and executed. The system is fully functional in demo mode, demonstrating:
- ✅ Image processing capabilities
- ✅ Command-line interface
- ✅ Area calculations and measurements
- ✅ Image splitting and crop generation
- ✅ Output file generation

To unlock the full stain detection capabilities, the only remaining step is downloading the YOLO weights file from the provided MediaFire link.
