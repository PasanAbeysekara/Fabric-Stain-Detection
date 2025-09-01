import numpy as np
import argparse
import time
import cv2
from pathlib import Path
import os
import tkinter.filedialog
from tkinter import *
import decimal
decimal.getcontext().rounding = decimal.ROUND_DOWN

ap = argparse.ArgumentParser()

ap.add_argument("-y", "--yolo", required=False, default="yolo_model",
                help="base path to YOLO directory")

ap.add_argument("-c", "--confidence", type=float, default=0.3,
                help="minimum probability to filter weak detections")

ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Check if required files exist
labelsPath = os.path.sep.join([args["yolo"], "classes.names"])
weightsPath = os.path.sep.join([args["yolo"], "custom.weights"])
configPath = os.path.sep.join([args["yolo"], "custom.cfg"])

print("[INFO] Checking YOLO files...")
if not os.path.exists(labelsPath):
    print(f"[ERROR] Classes file not found: {labelsPath}")
    exit(1)
if not os.path.exists(configPath):
    print(f"[ERROR] Config file not found: {configPath}")
    exit(1)
if not os.path.exists(weightsPath):
    print(f"[ERROR] Weights file not found: {weightsPath}")
    print("[INFO] Please download the weights file from:")
    print("https://www.mediafire.com/file/k7eh9v107de1z2x/custom.weights/file")
    print("[INFO] Running in DEMO mode without YOLO detection...")
    DEMO_MODE = True
else:
    DEMO_MODE = False

if not DEMO_MODE:
    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

mts = 0.0000002645833

def listToString(s):
    return ''.join(str(x) for x in s)

def demo_process_image(image_path):
    """Demo function to process image without YOLO detection"""
    print(f"[DEMO] Processing image: {os.path.basename(image_path)}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return
    
    (H, W) = image.shape[:2]
    total_area = H * W * mts
    
    print(f"[DEMO] Image dimensions: {W}x{H}")
    print(f"[DEMO] Total area: {total_area:.6f} m²")
    
    # Create demo crop functionality
    hww, www, channels = image.shape
    half = www // 2
    half2 = hww // 2
    
    top = image[:half2, :]
    bottom = image[half2:, :]
    
    # Ensure crop directory exists
    os.makedirs('crop', exist_ok=True)
    
    cv2.imwrite('crop/top.jpg', top)
    cv2.imwrite('crop/bottom.jpg', bottom)
    
    print("[DEMO] Image split into top and bottom halves")
    print("[DEMO] Saved: crop/top.jpg and crop/bottom.jpg")
    
    # Display images
    cv2.imshow('Original Image', cv2.resize(image, (640, 480)))
    cv2.imshow('Top Half', cv2.resize(top, (640, 240)))
    cv2.imshow('Bottom Half', cv2.resize(bottom, (640, 240)))
    
    print("[DEMO] Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_with_yolo(image_path):
    """Original YOLO processing function"""
    count = 0
    total_area = 0
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    total_area = H * W * mts
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []
    stain_loc_mts = []
    total_stain_area = 0
    coordinate = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                total_stain_area = total_stain_area + (width * height)

                stain_loc_mts.append([float(round(float((x * mts)),5)), float(round(float((y * mts)),5))])
                coordinate.append([x, y])
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    total_stain_area = float(round((total_stain_area * mts),2))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    if len(idxs) > 0:
        for i in idxs.flatten():
            count += 1

    print(f"[INFO] Detected {count} stains")
    print(f"[INFO] Total stain area: {total_stain_area} m²")
    
    # Process crop images
    hww, www, channels = image.shape
    half = www // 2
    half2 = hww // 2

    top = image[:half2, :]
    bottom = image[half2:, :]

    cv2.imshow('Top', top)
    cv2.imshow('Bottom', bottom)

    cv2.imwrite('crop/top.jpg', top)
    cv2.imwrite('crop/bottom.jpg', bottom)
    cv2.waitKey(0)

    basepath = Path('crop/')
    files_in_basepath = basepath.iterdir()
    for item in files_in_basepath:
        print(item)
        image = cv2.imread(str(item))
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                args["threshold"])

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        image = cv2.resize(image, (640, 480))
        cv2.imshow("Image", image)
        cv2.waitKey(0)

def browse():
    global path
    path = tkinter.filedialog.askopenfilename(
        title="Select fabric image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    
    if len(path) > 0:
        print(f"[INFO] Selected image: {path}")
        
        if DEMO_MODE:
            demo_process_image(path)
        else:
            process_with_yolo(path)

def demo_with_sample():
    """Demo function using sample images"""
    sample_images = []
    
    # Collect sample images
    for img_dir in ['images/defect_free', 'images/stain']:
        if os.path.exists(img_dir):
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_images.append(os.path.join(img_dir, img_file))
    
    if os.path.exists('images/23.jpg'):
        sample_images.append('images/23.jpg')
    
    if sample_images:
        print(f"[DEMO] Found {len(sample_images)} sample images")
        print("[DEMO] Processing first sample image...")
        demo_process_image(sample_images[0])
    else:
        print("[DEMO] No sample images found")

# Create GUI
root = Tk()
root.geometry("700x400")
root.title("Fabric Stain Detection System")

# Main title
title_text = "Cloth Defect Detector using YOLO"
if DEMO_MODE:
    title_text += " (DEMO MODE)"

label = Label(root, text=title_text, font=("Courier", 16, "bold"))
label.pack(pady=20)

# Status label
if DEMO_MODE:
    status_label = Label(root, text="⚠️ Running in DEMO mode - YOLO weights not found", 
                        fg="orange", font=("Arial", 12))
    status_label.pack(pady=5)
    
    info_label = Label(root, text="Demo mode shows image processing without stain detection", 
                      fg="gray", font=("Arial", 10))
    info_label.pack(pady=5)
else:
    status_label = Label(root, text="✅ YOLO model loaded successfully", 
                        fg="green", font=("Arial", 12))
    status_label.pack(pady=5)

# Buttons frame
button_frame = Frame(root)
button_frame.pack(pady=30)

# Browse button
browse_btn = Button(button_frame, text="Browse Image", command=browse, 
                   font=("Arial", 12), bg="lightblue", padx=20, pady=10)
browse_btn.pack(side=LEFT, padx=10)

# Demo button (only in demo mode)
if DEMO_MODE:
    demo_btn = Button(button_frame, text="Try Sample Image", command=demo_with_sample,
                     font=("Arial", 12), bg="lightgreen", padx=20, pady=10)
    demo_btn.pack(side=LEFT, padx=10)

# Instructions
instructions = """
Instructions:
1. Click 'Browse Image' to select a fabric image
2. The system will process and analyze the image
3. Results will be displayed in new windows
"""

if DEMO_MODE:
    instructions += "\nDEMO MODE: Only image splitting is available without YOLO weights"

instruction_label = Label(root, text=instructions, font=("Arial", 10), 
                         justify=LEFT, wraplength=600)
instruction_label.pack(pady=20)

# Add info about missing weights
if DEMO_MODE:
    weights_info = Label(root, text="To enable full functionality, download custom.weights from:\nhttps://www.mediafire.com/file/k7eh9v107de1z2x/custom.weights/file", 
                        font=("Arial", 9), fg="blue", wraplength=600)
    weights_info.pack(pady=10)

print("[INFO] Starting Fabric Stain Detection System...")
if DEMO_MODE:
    print("[INFO] Running in DEMO mode")
    print("[INFO] To enable full YOLO detection, download the weights file")
else:
    print("[INFO] YOLO model loaded successfully")

root.mainloop()
