import numpy as np
import argparse
import time
import cv2
from pathlib import Path
import os
import decimal
decimal.getcontext().rounding = decimal.ROUND_DOWN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
                    help="path to input image")
    ap.add_argument("-y", "--yolo", required=False, default="yolo_model",
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.3,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applying non-maxima suppression")
    ap.add_argument("--demo", action="store_true",
                    help="run demo with sample images")
    args = vars(ap.parse_args())

    # Check if required files exist
    labelsPath = os.path.sep.join([args["yolo"], "classes.names"])
    weightsPath = os.path.sep.join([args["yolo"], "custom.weights"])
    configPath = os.path.sep.join([args["yolo"], "custom.cfg"])

    print("=" * 60)
    print("FABRIC STAIN DETECTION SYSTEM")
    print("=" * 60)
    print("[INFO] Checking YOLO files...")
    
    DEMO_MODE = True
    if not os.path.exists(labelsPath):
        print(f"[ERROR] Classes file not found: {labelsPath}")
    elif not os.path.exists(configPath):
        print(f"[ERROR] Config file not found: {configPath}")
    elif not os.path.exists(weightsPath):
        print(f"[ERROR] Weights file not found: {weightsPath}")
        print("[INFO] Please download the weights file from:")
        print("https://www.mediafire.com/file/k7eh9v107de1z2x/custom.weights/file")
        print("[INFO] Running in DEMO mode without YOLO detection...")
    else:
        DEMO_MODE = False
        print("[INFO] All YOLO files found!")

    if not DEMO_MODE:
        LABELS = open(labelsPath).read().strip().split("\n")
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
        print("[INFO] Loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        print("[INFO] YOLO model loaded successfully!")

    mts = 0.0000002645833

    def process_image_demo(image_path):
        """Demo function to process image without YOLO detection"""
        print(f"\n[DEMO] Processing image: {os.path.basename(image_path)}")
        
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
        
        # Save processed image with annotations
        output_path = f"output_{os.path.basename(image_path)}"
        annotated = image.copy()
        
        # Add demo annotations
        cv2.putText(annotated, "DEMO MODE - No YOLO Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated, f"Image: {W}x{H}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw split line
        cv2.line(annotated, (0, half2), (www, half2), (0, 255, 0), 2)
        cv2.putText(annotated, "Split Line", (10, half2-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(output_path, annotated)
        print(f"[DEMO] Annotated image saved: {output_path}")

    def process_image_yolo(image_path):
        """Full YOLO processing function"""
        print(f"\n[YOLO] Processing image: {os.path.basename(image_path)}")
        
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
        print(f"[INFO] Total image area: {total_area:.6f} m²")
        
        # Save annotated image
        annotated = image.copy()
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(annotated, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
        
        output_path = f"detected_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Annotated image saved: {output_path}")

    def find_sample_images():
        """Find sample images in the project"""
        sample_images = []
        
        # Collect sample images
        for img_dir in ['images/defect_free', 'images/stain']:
            if os.path.exists(img_dir):
                for img_file in os.listdir(img_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample_images.append(os.path.join(img_dir, img_file))
        
        if os.path.exists('images/23.jpg'):
            sample_images.append('images/23.jpg')
        
        return sample_images

    # Main execution
    if args["demo"]:
        print("\n[INFO] Running demo mode with sample images...")
        sample_images = find_sample_images()
        
        if sample_images:
            print(f"[INFO] Found {len(sample_images)} sample images:")
            for i, img in enumerate(sample_images, 1):
                print(f"  {i}. {img}")
            
            for img_path in sample_images[:3]:  # Process first 3 images
                if DEMO_MODE:
                    process_image_demo(img_path)
                else:
                    process_image_yolo(img_path)
        else:
            print("[ERROR] No sample images found")
    
    elif args["image"]:
        if os.path.exists(args["image"]):
            if DEMO_MODE:
                process_image_demo(args["image"])
            else:
                process_image_yolo(args["image"])
        else:
            print(f"[ERROR] Image file not found: {args['image']}")
    
    else:
        print("\n[INFO] Usage examples:")
        print(f"  python3 {os.path.basename(__file__)} --demo")
        print(f"  python3 {os.path.basename(__file__)} -i images/defect_free/1.jpg")
        print(f"  python3 {os.path.basename(__file__)} -i images/stain/3.jpg")
        
        sample_images = find_sample_images()
        if sample_images:
            print(f"\n[INFO] Available sample images ({len(sample_images)} found):")
            for i, img in enumerate(sample_images, 1):
                print(f"  {i}. {img}")

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
