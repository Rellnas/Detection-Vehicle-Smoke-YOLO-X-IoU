import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import glob
import argparse

# Cek apakah GPU tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
smoke_model = YOLO('bestv8l.pt')
smoke_model.to(device)
vehicle_model = YOLO('yolov8l.pt')
vehicle_model.to(device)

# Kelas COCO yang dianggap sebagai kendaraan
VEHICLE_CLASS_IDS = {
    2: "car",    # Car
    5: "bus",    # Bus
    7: "truck"   # Truck
}

# Warna untuk masing-masing kelas
COLORS = {
    "car": (0, 255, 0),     # Hijau
    "bus": (255, 165, 0),   # Oranye
    "truck": (0, 165, 255), # Kuning
    "smoke": (0, 0, 255),   # Merah
    "vehicle_smoke": (255, 0, 255)  # Magenta
}

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area != 0 else 0

def calculate_dynamic_threshold(iou_list, min_thresh=0.3, max_thresh=0.7):
    if not iou_list:
        return min_thresh
    mean_iou = np.mean(iou_list)
    std_iou = np.std(iou_list)
    dynamic_thresh = mean_iou + std_iou
    return float(np.clip(dynamic_thresh, min_thresh, max_thresh))

def detect_objects_on_images(input_dir, output_dir, smoke_conf=0.5, vehicle_conf=0.5, manual_iou_thresh=None):
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(input_dir, '*.[jJ][pP][gG]')) + \
                  glob.glob(os.path.join(input_dir, '*.[pP][nN][gG]')) + \
                  glob.glob(os.path.join(input_dir, '*.[bB][mM][pP]'))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Statistik
    stats = {
        'total_images': len(image_files),
        'total_smoke': 0,
        'total_vehicles': 0,
        'total_pairs': 0,
        'total_combined_boxes': 0,
        'max_iou': 0,
        'total_iou': 0,
        'dynamic_thresholds': []
    }

    for idx, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        # Deteksi asap
        smoke_boxes = []
        smoke_results = smoke_model.predict(image, conf=smoke_conf, verbose=False)
        for result in smoke_results[0].boxes:
            box = result.xyxy[0].cpu().numpy().astype(int)
            smoke_boxes.append(box)
            stats['total_smoke'] += 1
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), COLORS["smoke"], 2)
            cv2.putText(image, f"Smoke {result.conf[0]:.2f}", 
                       (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, COLORS["smoke"], 2)

        # Deteksi kendaraan
        vehicle_boxes = []
        vehicle_results = vehicle_model.predict(image, conf=vehicle_conf, verbose=False)
        for result in vehicle_results[0].boxes:
            cls_id = int(result.cls)
            if cls_id in VEHICLE_CLASS_IDS:
                box = result.xyxy[0].cpu().numpy().astype(int)
                vehicle_type = VEHICLE_CLASS_IDS[cls_id]
                vehicle_boxes.append((box, vehicle_type))
                stats['total_vehicles'] += 1
                color = COLORS[vehicle_type]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(image, f"{vehicle_type} {result.conf[0]:.2f}", 
                           (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)

        # Proses pasangan
        iou_list = []
        combined_boxes = []
        for (v_box, v_type) in vehicle_boxes:
            for s_box in smoke_boxes:
                iou = calculate_iou(v_box, s_box)
                iou_list.append(iou)
                
                if iou > 0:
                    combined_x1 = min(v_box[0], s_box[0])
                    combined_y1 = min(v_box[1], s_box[1])
                    combined_x2 = max(v_box[2], s_box[2])
                    combined_y2 = max(v_box[3], s_box[3])
                    combined_box = (combined_x1, combined_y1, combined_x2, combined_y2)

                    is_duplicate = False
                    for existing in combined_boxes:
                        if calculate_iou(combined_box, existing) > 0.9:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        combined_boxes.append(combined_box)
                        stats['total_pairs'] += 1
                        stats['total_iou'] += iou
                        stats['max_iou'] = max(stats['max_iou'], iou)
                        
                        cv2.rectangle(image, 
                                    (combined_x1, combined_y1),
                                    (combined_x2, combined_y2),
                                    COLORS["vehicle_smoke"], 2)
                        cv2.putText(image, "Asap Kendaraan",
                                  (combined_x1, combined_y1-30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                  COLORS["vehicle_smoke"], 2)
                        cv2.putText(image, f"IoU: {iou:.2f}",
                                  (combined_x1, combined_y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                  (255,255,255), 2)

        # Threshold manual atau dinamis
        if manual_iou_thresh is not None:
            dynamic_thresh = manual_iou_thresh
        else:
            dynamic_thresh = calculate_dynamic_threshold(iou_list)
        stats['dynamic_thresholds'].append(dynamic_thresh)
        
        cv2.imwrite(output_path, image)
        print(f"[{idx+1}/{len(image_files)}] Processed: {filename} | Threshold used: {dynamic_thresh:.2f}")

    # Laporan akhir
    avg_iou = stats['total_iou'] / stats['total_pairs'] if stats['total_pairs'] > 0 else 0
    avg_thresh = np.mean(stats['dynamic_thresholds']) if stats['dynamic_thresholds'] else 0
    
    report = f"""=== Detection Report ===
Total Images Processed: {stats['total_images']}
Total Smoke Detected: {stats['total_smoke']}
Total Vehicles Detected: {stats['total_vehicles']}
Total Vehicle-Smoke Pairs: {stats['total_pairs']}
Maximum IoU: {stats['max_iou']:.2f}
Average IoU: {avg_iou:.2f}
Average Threshold Used: {avg_thresh:.2f}
"""
    print("\n" + report)
    
    with open(os.path.join(output_dir, "detection_report.txt"), "w") as f:
        f.write(report)

# Jalankan dari CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='test_set/image', help='Folder input gambar')
    parser.add_argument('--output_dir', type=str, default='output_resultssssssss', help='Folder hasil output')
    parser.add_argument('--smoke_conf', type=float, default=0.4, help='Confidence threshold untuk deteksi asap')
    parser.add_argument('--vehicle_conf', type=float, default=0.4, help='Confidence threshold untuk deteksi kendaraan')
    parser.add_argument('--iou_thresh', type=float, default=None, help='IoU Threshold manual (jika ingin override threshold dinamis)')
    args = parser.parse_args()

    detect_objects_on_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        smoke_conf=args.smoke_conf,
        vehicle_conf=args.vehicle_conf,
        manual_iou_thresh=args.iou_thresh
    )
