import json

with open(r'c:\D-Drive\Projects\DIP\acne-dataset-6-classesv1iyolov8.ndjson', 'r') as f:
    lines = f.readlines()
    for i in [0, 1, 2]:
        obj = json.loads(lines[i])
        print(f'Line {i+1} type: {obj.get("type")}')
        print(f'  file: {obj.get("file", "N/A")[:80] if obj.get("file") else "N/A"}')
        ann = obj.get("annotations", {})
        boxes = ann.get("boxes", []) if isinstance(ann, dict) else "N/A"
        print(f'  boxes count: {len(boxes) if isinstance(boxes, list) else "N/A"}')
        if boxes and len(boxes) > 0:
            print(f'  Box 0: {boxes[0]}')
        print()
PYEOF