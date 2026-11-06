import json

with open(r'c:\D-Drive\Projects\DIP\acne-dataset-6-classesv1iyolov8.ndjson', 'r') as f:
    header = f.readline()
    line = f.readline()
    obj = json.loads(line)
    ann = obj.get('annotations', {})
    boxes = ann.get('boxes', [])
    print(f'Number of boxes: {len(boxes)}')
    if boxes:
        first_box = boxes[0]
        print(f'First box type: {type(first_box).__name__}')
        print(f'First box: {first_box}')
        # Check a few more
        print(f'\\nBox 1: {boxes[0]}')
        print(f'Box 2: {boxes[1] if len(boxes) > 1 else "N/A"}')
        print(f'Box 3: {boxes[2] if len(boxes) > 2 else "N/A"}')
PYEOF