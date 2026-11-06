import json
import os

ndjson_path = r'c:\D-Drive\Projects\DIP\acnedata3.ndjson'
file_size = os.path.getsize(ndjson_path)
print(f'.ndjson file size: {file_size} bytes ({file_size/1024/1024:.1f} MB)')

with open(ndjson_path, 'r') as f:
    lines = f.readlines()
    print(f'Total lines: {len(lines)}')
    
    # Process first 3 lines
    for i in range(min(3, len(lines))):
        line = lines[i].strip()
        if not line:
            print(f'Line {i+1}: EMPTY')
            continue
        try:
            obj = json.loads(line)
            obj_type = obj.get('type', 'unknown')
            print(f'Line {i+1} type: {obj_type}')
            if obj_type == 'image':
                ann = obj.get('annotations', {})
                boxes = ann.get('boxes', []) if isinstance(ann, dict) else []
                print(f'  image: file info present, boxes: {len(boxes)}')
                if boxes:
                    print(f'  Box 0: {boxes[0]}')
            elif obj_type == 'dataset':
                print(f'  dataset info: class_names={obj.get("class_names", "N/A")}')
        except json.JSONDecodeError as e:
            print(f'Line {i+1}: JSON parse error: {e}')
PYEOF