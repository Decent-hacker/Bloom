import json
import os

ndjson_path = r'c:\D-Drive\Projects\DIP\acne-dataset-6-classesv1iyolov8.ndjson'
file_size = os.path.getsize(ndjson_path)
print(f'.ndjson file size: {file_size} bytes ({file_size/1024/1024:.1f} MB)')

# Check if there's a folder with the same name
folder_name = os.path.splitext(os.path.basename(ndjson_path))[0]
folder_path = os.path.join(os.path.dirname(ndjson_path), folder_name)
if os.path.isdir(folder_path):
    print(f'Folder exists: {folder_path}')
    # List first 5 items
    items = os.listdir(folder_path)[:5]
    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            print(f'  File: {item} ({os.path.getsize(item_path)} bytes)')
        else:
            print(f'  Dir: {item}')
else:
    print(f'No folder found at: {folder_path}')

# Check for .ndjson files in the same directory
print(f'\n.ndjson files in directory:')
for root, dirs, files in os.walk(os.path.dirname(ndjson_path)):
    for f in files:
        if f.endswith('.ndjson'):
            full_path = os.path.join(root, f)
            print(f'  {full_path} ({os.path.getsize(full_path)} bytes)')
PYEOF