import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

NDJSON_PATH = r'c:\D-Drive\Projects\DIP\acnedata3.ndjson'
OUTPUT_DIR = r'c:\D-Drive\Projects\DIP\acnedata3_images'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track statistics
total_lines = 0
downloaded = 0
skipped_no_url = 0
skipped_download_error = 0
skipped_already_exists = 0

print(f'Starting download from {NDJSON_PATH}')
print(f'Output directory: {OUTPUT_DIR}')
print()

with open(NDJSON_PATH, 'r') as f:
    lines = f.readlines()
    
total_lines = len(lines)
print(f'Total lines in .ndjson: {total_lines}')

for i, line in enumerate(lines, 1):
    line = line.strip()
    if not line:
        continue
    
    try:
        obj = json.loads(line)
        file_name = obj.get('file', '')
        url = obj.get('url', '')
        
        if not file_name or not url:
            skipped_no_url += 1
            continue
        
        # Determine the file extension from the URL or file name
        ext = Path(file_name).suffix.lower()
        if not ext:
            ext = '.jpg'
        
        # Create a safe filename
        stem = Path(file_name).stem.replace('/', '_').replace('\\', '_')
        safe_filename = f'{stem}{ext}'
        save_path = os.path.join(OUTPUT_DIR, safe_filename)
        
        # Check if already downloaded
        if os.path.exists(save_path):
            skipped_already_exists += 1
            continue
        
        # Download the image inside a try block
        try:
            urllib.request.urlretrieve(url, save_path)
            downloaded += 1
            
            # Progress update every 100 images
            if downloaded % 100 == 0:
                print(f'Downloaded {downloaded}/{total_lines} images ({downloaded/total_lines*100:.1f}%)')
        except Exception as e:
            skipped_download_error += 1
    except json.JSONDecodeError:
        skipped_no_url += 1
    
    # Print progress every 500 lines
    if i % 500 == 0:
        print(f'Progress: {i}/{total_lines} lines processed '
              f'(downloaded: {downloaded}, errors: {skipped_download_error + skipped_no_url})')

# Final summary
print(f'\n{"="*60}')
print(f'Download complete!')
print(f'  Downloaded:        {downloaded}')
print(f'  Already exists:    {skipped_already_exists}')
print(f'  No URL / Errors:   {skipped_no_url + skipped_download_error}')
print(f'  Total lines:       {total_lines}')
print(f'  Output dir:        {OUTPUT_DIR}')
print(f'\nNext step:')
print(f'  python ml/datasets/fetch_acnedata3.py --data-dir "{OUTPUT_DIR}"')