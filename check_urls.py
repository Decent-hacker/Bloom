import json
import os

ndjson_path = r'c:\D-Drive\Projects\DIP\acnedata3.ndjson'
with open(ndjson_path, 'r') as f:
    lines = f.readlines()

# Get the first image line to see the URL format
obj = json.loads(lines[1].strip())
file_name = obj.get('file', '')
url = obj.get('url', '')
print(f'Line 2 file: {file_name}')
print(f'Line 2 url: {url[:150] if url else "N/A"}...')
print(f'Line 2 split: {obj.get("split")}')

# Check how many have URLs
url_count = 0
for line in lines[1:]:
    try:
        obj = json.loads(line.strip())
        if obj.get('url'):
            url_count += 1
    except:
        pass
        
print(f'Lines with URLs: {url_count} out of {len(lines)-1}')