import json
path=r"./civic_model_final/config.json"
try:
    with open(path, 'r') as f:
        config = json.load(f)
    print("JSON is valid.")
except Exception as e:
    print(f"Invalid JSON: {e}")

import os
print("File Exist?",os.path.exists("./civic_model_final/config.json"))
print("File size:",os.path.getsize("./civic_model_final/config.json"))