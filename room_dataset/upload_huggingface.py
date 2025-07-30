import os
import json
from datasets import Dataset, DatasetDict, Image
import pandas as pd

# Paths
json_path = r"M:\college\GraduationProject\room_dataset\updated_Dataset.json"
base_path = r"M:\college\GraduationProject\room_dataset"  # root folder containing all images

# Load JSON entries
with open(json_path, "r") as f:
    data = json.load(f)

# Filter and resolve correct paths
clean_data = []
for item in data:
    full_path = os.path.join(base_path, item["full_room"])
    empty_path = os.path.join(base_path, item["empty_room"])
    depth_path = os.path.join(base_path, item["depth"])
    mask_path = os.path.join(base_path, item["mask"])
    
    if all(os.path.exists(p) for p in [full_path, empty_path, depth_path, mask_path]):
        clean_data.append({
            "full_room": full_path,
            "empty_room": empty_path,
            "depth": depth_path,
            "mask": mask_path,
            "caption": item["caption"]
        })

# Convert to Hugging Face dataset format
df = pd.DataFrame(clean_data)
dataset = Dataset.from_pandas(df)

# Cast columns to Image format
dataset = dataset.cast_column("full_room", Image())
dataset = dataset.cast_column("empty_room", Image())
dataset = dataset.cast_column("depth", Image())
dataset = dataset.cast_column("mask", Image())

# Push to hub (you need to be logged in via `huggingface-cli login`)
dataset.push_to_hub("MohamedAli77/interior-rooms")
