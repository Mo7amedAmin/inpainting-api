from datasets import load_dataset
from transformers import pipeline
import torch
from PIL import Image
import json
import os


dataset = load_dataset('hammer888/interior_style_dataset')
ds = dataset["train"]
ds = ds.remove_columns(['text'])

# Load the image-to-text pipeline with the model
pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf", device_map="auto")

# Define the prompt for the image captioning model
prompt = "USER: <image>\nDescribe the interior image. Be detailed, describe a style, a color, and furniture fabric. Use only one but detailed sentence. It must begin with room type description, then always describe the general style and after that describe all furniture items and their arrangement in the room and color. \n ASSISTANT:"

def generate_caption(image):
    """Generate caption for the given image."""
    outputs = pipe(image, text=prompt, generate_kwargs={"max_new_tokens": 100, "temperature": 0.4, "do_sample": True})
    text = outputs[0]["generated_text"]
    return text[text.find('ASSISTANT:') + len('ASSISTANT:') + 1:]


captions = []

# Iterate over the dataset
for i, example in enumerate(ds):
    image = example["image"] 
    caption = generate_caption(image)

    image_filename = f"image_{i}.jpg"
    image_path = os.path.join("/kaggle/working/images", image_filename)
    os.makedirs("/kaggle/working/images", exist_ok=True)
    image.save(image_path)
    
    captions.append({"image": image_path, "caption": caption})

# Save the generated captions to a JSON file
with open('/kaggle/working/captions.json', 'w') as f:
    json.dump(captions, f, indent=4)