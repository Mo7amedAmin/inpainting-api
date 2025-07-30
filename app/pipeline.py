import torch
from PIL import Image, ImageFilter
import numpy as np
from torchvision import transforms
from typing import Union
import cv2

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

from app.segmentation_colors import ade_palette, map_colors_rgb



############# Load Models ##############
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# ControlNet models
controlnet_seg = ControlNetModel.from_pretrained(
    "BertChristiaens/controlnet-seg-room",
    torch_dtype=dtype,
    use_safetensors=False
).to(device)

controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=dtype,
).to(device)

# Inpainting model
inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=dtype,
).to(device)

# Mask2Former for segmentation
processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
mask_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic").to(device)

# Depth estimation
depth_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)






############# Our Functions #############
def resize_image(image: Image.Image, size=(512, 512)):
    return image.resize(size)

def get_segmentation_of_room(image: Image.Image): 
    with torch.inference_mode():
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt", size={"height": 256, "width": 256})
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = mask_model(**inputs)
        maps = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])
        seg_map = maps[0].cpu()

    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[seg_map == label, :] = color

    seg_image = Image.fromarray(color_seg).convert("RGB")
    return color_seg, seg_image

def filter_items(colors_list, items_list, items_to_remove):
    filtered_colors, filtered_items = [], []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_remove:
            filtered_colors.append(color)
            filtered_items.append(item)
    return filtered_colors, filtered_items

def get_inpainting_mask(seg_mask: np.ndarray):
    unique_colors = np.unique(seg_mask.reshape(-1, seg_mask.shape[2]), axis=0)
    unique_colors = [tuple(c) for c in unique_colors]
    segment_items = [map_colors_rgb(c) for c in unique_colors]

    items_to_remove = ["windowpane;window", "door;double;door", "stairs;steps", "escalator;moving;staircase;moving;stairway"]
    chosen_colors, _ = filter_items(unique_colors, segment_items, items_to_remove)

    mask = np.zeros_like(seg_mask)
    for color in chosen_colors:
        matches = (seg_mask == color).all(axis=2)
        mask[matches] = 1

    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
    return mask_img.filter(ImageFilter.MaxFilter(25))

def get_depth_image(image: Image.Image) -> Image.Image:
    inputs = depth_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = depth_model(**inputs).predicted_depth

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1).float(), size=image.size[::-1], mode="bicubic", align_corners=False
    )
    depth = (depth - depth.amin()) / (depth.amax() - depth.amin())
    depth_img = torch.cat([depth] * 3, dim=1)
    np_depth = depth_img.permute(0, 2, 3, 1).cpu().numpy()[0]
    return Image.fromarray((np_depth * 255).astype(np.uint8))





############## Main Function ###############

def process_image(image: Image.Image, prompt: str):
    image = resize_image(image)
    seg_map, _ = get_segmentation_of_room(image)
    inpaint_mask = get_inpainting_mask(seg_map)
    depth_mask = get_depth_image(image)

    control_images = [inpaint_mask, depth_mask]
    control_nets = [controlnet_seg, controlnet_depth]

    result = inpaint_model(
        prompt=prompt,
        image=image,
        mask_image=inpaint_mask,
        controlnet=control_nets,
        controlnet_conditioning_image=control_images,
        num_inference_steps=50,
        strength=0.7,
        generator=torch.manual_seed(42),
    ).images[0]

    return result
