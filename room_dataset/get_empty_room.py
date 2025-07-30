from segmentation_colors import ade_palette, map_colors_rgb
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image, ImageFilter
import numpy as np
from typing import Union
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler, StableDiffusionInpaintPipeline
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
from glob import glob
import os
from tqdm import tqdm
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import shutil
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from datasets import load_dataset



# Load Mask2Former
processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = model.cuda()

pipe = AutoPipelineForInpainting.from_pretrained('lykon/absolute-reality-1.6525-inpainting', torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

depth_image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf", torch_dtype=torch.float16)
depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf", torch_dtype=torch.float16)
depth_model = depth_model.cuda()


def resize_image(input_image, resolution, interpolation=None):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    if interpolation is None:
        interpolation = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    img = cv2.resize(input_image, (W, H), interpolation=interpolation)
    return img

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
    

def get_segmentation_of_room(image: Image):
    # Semantic Segmentation
    with torch.inference_mode():
        semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt", size={"height": 256, "width": 256})
        semantic_inputs = {key: value.to("cuda") for key, value in semantic_inputs.items()}
        semantic_outputs = model(**semantic_inputs)
        # pass through image_processor for postprocessing
        segmentation_maps = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])
        predicted_semantic_map = segmentation_maps[0]

    predicted_semantic_map = predicted_semantic_map.cpu()
    color_seg = np.zeros((predicted_semantic_map.shape[0], predicted_semantic_map.shape[1], 3), dtype=np.uint8)

    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[predicted_semantic_map == label, :] = color

    color_seg = color_seg.astype(np.uint8)
    seg_image = Image.fromarray(color_seg).convert('RGB')
    return color_seg, seg_image


def filter_items(colors_list: Union[list, np.ndarray], items_list: Union[list, np.ndarray], items_to_remove: Union[list, np.ndarray]):
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_remove:
            filtered_colors.append(color)
            filtered_items.append(item)
    return filtered_colors, filtered_items


def get_inpating_mask(segmentation_mask: np.ndarray):
    unique_colors = np.unique(segmentation_mask.reshape(-1, segmentation_mask.shape[2]), axis=0)
    unique_colors = [tuple(color) for color in unique_colors]
    segment_items = [map_colors_rgb(i) for i in unique_colors]

    control_items = ["windowpane;window", "wall", "floor;flooring", "ceiling", "sconce", "door;double;door", "light;light;source", "painting;picture", "stairs;steps", "escalator;moving;staircase;moving;stairway"]
    chosen_colors, segment_items = filter_items(colors_list=unique_colors, items_list=segment_items, items_to_remove=control_items)

    mask = np.zeros_like(segmentation_mask)
    for color in chosen_colors:
        color_matches = (segmentation_mask == color).all(axis=2)
        mask[color_matches] = 1

    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(25))
    return mask_image

def cleanup_room(image: Image, mask: Image):
    image = transforms.ToPILImage()(image).convert("RGB")
    inpaint_prompt = "Empty room, with only empty walls, floor, ceiling, doors, windows"
    negative_prompt = "furnitures, sofa, cough, table, plants, rug, home equipment, music equipment, shelves, books, light, lamps, window, radiator"
    image_source_for_inpaint = image.resize((512, 512))
    image_mask_for_inpaint = mask.resize((512, 512))
    generator = [torch.Generator(device="cuda").manual_seed(20)]

    image_inpainting_auto = pipe(prompt=inpaint_prompt, negative_prompt=negative_prompt, generator=generator, strength=0.8,
         image=image_source_for_inpaint, mask_image=image_mask_for_inpaint, guidance_scale=10.0,
         num_inference_steps=10).images[0]
    image_inpainting_auto = image_inpainting_auto.resize((image.size[0], image.size[1]))
    return image_inpainting_auto

def get_depth_image(image: Image) -> Image:
    image_to_depth = depth_image_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        depth_map = depth_model(**image_to_depth).predicted_depth

    width, height = image.size
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1).float(),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), 
])


# Define Custom Dataset for Image Loading
class InteriorDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform if transform else transforms.ToTensor()

    def __iter__(self):
        for example in self.dataset: 
            img = example["image"]  
            if self.transform:
                img = self.transform(img) 
            yield img
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        image_id = f"{idx}"

        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        image = self.transform(image)
        return image_id, image
    

    
if __name__ == "__main__":
    ds = load_dataset("hammer888/interior_style_dataset")
    ds = ds['train']

    interior_dataset = InteriorDataset(ds, transform=transform)

    data_loader = DataLoader(interior_dataset, batch_size=50, shuffle=False, num_workers=4, pin_memory=True)

    save_dir = "/kaggle/working/GraduationProject/"
    save_dir_clean = os.path.join(save_dir, "clean_images/")
    save_dir_segmentation = os.path.join(save_dir, "segmentation_images/")
    save_dir_depth = os.path.join(save_dir, "depth_images/")

    os.makedirs(save_dir_clean, exist_ok=True)
    os.makedirs(save_dir_segmentation, exist_ok=True)
    os.makedirs(save_dir_depth, exist_ok=True)

    batch_num = 0
    for batch in tqdm(data_loader):
        image_ids, images = batch 
        batch_num += 1
        for image_id, image in zip(image_ids, images):
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(image)
            color_map, segmentation_map = get_segmentation_of_room(pil_image)
            inpaiting_mask = get_inpating_mask(color_map)
            clean_room = cleanup_room(image, inpaiting_mask)
            color_map_clean, segmentation_map_clean_room = get_segmentation_of_room(clean_room)
            depth_clean_room = get_depth_image(clean_room)

            # Save results
            image_id = int(image_id)
            len_id = len(str(image_id))
            temp = ''
            if len_id < 4:
                temp = '0' * (4-len_id)
                
            clean_room.save(os.path.join(save_dir_clean, "image_" + temp + str(image_id) + "_clean.png"))
            segmentation_map_clean_room.save(os.path.join(save_dir_segmentation, "image_" + temp + str(image_id) + "_segmentation.png"))
            depth_clean_room.save(os.path.join(save_dir_depth,"image_" + temp + str(image_id) + "_depth.png"))
            
        
        shutil.make_archive("/kaggle/working/GraduationProject", 'zip', save_dir)
        print(f"{batch_num*50} images added")