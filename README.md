# 🛋️ Interior Image Captioning & Styling Assistant

This project focuses on generating **detailed textual descriptions** for interior room images using a multimodal vision-language AI model. These descriptions serve as a foundation for downstream applications like interior design assistance, image inpainting, or virtual styling.

## 📌 Project Description

The system analyzes images of rooms and produces **one detailed sentence** per image. Each sentence follows a structured format:

1. Starts with the **room type** (e.g., living room, bedroom)  
2. Describes the **general interior style** (e.g., modern, rustic)  
3. Mentions the **colors**, **furniture types**, **fabric materials**, and **arrangement**  

This structured captioning allows for easy integration into further AI pipelines like design suggestion engines or generative models.

## 🧠 Model & Approach

The project uses a powerful vision-language model from Hugging Face capable of interpreting images and generating natural language descriptions. A custom prompt is used to ensure consistency and detail in the generated outputs.

## 🖼️ Dataset

The dataset contains images of various interior rooms, covering different styles and setups. Any existing captions or labels in the dataset are ignored, and fresh, structured descriptions are generated using the AI model.

## 💾 Output Format

The output consists of:

- A folder of saved images from the dataset  
- A JSON file containing the image paths and their corresponding generated captions  

These outputs can later be used in:

- Interior design recommender systems  
- Generative AI tools like inpainting or image editing  
- Dataset creation for segmentation or ControlNet training  

## 🚀 Deployment

The system has been successfully deployed on an **Azure Virtual Machine (VM)** for real-time usage and scalability testing.

This deployment setup enables:

- Hosting the pipeline as a **FastAPI server**  
- Remote access to the model for inference  
- Scalability for future integration into frontend tools or mobile apps  

Azure provides GPU-based resources to support image-based inference efficiently.

## ✅ Use Cases

- AI-assisted room styling  
- Image captioning for interior design apps  
- Dataset generation for vision-language models  
- Augmented Reality / Virtual Reality room simulation  
- Content generation for real estate platforms  

## 🛠️ Tools & Technologies

- Vision-Language AI models  
- Hugging Face Transformers  
- Python  
- FastAPI (for API deployment)  
- Azure VM (for production hosting)  
- Kaggle (for initial prototyping and testing)  

## 🙋 Author

Developed by **Mohamed Ali** and **Ahmed Elsharkay** as part of a graduation project focused on AI-driven interior styling and smart home visualization tools.

