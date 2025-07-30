# 🏫️ Interior Image Generation & Styling Assistant

This project focuses on generating realistic interior room images from textual descriptions using a multimodal vision-language pipeline. It combines caption generation, dataset preparation, and model training to build a system capable of interior image synthesis conditioned on user prompts.

---

## 📌 Project Overview

Initially, the dataset consisted of **room images only** (without any annotations). We extended the dataset by generating:

- ✏️ **Textual descriptions** (captions) for each room image
- 🏜️ **Furniture-free images** (rooms without any furniture)
- 🎯 **Segmentation maps**
- 🌊 **Depth maps**

These generated assets were then used to **train a ControlNet model**, enabling it to synthesize interior images based on user prompts and structural guidance.

---

## 🧠 Model & Pipeline

We built a pipeline that takes:

- 🖼️ An **input room image**
- 📝 A **textual prompt** (detailed description of desired layout or style)

And returns:

- 🏡 A **synthesized interior image** that reflects the described style, furniture, and arrangement.

This is powered by a **ControlNet-based architecture**, trained on the enhanced dataset we prepared.

---

## 🧾 Dataset Construction

We used an unlabelled dataset of room images and extended it with:

- **Captions**: Automatically generated using a vision-language model (`llava-hf/llava-1.5-7b-hf`) with a custom prompt.
- **Empty room generation**: Furniture removed via inpainting techniques.
- **Segmentation**: Generated using semantic segmentation models.
- **Depth maps**: Estimated using monocular depth prediction models.

These outputs were saved in JSON and image format for reuse.

---


## 🚀 Deployment

The model was deployed on an **Azure Virtual Machine** for scalable, real-time inference:

- ✅ API hosted via **FastAPI**
- 🧠 ControlNet pipeline served for prompt-to-image generation
- 🔗 Accessible remotely for frontend integration

---

### 🐳 Run with Docker (Azure VM or Local)

1. **Clone the repo**:
```bash
git clone https://github.com/Mo7amedAmin/interior-styling-ai
cd interior-styling-ai
```

2. **Build the Docker image**:
```bash
docker build -t interior-ai-app .
```

3. **Run the Docker container**:
```bash
docker run -d -p 8000:8000 interior-ai-app
```

4. **Access the API**:
Open your browser or use a tool like Postman to test:
```
http://<your-vm-ip>:8000/docs
```

This starts the FastAPI server and serves the AI pipeline on your VM.

---

## 🛠️ Tools & Technologies

- **Python**
- **Hugging Face Transformers**
- **Diffusers + ControlNet**
- **FastAPI** for deployment
- **Azure VM** for hosting
- **Kaggle** for prototyping & caption generation

---

## 🙋 Authors

Developed by **Mohamed Ali** and **Ahmed Elsharkay** as part of a graduation project focused on AI-driven interior design, visualization, and generative modeling.

---


