# 🏫️ Interior Image Generation & Styling Assistant

This project focuses on generating realistic interior room images from textual descriptions using a multimodal vision-language pipeline. It combines caption generation, dataset preparation, and model training to build a system capable of interior image synthesis conditioned on user prompts.

---

## 🧾 Dataset Construction & Model Training Attempts

We started with an **unlabelled dataset of interior room images** and extended it using AI-generated annotations to support training and inference:

- ✏️ **Captions**: Automatically generated using a vision-language model (`llava-hf/llava-1.5-7b-hf`) with custom prompts to describe each room image.
- 🏜️ **Empty room images**: Created by removing furniture using inpainting techniques.
- 🎯 **Segmentation maps**: Generated using semantic segmentation models to detect and label room elements (walls, floor, furniture, etc.).
- 🌊 **Depth maps**: Estimated using monocular depth prediction models to capture spatial layout and depth cues.

All outputs were stored in structured folders and JSON metadata for reuse during training and deployment.

🔧 After constructing the dataset, we attempted to **fine-tune ControlNet models** (segmentation- and depth-based) using the generated data. Unfortunately, we encountered **hardware limitations** — particularly GPU memory constraints — that made full-scale training infeasible in our local/academic environment.

As a result, we used **pretrained ControlNet models**, leveraging the generated segmentation and depth maps to guide image generation during inference. The results were promising and demonstrated the viability of the pipeline in real-world applications.

---

## 🧠 Model & Pipeline

We built a pipeline that takes:

- 🖼️ An **input room image**
- 📝 A **textual prompt** (detailed description of desired layout or style)

And returns:

- 🏡 A **synthesized interior image** that reflects the described style, furniture, and arrangement.

This is powered by a **ControlNet-based architecture**, trained on the enhanced dataset we prepared.

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

Developed by **Mohamed Ali** and **Ahmed Elsharkawy** as part of a graduation project focused on AI-driven interior design, visualization, and generative modeling.

---


