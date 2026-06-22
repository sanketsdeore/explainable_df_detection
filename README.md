# Explainable Deepfake Detection App (ECS + Gemini)

A premium, state-of-the-art **Deepfake Detection Web Application** built with Streamlit and PyTorch. The application utilizes a Vision Transformer (ViT) to perform face-level deepfake classification and integrates explainability features via **Grad-CAM** visual attention maps and **Gemini AI** natural language explanations.

---

## 🌟 Key Features

- **Dual Mode Input**: Supports high-fidelity deepfake analysis for both **videos** (analyzed frame-by-frame) and static **images**.
- **State-of-the-Art Detection**: Utilizes the pre-trained `dima806/deepfake_vs_real_image_detection` Vision Transformer (ViT) with ~96% classification accuracy.
- **Explainability Consistency Score (ECS)**: A metric calculated using structural similarity (SSIM) and cosine similarity across successive Grad-CAM heatmaps to measure how consistently the model focuses on specific facial regions.
- **AI-Powered Explanations**: Integrates with the **Gemini 1.5 Flash API** to generate automated, plain-English explanations of the model's focus areas, patterns of artifact detection, and classification confidence.
- **Hardware Acceleration**: Built-in CUDA support for real-time inference using Nvidia GPU.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12 or 3.13
- CUDA Toolkit (optional, for GPU acceleration)

### 2. Setup & Installation
Clone the repository:
```bash
git clone https://github.com/sanketsdeore/explainable_df_detection.git
cd explainable_df_detection
```

Set up virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Install requirements:
```bash
pip install streamlit torch torchvision transformers grad-cam scikit-image opencv-python google-generativeai
```
*(For GPU support on Windows/Linux, install CUDA-enabled PyTorch directly from the PyTorch index).*

### 3. Run the App
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

---

## ⚙️ How it Works

1. **Face Detection**: Uses Multi-task Cascaded Convolutional Networks (MTCNN) via `facenet-pytorch` to extract and crop faces from input files.
2. **Deepfake Inference**: Processes cropped face frames through the ViT classifier.
3. **Visual Explanation**: Computes Grad-CAM heatmaps showing which parts of the face (eyes, mouth, skin blending) triggered the classification decision.
4. **Natural Language Explanation**: Feeds the heatmap and verdict to Google's Gemini LLM to write a concise diagnostic description for the user.

---

## 🛠️ Technology Stack
- **Frontend/UI**: Streamlit
- **Deep Learning**: PyTorch, torchvision
- **Face Detection**: facenet-pytorch (MTCNN)
- **Model Explainability**: PyTorch Grad-CAM
- **Generative AI**: Google Gemini API
- **Computer Vision**: OpenCV (cv2), scikit-image
