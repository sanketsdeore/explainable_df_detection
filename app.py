# ============================================================
# app.py — Complete Deepfake Detection App with ECS + Gemini
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
import math
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.metrics import structural_similarity as ssim
import google.generativeai as genai

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Deepfake Detection with ECS + Gemini Explanation")
st.markdown("Upload a **video or image** to detect deepfakes with visual explanation.")

# ── Sidebar — API Key input (never hardcode this!) ────────
st.sidebar.header("⚙️ Settings")
gemini_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    help="Get your key from aistudio.google.com"
)
frame_interval = st.sidebar.slider(
    "Frame sampling interval", 5, 30, 10,
    help="Analyze every Nth frame from video"
)
fake_threshold = st.sidebar.slider(
    "Fake ratio threshold", 0.1, 0.9, 0.4,
    help="If more than X% of frames are fake, video is FAKE"
)

# ── Load models (cached so they don't reload every run) ──
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Using device: `{device}`")

    # Face detector
    mtcnn = MTCNN(keep_all=False, device=device)

    # Deepfake detector — dima806 ViT (~96% accuracy)
    MODEL_NAME = "dima806/deepfake_vs_real_image_detection"
    processor  = ViTImageProcessor.from_pretrained(MODEL_NAME)
    vit_model  = ViTForImageClassification.from_pretrained(MODEL_NAME).to(device)
    vit_model.eval()

    # Confirm label mapping
    id2label = vit_model.config.id2label
    FAKE_IDX = [k for k, v in id2label.items() if 'fake' in v.lower()][0]
    REAL_IDX = [k for k, v in id2label.items() if 'real' in v.lower()][0]

    # Wrap model for Grad-CAM
    class WrappedViT(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.model = m
        def forward(self, x): return self.model(pixel_values=x).logits

    wrapped = WrappedViT(vit_model)
    wrapped.eval()

    if hasattr(vit_model.vit, "encoder"):
        target_layers = [vit_model.vit.encoder.layer[-1].layernorm_before]
    else:
        target_layers = [vit_model.vit.layers[-1].layernorm_before]

    return device, mtcnn, processor, vit_model, wrapped, target_layers, FAKE_IDX, REAL_IDX

# ── Helper functions ──────────────────────────────────────

def reshape_transform(tensor):
    result = tensor[:, 1:, :]
    B, N, C = result.shape
    H = W = int(N ** 0.5)
    return result.reshape(B, H, W, C).permute(0, 3, 1, 2)

def extract_face(frame_rgb, mtcnn):
    h, w, _ = frame_rgb.shape
    boxes, _ = mtcnn.detect(frame_rgb)
    if boxes is None:
        return None
    x1, y1, x2, y2 = map(int, boxes[0])
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * 0.3))
    y1 = max(0, int(y1 - bh * 0.3))
    x2 = min(w,  int(x2 + bw * 0.3))
    y2 = min(h,  int(y2 + bh * 0.3))
    face = frame_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (224, 224), interpolation=cv2.INTER_CUBIC)

def predict_face(face_np, processor, vit_model, device, FAKE_IDX, REAL_IDX):
    image  = Image.fromarray(face_np)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(vit_model(**inputs).logits, dim=1)[0]
    fake_conf = probs[FAKE_IDX].item()
    real_conf = probs[REAL_IDX].item()
    label = "FAKE" if fake_conf > 0.5 else "REAL"
    return label, fake_conf, real_conf, inputs["pixel_values"]

def generate_gradcam(pixel_values, face_np, wrapped, target_layers):
    rgb_float = face_np.astype(np.float32) / 255.0
    cam = GradCAM(
        model=wrapped,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )
    grayscale_cam = cam(input_tensor=pixel_values)[0]
    heatmap_vis   = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return grayscale_cam, heatmap_vis

def normalize_heatmap(hm):
    mn, mx = hm.min(), hm.max()
    if mx - mn < 1e-8:
        return hm
    return (hm - mn) / (mx - mn)

def compute_ecs(heatmaps):
    if len(heatmaps) < 2:
        return 0.0
    scores = []
    for i in range(len(heatmaps) - 1):
        h1 = normalize_heatmap(heatmaps[i])
        h2 = normalize_heatmap(heatmaps[i+1])
        cos = np.dot(h1.flatten(), h2.flatten()) / (
            np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8
        )
        struct = ssim(h1, h2, data_range=1.0)
        scores.append((cos + struct) / 2.0)
    return float(np.mean(scores))

def get_gemini_explanation(label, confidence, heatmap_pil, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        This is a deepfake detection result.
        Prediction: {label}
        Confidence: {confidence:.2%}

        The attached image is a Grad-CAM heatmap showing where the model focused.
        Red/yellow areas = high attention. Blue = low attention.

        Explain in simple language:
        1. Which facial regions drew attention and why
        2. What deepfake artifacts (if any) were likely detected
        3. Why the model is {'confident' if confidence > 0.75 else 'uncertain'} in this prediction

        Keep it concise, 3-4 sentences max.
        """
        response = model.generate_content([prompt, heatmap_pil])
        return response.text
    except Exception as e:
        return f"Gemini explanation unavailable: {str(e)}"

def process_video(video_path, models, frame_interval, fake_threshold):
    device, mtcnn, processor, vit_model, wrapped, target_layers, FAKE_IDX, REAL_IDX = models

    cap        = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_results = []
    heatmaps      = []
    heatmap_images = []

    progress = st.progress(0)
    status   = st.empty()
    count    = 0
    analyzed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = extract_face(rgb, mtcnn)

            if face is not None:
                label, fake_conf, real_conf, pv = predict_face(
                    face, processor, vit_model, device, FAKE_IDX, REAL_IDX
                )
                cam_map = None
                heatmap_img = None
                if label == "FAKE":
                    try:
                        cam_map, heatmap_img = generate_gradcam(pv, face, wrapped, target_layers)
                        heatmaps.append(cam_map)
                        heatmap_images.append(heatmap_img)
                    except:
                        pass

                frame_results.append({
                    "frame":      count,
                    "label":      label,
                    "fake_conf":  fake_conf,
                    "real_conf":  real_conf,
                    "face":       face,
                    "heatmap":    heatmap_img,
                })
                analyzed += 1
                status.text(f"Analyzed frame {count} → {label} "
                            f"(fake={fake_conf:.2%})")

        count += 1
        progress.progress(min(count / max(total_frames, 1), 1.0))

    cap.release()
    progress.empty()
    status.empty()

    if not frame_results:
        return None

    n_fake     = sum(1 for r in frame_results if r["label"] == "FAKE")
    n_total    = len(frame_results)
    fake_ratio = n_fake / n_total
    verdict    = "FAKE" if fake_ratio > fake_threshold else "REAL"
    avg_conf   = np.mean([r["fake_conf"] for r in frame_results
                          if r["label"] == "FAKE"]) if n_fake > 0 else 0.0
    ecs        = compute_ecs(heatmaps)

    return {
        "verdict":      verdict,
        "fake_ratio":   fake_ratio,
        "avg_conf":     avg_conf,
        "ecs":          ecs,
        "n_fake":       n_fake,
        "n_total":      n_total,
        "frames":       frame_results,
        "heatmaps":     heatmap_images,
    }

# ── Main UI ───────────────────────────────────────────────

# Load models once
with st.spinner("Loading models (first run takes ~30 seconds)..."):
    models = load_models()

_, mtcnn, processor, vit_model, wrapped, target_layers, FAKE_IDX, REAL_IDX = models

# File uploader
uploaded = st.file_uploader(
    "Upload a video or image",
    type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
)

if uploaded is not None:
    file_type = uploaded.name.split(".")[-1].lower()
    is_video  = file_type in ["mp4", "avi", "mov"]

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # ── VIDEO ──
    if is_video:
        st.video(tmp_path)
        if st.button("🔍 Analyze Video"):
            with st.spinner("Analyzing video..."):
                result = process_video(
                    tmp_path, models, frame_interval, fake_threshold
                )

            if result is None:
                st.error("No faces detected in video.")
            else:
                # Verdict banner
                color = "🔴" if result["verdict"] == "FAKE" else "🟢"
                st.markdown(f"## {color} Verdict: **{result['verdict']}**")

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Fake Frames",
                            f"{result['n_fake']}/{result['n_total']}")
                col2.metric("Fake Ratio",
                            f"{result['fake_ratio']:.1%}")
                col3.metric("Avg Confidence",
                            f"{result['avg_conf']:.1%}")
                col4.metric("ECS Score",
                            f"{result['ecs']:.4f}",
                            help="Explainability Consistency Score — "
                                 "higher = more stable attention across frames")

                # Reliability
                if result["avg_conf"] > 0.8 and result["ecs"] > 0.5:
                    st.success("✅ HIGH reliability — model is confident "
                               "and attention is consistent across frames")
                elif result["avg_conf"] > 0.8:
                    st.warning("⚠️ MEDIUM reliability — confident but "
                               "attention patterns are inconsistent")
                else:
                    st.info("ℹ️ LOW reliability — model is uncertain")

                # Show heatmaps
                if result["heatmaps"]:
                    st.subheader("🔥 Grad-CAM Heatmaps (Fake Frames)")
                    cols = st.columns(min(len(result["heatmaps"]), 4))
                    for i, hm in enumerate(result["heatmaps"][:4]):
                        cols[i % 4].image(hm, caption=f"Fake frame {i+1}",
                                          use_container_width=True)

                # Frame-by-frame table
                with st.expander("📊 Frame-by-frame results"):
                    for r in result["frames"]:
                        emoji = "🔴" if r["label"] == "FAKE" else "🟢"
                        st.write(f"{emoji} Frame {r['frame']:4d} | "
                                 f"{r['label']} | "
                                 f"Fake: {r['fake_conf']:.3f} | "
                                 f"Real: {r['real_conf']:.3f}")

                # Gemini explanation for most suspicious frame
                if gemini_key and result["heatmaps"]:
                    st.subheader("🤖 Gemini AI Explanation")
                    most_suspicious = max(
                        result["frames"],
                        key=lambda x: x["fake_conf"]
                    )
                    if most_suspicious["heatmap"] is not None:
                        hm_pil = Image.fromarray(most_suspicious["heatmap"])
                        with st.spinner("Generating Gemini explanation..."):
                            explanation = get_gemini_explanation(
                                most_suspicious["label"],
                                most_suspicious["fake_conf"],
                                hm_pil,
                                gemini_key
                            )
                        st.write(explanation)
                elif not gemini_key:
                    st.info("💡 Enter your Gemini API key in the sidebar "
                            "to get AI explanations")

    # ── IMAGE ──
    else:
        image = Image.open(tmp_path).convert("RGB")
        st.image(image, caption="Uploaded image", width=300)

        if st.button("🔍 Analyze Image"):
            face_np = extract_face(np.array(image), mtcnn)

            if face_np is None:
                st.error("No face detected in image.")
            else:
                label, fake_conf, real_conf, pv = predict_face(
                    face_np, processor, vit_model,
                    models[0], FAKE_IDX, REAL_IDX
                )

                # Result
                color = "🔴" if label == "FAKE" else "🟢"
                st.markdown(f"## {color} Prediction: **{label}**")

                col1, col2 = st.columns(2)
                col1.metric("Fake Confidence", f"{fake_conf:.1%}")
                col2.metric("Real Confidence", f"{real_conf:.1%}")

                # Grad-CAM
                try:
                    cam_map, heatmap_img = generate_gradcam(
                        pv, face_np, wrapped, target_layers
                    )
                    c1, c2 = st.columns(2)
                    c1.image(face_np, caption="Detected Face",
                             use_container_width=True)
                    c2.image(heatmap_img, caption="Grad-CAM Heatmap",
                             use_container_width=True)

                    # Gemini
                    if gemini_key:
                        st.subheader("🤖 Gemini AI Explanation")
                        hm_pil = Image.fromarray(heatmap_img)
                        with st.spinner("Generating explanation..."):
                            explanation = get_gemini_explanation(
                                label, fake_conf, hm_pil, gemini_key
                            )
                        st.write(explanation)
                    else:
                        st.info("💡 Enter Gemini API key in sidebar "
                                "for AI explanation")
                except Exception as e:
                    st.warning(f"Grad-CAM skipped: {e}")

    os.unlink(tmp_path)