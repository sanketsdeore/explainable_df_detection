# ============================================================
# app.py — Complete Deepfake Detection App with ECS + Gemini
# Run with:  streamlit run app.py
# ============================================================

import sys
import os
# Fix Windows Streamlit path resolution issues (remove file path entries from sys.path)
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)
sys.path = [p for p in sys.path if not p.endswith(".py")]

import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
import math
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
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

AVAILABLE_MODELS = {
    "Dima806 ViT (Vision Transformer)": "dima806/deepfake_vs_real_image_detection",
    "PrithivMLmods SigLIP (CNN/ViT Hybrid)": "prithivMLmods/Deep-Fake-Detector-Model",
    "SuriyaaMM EfficientNet (CNN)": "SuriyaaMM/google-efficientnet-b1-deepfake",
    "Purnachander Swin (Shifted Window)": "Purnachander-Konda/deepfake-detection-swin"
}

selected_model_label = st.sidebar.selectbox(
    "Select Pretrained Model", 
    list(AVAILABLE_MODELS.keys()) + ["Custom (HuggingFace ID)"],
    help="Choose a HuggingFace ViT model to use for deepfake detection. Different models have different ECS performance."
)

if selected_model_label == "Custom (HuggingFace ID)":
    selected_model_name = st.sidebar.text_input("Enter HuggingFace Model ID", "dima806/deepfake_vs_real_image_detection")
else:
    selected_model_name = AVAILABLE_MODELS[selected_model_label]

use_gemini = st.sidebar.checkbox(
    "Use Gemini AI for explanation",
    value=False,
    help="Enable to use Gemini 3.5 Flash instead of built-in explanation. Requires API key and is subject to rate limits."
)
gemini_key = ""
if use_gemini:
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

# ── Wrapped model for Grad-CAM ──
class WrappedModel(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m
    def forward(self, x):
        return self.model(pixel_values=x).logits

# ── Load models (cached so they don't reload every run) ──
@st.cache_resource
def load_models(model_name="dima806/deepfake_vs_real_image_detection"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Face detector
    mtcnn = MTCNN(keep_all=False, device=device)

    # Deepfake detector
    try:
        processor  = AutoImageProcessor.from_pretrained(model_name)
    except OSError:
        # Fallback if the model repository lacks a preprocessor_config.json
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    base_model  = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    base_model.eval()

    # Confirm label mapping
    id2label = base_model.config.id2label
    try:
        FAKE_IDX = [k for k, v in id2label.items() if 'fake' in v.lower()][0]
        REAL_IDX = [k for k, v in id2label.items() if 'real' in v.lower()][0]
    except IndexError:
        print(f"Label mapping could not strictly find 'fake'/'real' in model config: {id2label}. Defaulting to 1 for FAKE, 0 for REAL.")
        REAL_IDX = 0
        FAKE_IDX = 1

    return device, mtcnn, processor, base_model, FAKE_IDX, REAL_IDX

# ── Helper functions ──────────────────────────────────────

def reshape_transform(tensor):
    if tensor.ndim == 4:
        return tensor  # CNNs (ResNet, EfficientNet) already output 4D
    elif tensor.ndim == 3:
        B, N, C = tensor.shape
        # Check if it has a CLS token
        if int(N ** 0.5) ** 2 == N:
            # No CLS token (e.g. SigLIP)
            result = tensor
            H = W = int(N ** 0.5)
        elif int((N - 1) ** 0.5) ** 2 == (N - 1):
            # One CLS token (e.g. standard ViT)
            result = tensor[:, 1:, :]
            H = W = int((N - 1) ** 0.5)
        else:
            return tensor
        return result.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return tensor

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

def predict_face(face_np, processor, base_model, device, FAKE_IDX, REAL_IDX):
    image  = Image.fromarray(face_np)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(base_model(**inputs).logits, dim=1)[0]
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

def describe_gradcam_regions(gradcam_array):
    """Extract factual spatial statistics from the raw GradCAM heatmap array.
    Returns a text description of which facial quadrants have the highest
    activation — grounded entirely in the actual numpy data."""
    h, w = gradcam_array.shape
    mid_h, mid_w = h // 2, w // 2

    quadrants = {
        "top-left (forehead / left eye region)": gradcam_array[:mid_h, :mid_w],
        "top-right (forehead / right eye region)": gradcam_array[:mid_h, mid_w:],
        "bottom-left (left cheek / mouth region)": gradcam_array[mid_h:, :mid_w],
        "bottom-right (right cheek / chin region)": gradcam_array[mid_h:, mid_w:],
    }

    stats = []
    for name, region in quadrants.items():
        mean_val = float(region.mean())
        max_val = float(region.max())
        stats.append((name, mean_val, max_val))

    # Sort by mean activation descending
    stats.sort(key=lambda x: x[1], reverse=True)

    overall_mean = float(gradcam_array.mean())
    overall_max = float(gradcam_array.max())

    lines = [f"Overall heatmap — mean activation: {overall_mean:.3f}, peak activation: {overall_max:.3f}"]
    for name, mean_val, max_val in stats:
        lines.append(f"  • {name}: mean={mean_val:.3f}, max={max_val:.3f}")

    return "\n".join(lines)


def generate_local_explanation(label, gradcam_array, ecs_score=None):
    """Generate a grounded explanation from GradCAM stats and ECS score.
    No API call — instant, free, and cannot hallucinate."""
    h, w = gradcam_array.shape
    mid_h, mid_w = h // 2, w // 2

    quadrants = {
        "forehead/eye region": gradcam_array[:mid_h, :],
        "mouth/chin region": gradcam_array[mid_h:, :],
        "left side of face": gradcam_array[:, :mid_w],
        "right side of face": gradcam_array[:, mid_w:],
    }

    # Find the region with highest mean activation
    region_stats = [(name, float(region.mean())) for name, region in quadrants.items()]
    region_stats.sort(key=lambda x: x[1], reverse=True)
    top_region = region_stats[0][0]
    top_val = region_stats[0][1]
    second_region = region_stats[1][0]

    overall_mean = float(gradcam_array.mean())
    overall_max = float(gradcam_array.max())

    # Build explanation sentences
    parts = []

    # Sentence 1: Where the model focused
    if top_val > 0.5:
        parts.append(
            f"The model's attention was strongly concentrated on the **{top_region}** "
            f"(mean activation: {top_val:.3f}, peak: {overall_max:.3f}), "
            f"with secondary focus on the **{second_region}**."
        )
    elif top_val > 0.2:
        parts.append(
            f"The model showed moderate attention on the **{top_region}** "
            f"(mean activation: {top_val:.3f}), "
            f"with some focus also on the **{second_region}**."
        )
    else:
        parts.append(
            f"The model's attention was diffuse across the face "
            f"(overall mean: {overall_mean:.3f}), without strong focus on any single region."
        )

    # Sentence 2: What this means for the prediction
    if label == "FAKE":
        if top_val > 0.5:
            parts.append(
                f"The high activation in the {top_region} suggests the model detected "
                f"potential manipulation artifacts in that area."
            )
        else:
            parts.append(
                f"The model flagged this as fake but without strongly localized attention, "
                f"suggesting subtle or distributed manipulation."
            )
    else:
        if top_val > 0.3:
            parts.append(
                f"The model examined the {top_region} closely and found no signs of manipulation, "
                f"supporting the REAL classification."
            )
        else:
            parts.append(
                f"The model found no concentrated anomalies in any facial region, "
                f"consistent with an authentic image."
            )

    # Sentence 3: ECS context (video only)
    if ecs_score is not None:
        if ecs_score > 0.7:
            parts.append(
                f"The ECS score of {ecs_score:.4f} indicates highly consistent attention "
                f"across frames, suggesting a reliable detection."
            )
        elif ecs_score > 0.4:
            parts.append(
                f"The ECS score of {ecs_score:.4f} shows moderately consistent attention "
                f"across frames."
            )
        else:
            parts.append(
                f"The ECS score of {ecs_score:.4f} indicates inconsistent attention "
                f"across frames — the detection may be less reliable."
            )

    return " ".join(parts)


def get_gemini_explanation(label, gradcam_array, heatmap_pil, api_key, ecs_score=None):
    """Generate a brief AI explanation using Gemini 3.5 Flash, grounded on
    the GradCAM heatmap data and ECS score from this project."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.5-flash")

        region_desc = describe_gradcam_regions(gradcam_array)

        ecs_line = ""
        if ecs_score is not None:
            ecs_line = f"ECS (attention consistency across frames, 0-1): {ecs_score:.4f}"

        prompt = (
            f"Deepfake detection result: {label}.\n"
            f"Grad-CAM attention stats:\n{region_desc}\n"
            f"{ecs_line}\n"
            f"Using ONLY the stats above and the attached heatmap, "
            f"explain in 2-3 simple sentences which face regions the model focused on "
            f"and what that means for the {label} prediction. "
            f"Do not invent any data not provided above."
        )

        response = model.generate_content([prompt, heatmap_pil])
        return response.text
    except Exception as e:
        return f"Gemini explanation unavailable: {str(e)}"

def process_video(video_path, models, frame_interval, fake_threshold):
    device, mtcnn, processor, base_model, wrapped, target_layers, FAKE_IDX, REAL_IDX = models

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
                    face, processor, base_model, device, FAKE_IDX, REAL_IDX
                )
                cam_map = None
                heatmap_img = None
                if label == "FAKE":
                    try:
                        cam_map, heatmap_img = generate_gradcam(pv, face, wrapped, target_layers)
                        heatmaps.append(cam_map)
                        heatmap_images.append(heatmap_img)
                    except Exception as e:
                        import traceback
                        st.error(f"Grad-CAM Error: {traceback.format_exc()}")

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

with st.spinner(f"Loading model '{selected_model_name}'..."):
    models = load_models(selected_model_name)

device, mtcnn, processor, base_model, FAKE_IDX, REAL_IDX = models

# Create Grad-CAM wrapper and target layers on the fly
wrapped = WrappedModel(base_model)
wrapped.eval()

# Dynamically find target layer based on model architecture
if hasattr(base_model, "vit"):
    if hasattr(base_model.vit, "encoder"):
        target_layers = [base_model.vit.encoder.layer[-1].layernorm_before]
    else:
        target_layers = [base_model.vit.layers[-1].layernorm_before]
elif hasattr(base_model, "resnet"):
    target_layers = [base_model.resnet.encoder.stages[-1].layers[-1]]
elif hasattr(base_model, "efficientnet"):
    target_layers = [base_model.efficientnet.encoder.blocks[-1]]
elif hasattr(base_model, "swin"):
    target_layers = [base_model.swin.layernorm]
elif hasattr(base_model, "vision_model"):
    # Often SigLIP or CLIP-like vision encoders
    try:
        target_layers = [base_model.vision_model.encoder.layers[-1].layer_norm1]
    except:
        target_layers = [list(base_model.vision_model.children())[-1]]
elif hasattr(base_model, "xception"):
    try:
        target_layers = [base_model.xception.encoder.blocks[-1]]
    except:
        target_layers = [list(base_model.children())[-1]]
else:
    # Fallback to the last immediate child (often works for custom timm architectures)
    try:
        target_layers = [list(list(base_model.children())[0].children())[-1]]
    except:
        target_layers = [list(base_model.children())[-1]]

all_models = (device, mtcnn, processor, base_model, wrapped, target_layers, FAKE_IDX, REAL_IDX)

st.write(f"Using device: `{device}`")

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
                    tmp_path, all_models, frame_interval, fake_threshold
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

                # Explanation for most representative frame
                st.subheader("🤖 AI Explanation")
                if result["verdict"] == "FAKE":
                    most_suspicious = max(
                        result["frames"],
                        key=lambda x: x["fake_conf"]
                    )
                else:
                    most_suspicious = max(
                        result["frames"],
                        key=lambda x: x["real_conf"]
                    )

                # Get GradCAM raw array for the representative frame
                frame_gradcam_array = None
                if most_suspicious["heatmap"] is None:
                    try:
                        _, _, _, pv = predict_face(
                            most_suspicious["face"], processor, base_model,
                            device, FAKE_IDX, REAL_IDX
                        )
                        cam_arr, heatmap_img = generate_gradcam(
                            pv, most_suspicious["face"], wrapped, target_layers
                        )
                        most_suspicious["heatmap"] = heatmap_img
                        frame_gradcam_array = cam_arr
                    except:
                        pass
                else:
                    try:
                        _, _, _, pv = predict_face(
                            most_suspicious["face"], processor, base_model,
                            device, FAKE_IDX, REAL_IDX
                        )
                        frame_gradcam_array, _ = generate_gradcam(
                            pv, most_suspicious["face"], wrapped, target_layers
                        )
                    except:
                        pass

                if frame_gradcam_array is not None:
                    # Use Gemini if toggled on + key provided, else local
                    if use_gemini and gemini_key:
                        hm_pil = Image.fromarray(most_suspicious["heatmap"])
                        with st.spinner("Generating Gemini explanation..."):
                            explanation = get_gemini_explanation(
                                most_suspicious["label"],
                                frame_gradcam_array,
                                hm_pil,
                                gemini_key,
                                ecs_score=result["ecs"]
                            )
                    else:
                        explanation = generate_local_explanation(
                            most_suspicious["label"],
                            frame_gradcam_array,
                            ecs_score=result["ecs"]
                        )
                    st.write(explanation)

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
                    face_np, processor, base_model,
                    device, FAKE_IDX, REAL_IDX
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

                    # Explanation
                    st.subheader("🤖 AI Explanation")
                    if use_gemini and gemini_key:
                        hm_pil = Image.fromarray(heatmap_img)
                        with st.spinner("Generating Gemini explanation..."):
                             explanation = get_gemini_explanation(
                                 label, cam_map, hm_pil, gemini_key
                             )
                    else:
                        explanation = generate_local_explanation(label, cam_map)
                    st.write(explanation)
                except Exception as e:
                    st.warning(f"Grad-CAM skipped: {e}")

    os.unlink(tmp_path)