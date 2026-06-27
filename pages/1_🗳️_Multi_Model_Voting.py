# ============================================================
# pages/1_🗳️_Multi_Model_Voting.py — Multi-Model Deepfake Detection
# This page runs multiple models simultaneously and aggregates votes
# to dramatically reduce false positives and false negatives.
# ============================================================

import sys
import os
# Fix Windows Streamlit path resolution issues
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)
sys.path = [p for p in sys.path if not p.endswith(".py")]

import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
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
    page_title="Multi-Model Deepfake Detector",
    page_icon="🗳️",
    layout="wide"
)

st.title("🗳️ Multi-Model Deepfake Detection")
st.markdown(
    "Run **multiple models simultaneously** and aggregate their votes. "
    "Different architectures have different failure modes — when one model is wrong, "
    "the others outvote it."
)

# ── Available Models ─────────────────────────────────────
AVAILABLE_MODELS = {
    "Dima806 ViT (Vision Transformer)": "dima806/deepfake_vs_real_image_detection",
    "PrithivMLmods SigLIP (CNN/ViT Hybrid)": "prithivMLmods/Deep-Fake-Detector-Model",
    "SuriyaaMM EfficientNet (CNN)": "SuriyaaMM/google-efficientnet-b1-deepfake",
    "Purnachander Swin (Shifted Window)": "Purnachander-Konda/deepfake-detection-swin"
}

# ── Sidebar — Multi-Model Settings ───────────────────────
st.sidebar.header("⚙️ Multi-Model Settings")

st.sidebar.markdown("**Select models for voting:**")
selected_models = {}
for label, model_id in AVAILABLE_MODELS.items():
    if st.sidebar.checkbox(label, value=True, key=f"mm_{model_id}"):
        selected_models[label] = model_id

if len(selected_models) < 2:
    st.sidebar.warning("⚠️ Select at least 2 models for meaningful voting.")

frame_interval = st.sidebar.slider(
    "Frame sampling interval", 5, 30, 10,
    help="Analyze every Nth frame from video"
)
fake_threshold = st.sidebar.slider(
    "Fake ratio threshold", 0.1, 0.9, 0.4,
    help="If more than X% of frames are fake, video is FAKE"
)

use_gemini = st.sidebar.checkbox(
    "Use Gemini AI for explanation",
    value=False,
    help="Enable to use Gemini 2.5 Flash instead of built-in explanation. Requires API key."
)
gemini_key = ""
if use_gemini:
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Get your key from aistudio.google.com"
    )

# ── Wrapped model for Grad-CAM ──
class WrappedModel(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m
    def forward(self, x):
        return self.model(pixel_values=x).logits

# ── Load models (cached) ─────────────────────────────────
@st.cache_resource
def load_models_multi(model_items_tuple):
    """Load multiple models for multi-model voting.

    Args:
        model_items_tuple: tuple of (display_name, model_id) pairs
                           (must be a tuple for st.cache_resource hashability)

    Returns:
        device, mtcnn, model_list
        where model_list = [(display_name, processor, model, FAKE_IDX, REAL_IDX), ...]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Multi-Model] Using device: {device}")

    mtcnn = MTCNN(keep_all=False, device=device)
    models = []

    for display_name, model_id in model_items_tuple:
        print(f"[Multi-Model] Loading {display_name} ({model_id})...")
        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
        except OSError:
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

        model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
        model.eval()

        id2label = model.config.id2label
        try:
            FAKE_IDX = [k for k, v in id2label.items() if 'fake' in v.lower()][0]
            REAL_IDX = [k for k, v in id2label.items() if 'real' in v.lower()][0]
        except IndexError:
            print(f"[Multi-Model] Label mapping issue for {display_name}: {id2label}. Defaulting.")
            REAL_IDX = 0
            FAKE_IDX = 1

        models.append((display_name, processor, model, FAKE_IDX, REAL_IDX))

    return device, mtcnn, models


# ── Dynamically find Grad-CAM target layers ──────────────
def get_target_layers(base_model):
    """Find the appropriate target layer for Grad-CAM based on model architecture."""
    if hasattr(base_model, "vit"):
        if hasattr(base_model.vit, "encoder"):
            return [base_model.vit.encoder.layer[-1].layernorm_before]
        else:
            return [base_model.vit.layers[-1].layernorm_before]
    elif hasattr(base_model, "resnet"):
        return [base_model.resnet.encoder.stages[-1].layers[-1]]
    elif hasattr(base_model, "efficientnet"):
        return [base_model.efficientnet.encoder.blocks[-1]]
    elif hasattr(base_model, "swin"):
        return [base_model.swin.layernorm]
    elif hasattr(base_model, "vision_model"):
        try:
            return [base_model.vision_model.encoder.layers[-1].layer_norm1]
        except:
            return [list(base_model.vision_model.children())[-1]]
    elif hasattr(base_model, "xception"):
        try:
            return [base_model.xception.encoder.blocks[-1]]
        except:
            return [list(base_model.children())[-1]]
    else:
        try:
            return [list(list(base_model.children())[0].children())[-1]]
        except:
            return [list(base_model.children())[-1]]


# ── Helper functions ──────────────────────────────────────

def reshape_transform(tensor):
    if tensor.ndim == 4:
        return tensor
    elif tensor.ndim == 3:
        B, N, C = tensor.shape
        if int(N ** 0.5) ** 2 == N:
            result = tensor
            H = W = int(N ** 0.5)
        elif int((N - 1) ** 0.5) ** 2 == (N - 1):
            result = tensor[:, 1:, :]
            H = W = int((N - 1) ** 0.5)
        else:
            return tensor
        return result.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return tensor

def extract_face(frame_rgb, mtcnn, prob_threshold=0.85):
    h, w, _ = frame_rgb.shape
    boxes, probs = mtcnn.detect(frame_rgb)
    if boxes is None or probs is None:
        return None
    if probs[0] is None or probs[0] < prob_threshold:
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


def predict_face_multi(face_np, model_list, device):
    """Run a face through all models and aggregate predictions.

    Uses both soft voting (average confidence) and hard voting (majority count).
    Ties (equal FAKE and REAL votes) produce a CONTESTED verdict.
    """
    per_model_results = []

    for display_name, processor, model, FAKE_IDX, REAL_IDX in model_list:
        pred_label, fake_conf, real_conf, pv = predict_face(
            face_np, processor, model, device, FAKE_IDX, REAL_IDX
        )
        per_model_results.append({
            "model_name": display_name,
            "label": pred_label,
            "fake_conf": fake_conf,
            "real_conf": real_conf,
            "pixel_values": pv,
        })

    # Soft voting: average confidence scores
    avg_fake_conf = float(np.mean([r["fake_conf"] for r in per_model_results]))
    avg_real_conf = float(np.mean([r["real_conf"] for r in per_model_results]))

    # Hard voting: count FAKE votes
    n_fake_votes = sum(1 for r in per_model_results if r["label"] == "FAKE")
    n_models = len(per_model_results)
    n_real_votes = n_models - n_fake_votes

    # Determine verdict — ties are CONTESTED
    if n_fake_votes == n_real_votes:
        label = "CONTESTED"
    elif n_fake_votes > n_real_votes:
        label = "FAKE"
    else:
        label = "REAL"

    vote_ratio = n_fake_votes / n_models
    return label, avg_fake_conf, avg_real_conf, per_model_results, vote_ratio


def generate_gradcam(pixel_values, face_np, wrapped, target_layers):
    cam = GradCAM(
        model=wrapped,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )
    grayscale_cam = cam(input_tensor=pixel_values)[0]
    
    # Resize rgb_float to match grayscale_cam shape
    h, w = grayscale_cam.shape
    face_resized = cv2.resize(face_np, (w, h))
    rgb_float = face_resized.astype(np.float32) / 255.0
    
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
        h1 = cv2.resize(normalize_heatmap(heatmaps[i]), (224, 224), interpolation=cv2.INTER_LINEAR)
        h2 = cv2.resize(normalize_heatmap(heatmaps[i+1]), (224, 224), interpolation=cv2.INTER_LINEAR)
        cos = np.dot(h1.flatten(), h2.flatten()) / (
            np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8
        )
        struct = ssim(h1, h2, data_range=1.0)
        scores.append((cos + struct) / 2.0)
    return float(np.mean(scores))


# ── Explanation generators ────────────────────────────────

def describe_gradcam_regions(gradcam_array):
    """Extract factual spatial statistics from the raw GradCAM heatmap array."""
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
        stats.append((name, float(region.mean()), float(region.max())))
    stats.sort(key=lambda x: x[1], reverse=True)

    lines = [f"Overall heatmap — mean: {float(gradcam_array.mean()):.3f}, max: {float(gradcam_array.max()):.3f}"]
    for name, mean_val, max_val in stats:
        lines.append(f"  • {name}: mean={mean_val:.3f}, max={max_val:.3f}")
    return "\n".join(lines)

def generate_local_explanation(label, gradcam_array, ecs_score=None, vote_info=None):
    """Generate a grounded explanation from GradCAM stats, ECS score, and vote info.

    Args:
        label: "FAKE", "REAL", or "CONTESTED"
        gradcam_array: raw Grad-CAM heatmap numpy array
        ecs_score: optional float (video only)
        vote_info: optional dict with keys:
            n_fake_votes, n_total, vote_ratio
    """
    h, w = gradcam_array.shape
    mid_h, mid_w = h // 2, w // 2
    quadrants = {
        "forehead/eye region": gradcam_array[:mid_h, :],
        "mouth/chin region": gradcam_array[mid_h:, :],
        "left side of face": gradcam_array[:, :mid_w],
        "right side of face": gradcam_array[:, mid_w:],
    }
    region_stats = [(name, float(region.mean())) for name, region in quadrants.items()]
    region_stats.sort(key=lambda x: x[1], reverse=True)
    top_region, top_val = region_stats[0]
    second_region = region_stats[1][0]
    overall_mean = float(gradcam_array.mean())
    overall_max = float(gradcam_array.max())

    parts = []

    # Sentence 1: Where the models focused
    if top_val > 0.5:
        parts.append(f"The models' attention was strongly concentrated on the **{top_region}** (mean activation: {top_val:.3f}, peak: {overall_max:.3f}), with secondary focus on the **{second_region}**.")
    elif top_val > 0.2:
        parts.append(f"The models showed moderate attention on the **{top_region}** (mean activation: {top_val:.3f}), with some focus also on the **{second_region}**.")
    else:
        parts.append(f"The models' attention was diffuse across the face (overall mean: {overall_mean:.3f}), without strong focus on any single region.")

    # Sentence 2: What this means for the prediction
    if label == "FAKE":
        if top_val > 0.5:
            parts.append(f"The high activation in the {top_region} suggests the models detected potential manipulation artifacts in that area.")
        else:
            parts.append("The models flagged this as fake but without strongly localized attention, suggesting subtle or distributed manipulation.")
    elif label == "CONTESTED":
        parts.append(f"The models are evenly split on this classification — some detected potential artifacts in the {top_region} while others found the face consistent with authentic content. This result requires manual review.")
    else:
        if top_val > 0.3:
            parts.append(f"The models examined the {top_region} closely and found no signs of manipulation, supporting the REAL classification.")
        else:
            parts.append("The models found no concentrated anomalies in any facial region, consistent with an authentic image.")

    # Sentence 3: ECS context (video only)
    if ecs_score is not None:
        if ecs_score > 0.7:
            parts.append(f"The ECS score of {ecs_score:.4f} indicates highly consistent attention across frames.")
        elif ecs_score > 0.4:
            parts.append(f"The ECS score of {ecs_score:.4f} shows moderately consistent attention across frames.")
        else:
            parts.append(f"The ECS score of {ecs_score:.4f} indicates inconsistent attention across frames.")

    # Sentence 4: Voting consensus
    if vote_info is not None:
        n_fv = vote_info["n_fake_votes"]
        n_t = vote_info["n_total"]
        n_rv = n_t - n_fv

        if label == "CONTESTED":
            parts.append(f"The vote was tied at {n_fv}/{n_t} models voting FAKE and {n_rv}/{n_t} voting REAL, resulting in a contested verdict that requires further investigation.")
        elif n_fv == n_t and label == "FAKE":
            parts.append(f"All {n_t} models unanimously voted FAKE, providing very high confidence in this classification.")
        elif n_rv == n_t and label == "REAL":
            parts.append(f"All {n_t} models unanimously voted REAL, providing very high confidence in this classification.")
        else:
            majority = n_fv if label == "FAKE" else n_rv
            strength = "strong" if majority / n_t >= 0.75 else "moderate"
            parts.append(f"{majority} out of {n_t} models voted {label}. The {strength} majority supports this classification.")

    return " ".join(parts)

def get_gemini_explanation(label, gradcam_array, heatmap_pil, api_key, ecs_score=None, vote_info=None):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        region_desc = describe_gradcam_regions(gradcam_array)
        ecs_line = f"ECS (attention consistency across frames, 0-1): {ecs_score:.4f}" if ecs_score is not None else ""

        vote_line = ""
        if vote_info is not None:
            n_fv = vote_info["n_fake_votes"]
            n_t = vote_info["n_total"]
            vote_line = f"Multi-model voting: {n_fv}/{n_t} models voted FAKE, {n_t - n_fv}/{n_t} voted REAL. Verdict: {label}."

        prompt = (
            f"Deepfake detection result: {label}.\n"
            f"Grad-CAM attention stats:\n{region_desc}\n"
            f"{ecs_line}\n"
            f"{vote_line}\n"
            f"Using ONLY the stats above and the attached heatmap, explain in 2-3 simple sentences "
            f"which face regions the model focused on and what that means for the {label} prediction. "
            f"Do not invent any data not provided above."
        )
        response = model.generate_content([prompt, heatmap_pil])
        return response.text
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"


# ── Vote display helpers ──────────────────────────────────

def display_vote_breakdown(per_model_results, verdict):
    """Display a styled per-model vote breakdown."""
    st.markdown("#### 🗳️ Per-Model Vote Breakdown")

    n_fake = sum(1 for r in per_model_results if r["label"] == "FAKE")
    n_real = len(per_model_results) - n_fake
    n_total = len(per_model_results)

    for r in per_model_results:
        if verdict == "CONTESTED":
            # In a tie, no model "agrees" or "dissents" — just show votes
            if r["label"] == "FAKE":
                st.markdown(f"🔴 **{r['model_name']}**: FAKE ({r['fake_conf']:.1%})")
            else:
                st.markdown(f"🟢 **{r['model_name']}**: REAL ({r['real_conf']:.1%})")
        else:
            agrees = r["label"] == verdict
            icon = "✅" if agrees else "❌"
            dissent = "" if agrees else " ← *dissenting*"
            if r["label"] == "FAKE":
                st.markdown(f"{icon} **{r['model_name']}**: 🔴 FAKE ({r['fake_conf']:.1%}){dissent}")
            else:
                st.markdown(f"{icon} **{r['model_name']}**: 🟢 REAL ({r['real_conf']:.1%}){dissent}")

    # Consensus strength badge
    if verdict == "CONTESTED":
        st.warning(f"⚖️ **TIED** — {n_fake} models voted FAKE and {n_real} voted REAL. Verdict is **CONTESTED**.")
    else:
        n_agree = sum(1 for r in per_model_results if r["label"] == verdict)
        ratio = n_agree / n_total
        if ratio == 1.0:
            st.success(f"💪 **UNANIMOUS** — All {n_total} models agree: {verdict}")
        elif ratio >= 0.75:
            st.success(f"✅ **STRONG MAJORITY** — {n_agree}/{n_total} models agree: {verdict}")
        elif ratio > 0.5:
            st.warning(f"⚠️ **NARROW MAJORITY** — {n_agree}/{n_total} models agree: {verdict}")


def get_best_model_for_gradcam(per_model_results, verdict, model_list):
    """Find the most confident model matching the verdict and return
    its base_model, processor, pixel_values, FAKE_IDX, REAL_IDX.
    For CONTESTED, pick the model with highest fake_conf (investigate the fake suspicion)."""
    if verdict == "FAKE" or verdict == "CONTESTED":
        best = max(per_model_results, key=lambda r: r["fake_conf"])
    else:
        best = max(per_model_results, key=lambda r: r["real_conf"])

    for dn, proc, mod, fi, ri in model_list:
        if dn == best["model_name"]:
            return mod, proc, best["pixel_values"], fi, ri, best["model_name"]

    return None, None, None, None, None, None


# ── Video processing (multi-model) ───────────────────────

def process_video_multi(video_path, device, mtcnn, model_list,
                        frame_interval, fake_threshold):
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
                label, fake_conf, real_conf, per_model, vote_ratio = predict_face_multi(
                    face, model_list, device
                )

                # Select best model for Grad-CAM
                best_model, best_proc, best_pv, best_fi, best_ri, best_name = \
                    get_best_model_for_gradcam(per_model, label, model_list)

                cam_map = None
                heatmap_img = None
                if best_model is not None:
                    try:
                        w = WrappedModel(best_model)
                        w.eval()
                        tl = get_target_layers(best_model)
                        cam_map, heatmap_img = generate_gradcam(best_pv, face, w, tl)
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
                    "per_model":  per_model,
                    "vote_ratio": vote_ratio,
                    "best_model": best_name,
                })
                analyzed += 1

                n_fv = sum(1 for r in per_model if r["label"] == "FAKE")
                status.text(f"Analyzed frame {count} → {label} "
                            f"(votes: {n_fv}/{len(per_model)} FAKE, "
                            f"avg conf={fake_conf:.2%})")

        count += 1
        progress.progress(min(count / max(total_frames, 1), 1.0))

    cap.release()
    progress.empty()
    status.empty()

    if not frame_results:
        return None

    n_fake      = sum(1 for r in frame_results if r["label"] == "FAKE")
    n_contested = sum(1 for r in frame_results if r["label"] == "CONTESTED")
    n_total     = len(frame_results)
    fake_ratio  = n_fake / n_total
    contested_ratio = n_contested / n_total

    # Video-level verdict
    if fake_ratio > fake_threshold:
        verdict = "FAKE"
    elif contested_ratio > 0.5:
        verdict = "CONTESTED"
    else:
        verdict = "REAL"

    if verdict == "FAKE":
        avg_conf = np.mean([r["fake_conf"] for r in frame_results if r["label"] == "FAKE"]) if n_fake > 0 else 0.0
    elif verdict == "REAL":
        n_real = n_total - n_fake - n_contested
        avg_conf = np.mean([r["real_conf"] for r in frame_results if r["label"] == "REAL"]) if n_real > 0 else 0.0
    else:
        # CONTESTED
        avg_conf = np.mean([r["fake_conf"] for r in frame_results if r["label"] == "CONTESTED"]) if n_contested > 0 else 0.0
    ecs      = compute_ecs(heatmaps)

    return {
        "verdict":          verdict,
        "fake_ratio":       fake_ratio,
        "contested_ratio":  contested_ratio,
        "avg_conf":         avg_conf,
        "ecs":              ecs,
        "n_fake":           n_fake,
        "n_contested":      n_contested,
        "n_total":          n_total,
        "frames":           frame_results,
        "heatmaps":         heatmap_images,
    }


# ── Main UI ───────────────────────────────────────────────

if len(selected_models) < 2:
    st.warning("⚠️ Please select at least **2 models** from the sidebar to use multi-model voting.")
    st.info("💡 Tip: Use the main **Deepfake Detector** page for single-model analysis.")
    st.stop()

# Load models
model_items = tuple(sorted(selected_models.items()))
with st.spinner(f"Loading {len(selected_models)} models... (first run downloads from HuggingFace)"):
    device, mtcnn, model_list = load_models_multi(model_items)

st.write(f"Using device: `{device}` | 🗳️ **{len(model_list)} models loaded**")

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
        if st.button("🗳️ Analyze Video"):
            with st.spinner("Analyzing video with multiple models..."):
                result = process_video_multi(
                    tmp_path, device, mtcnn, model_list,
                    frame_interval, fake_threshold
                )

            if result is None:
                st.error("No faces detected in video.")
                os.unlink(tmp_path)
                st.stop()
            else:
                # Verdict banner
                if result["verdict"] == "FAKE":
                    st.markdown(f"## 🔴 Multi-Model Verdict: **FAKE**")
                elif result["verdict"] == "CONTESTED":
                    st.markdown(f"## 🟡 Multi-Model Verdict: **CONTESTED**")
                    st.warning("⚖️ The models are evenly split — this result is inconclusive and requires manual review.")
                else:
                    st.markdown(f"## 🟢 Multi-Model Verdict: **REAL**")

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Fake Frames",
                            f"{result['n_fake']}/{result['n_total']}")
                col2.metric("Fake Ratio",
                            f"{result['fake_ratio']:.1%}")
                conf_label = "Avg Conf (Fake)" if result['verdict'] == "FAKE" else "Avg Conf (Real)" if result['verdict'] == "REAL" else "Avg Conf (Contested)"
                col3.metric(conf_label,
                            f"{result['avg_conf']:.1%}")
                col4.metric("ECS Score",
                            f"{result['ecs']:.4f}",
                            help="Explainability Consistency Score — "
                                 "higher = more stable attention across frames")

                if result["n_contested"] > 0:
                    st.info(f"⚖️ {result['n_contested']}/{result['n_total']} frames had tied votes (CONTESTED)")

                # ── Model Agreement (video-level aggregate) ──
                st.markdown("---")
                st.subheader("🗳️ Model Agreement")

                model_names = [r["model_name"] for r in result["frames"][0].get("per_model", [])]
                # Compute video-level per-model agreement
                video_model_fake_counts = {}
                video_model_confs = {}
                if model_names:
                    video_model_fake_counts = {name: 0 for name in model_names}
                    video_model_confs = {name: [] for name in model_names}

                    for frame_r in result["frames"]:
                        for pm in frame_r.get("per_model", []):
                            if pm["label"] == "FAKE":
                                video_model_fake_counts[pm["model_name"]] += 1
                            video_model_confs[pm["model_name"]].append(pm["fake_conf"])

                    n_frames = len(result["frames"])

                    # Per-model display
                    for name in model_names:
                        fake_pct = video_model_fake_counts[name] / n_frames
                        avg_fc = float(np.mean(video_model_confs[name]))
                        model_verdict = "FAKE" if fake_pct > fake_threshold else "REAL"

                        if result["verdict"] == "CONTESTED":
                            v_icon = "🔴" if model_verdict == "FAKE" else "🟢"
                            st.markdown(
                                f"{v_icon} **{name}**: {model_verdict} "
                                f"(fake in {fake_pct:.0%} of frames, avg conf: {avg_fc:.1%})"
                            )
                        else:
                            agrees = model_verdict == result["verdict"]
                            icon = "✅" if agrees else "❌"
                            v_icon = "🔴" if model_verdict == "FAKE" else "🟢"
                            dissent = "" if agrees else " ← *dissenting*"
                            st.markdown(
                                f"{icon} **{name}**: {v_icon} {model_verdict} "
                                f"(fake in {fake_pct:.0%} of frames, avg conf: {avg_fc:.1%}){dissent}"
                            )

                    # Overall consensus badge
                    video_model_verdicts = []
                    for name in model_names:
                        fake_pct = video_model_fake_counts[name] / n_frames
                        video_model_verdicts.append("FAKE" if fake_pct > fake_threshold else "REAL")

                    n_fake_models = sum(1 for v in video_model_verdicts if v == "FAKE")
                    n_real_models = sum(1 for v in video_model_verdicts if v == "REAL")
                    n_total_models = len(video_model_verdicts)

                    if result["verdict"] == "CONTESTED":
                        st.warning(f"⚖️ **TIED** — {n_fake_models} models lean FAKE and {n_real_models} lean REAL across all frames")
                    elif n_fake_models == n_total_models or n_real_models == n_total_models:
                        st.success(f"💪 **UNANIMOUS** — All {n_total_models} models agree: {result['verdict']}")
                    else:
                        n_agree = sum(1 for v in video_model_verdicts if v == result["verdict"])
                        ratio = n_agree / n_total_models
                        if ratio >= 0.75:
                            st.success(f"✅ **STRONG MAJORITY** — {n_agree}/{n_total_models} models agree")
                        elif ratio > 0.5:
                            st.warning(f"⚠️ **NARROW MAJORITY** — {n_agree}/{n_total_models} models agree")
                        else:
                            st.error(f"❓ **WEAK AGREEMENT** — Only {n_agree}/{n_total_models} models agree")

                # ── Explanation for most representative frame ──
                st.markdown("---")
                st.subheader("🤖 AI Explanation")

                if result["verdict"] == "FAKE":
                    most_suspicious = max(result["frames"], key=lambda x: x["fake_conf"])
                elif result["verdict"] == "CONTESTED":
                    most_suspicious = max(result["frames"], key=lambda x: x["fake_conf"])
                else:
                    most_suspicious = max(result["frames"], key=lambda x: x["real_conf"])

                # Generate heatmaps for all models on the representative frame
                cam_maps = []
                heatmap_imgs = []
                names = []
                best_cam_map = None
                best_heatmap_img = None

                best_model, best_proc, best_pv, best_fi, best_ri, best_name = \
                    get_best_model_for_gradcam(most_suspicious["per_model"], result["verdict"], model_list)

                for pm in most_suspicious["per_model"]:
                    mod_info = next((m for m in model_list if m[0] == pm["model_name"]), None)
                    if mod_info is not None:
                        try:
                            w = WrappedModel(mod_info[2])
                            w.eval()
                            tl = get_target_layers(mod_info[2])
                            c_map, h_img = generate_gradcam(pm["pixel_values"], most_suspicious["face"], w, tl)
                            cam_maps.append(c_map)
                            heatmap_imgs.append(h_img)
                            names.append(pm["model_name"])

                            if pm["model_name"] == best_name:
                                best_cam_map = c_map
                                best_heatmap_img = h_img
                        except:
                            pass

                # Show all heatmaps for the representative frame
                if heatmap_imgs:
                    st.markdown(f"**Heatmaps for representative Frame {most_suspicious['frame']}**")
                    cols = st.columns(len(heatmap_imgs))
                    for i, (h_img, m_name) in enumerate(zip(heatmap_imgs, names)):
                        cols[i].image(h_img, caption=f"{m_name}", use_container_width=True)

                if best_cam_map is not None:
                    # Use VIDEO-LEVEL model agreement for explanation (not per-frame!)
                    if model_names and video_model_fake_counts:
                        n_frames = len(result["frames"])
                        vid_n_fake_models = sum(
                            1 for name in model_names
                            if video_model_fake_counts[name] / n_frames > fake_threshold
                        )
                        vote_info = {
                            "n_fake_votes": vid_n_fake_models,
                            "n_total": len(model_names),
                            "vote_ratio": vid_n_fake_models / len(model_names)
                        }
                    else:
                        # Fallback: use the representative frame's votes
                        n_fv = sum(1 for r in most_suspicious["per_model"] if r["label"] == "FAKE")
                        vote_info = {
                            "n_fake_votes": n_fv,
                            "n_total": len(most_suspicious["per_model"]),
                            "vote_ratio": n_fv / len(most_suspicious["per_model"])
                        }

                    if use_gemini and gemini_key:
                        with st.spinner("Generating AI explanation..."):
                            explanation = get_gemini_explanation(
                                result["verdict"], best_cam_map, Image.fromarray(best_heatmap_img),
                                gemini_key, ecs_score=result["ecs"], vote_info=vote_info
                            )
                    else:
                        explanation = generate_local_explanation(
                            result["verdict"], best_cam_map,
                            ecs_score=result["ecs"], vote_info=vote_info
                        )
                    st.write(explanation)

                # Frame-by-frame table
                with st.expander("📊 Frame-by-frame results"):
                    for r in result["frames"]:
                        if r["label"] == "FAKE":
                            emoji = "🔴"
                        elif r["label"] == "CONTESTED":
                            emoji = "🟡"
                        else:
                            emoji = "🟢"
                        per = r.get("per_model", [])
                        n_fv = sum(1 for p in per if p["label"] == "FAKE")
                        st.write(f"{emoji} Frame {r['frame']:4d} | "
                                 f"{r['label']} | "
                                 f"Fake: {r['fake_conf']:.3f} | "
                                 f"Real: {r['real_conf']:.3f} | "
                                 f"Votes: {n_fv}/{len(per)} FAKE")

    # ── IMAGE ──
    else:
        image = Image.open(tmp_path).convert("RGB")
        st.image(image, caption="Uploaded image", width=300)

        if st.button("🗳️ Analyze Image"):
            face_np = extract_face(np.array(image), mtcnn)

            if face_np is None:
                st.error("No face detected in image.")
                os.unlink(tmp_path)
                st.stop()
            else:
                label, fake_conf, real_conf, per_model_results, vote_ratio = predict_face_multi(
                    face_np, model_list, device
                )

                # Result
                if label == "FAKE":
                    st.markdown(f"## 🔴 Multi-Model Prediction: **FAKE**")
                elif label == "CONTESTED":
                    st.markdown(f"## 🟡 Multi-Model Prediction: **CONTESTED**")
                    st.warning("⚖️ The models are evenly split — this result is inconclusive and requires manual review.")
                else:
                    st.markdown(f"## 🟢 Multi-Model Prediction: **REAL**")

                col1, col2 = st.columns(2)
                col1.metric("Avg Fake Confidence", f"{fake_conf:.1%}")
                col2.metric("Avg Real Confidence", f"{real_conf:.1%}")

                # Vote breakdown
                st.markdown("---")
                display_vote_breakdown(per_model_results, label)
                st.markdown("---")

                # Grad-CAM for all models
                try:
                    st.subheader("🔥 Model Heatmaps")
                    c_face, c_heatmaps = st.columns([1, 3])
                    c_face.image(face_np, caption="Detected Face", use_container_width=True)

                    cam_maps = []
                    heatmap_imgs = []
                    names = []
                    best_cam_map = None
                    best_heatmap_img = None

                    best_model, best_proc, best_pv, best_fi, best_ri, best_name = \
                        get_best_model_for_gradcam(per_model_results, label, model_list)

                    for pm in per_model_results:
                        mod_info = next((m for m in model_list if m[0] == pm["model_name"]), None)
                        if mod_info is not None:
                            try:
                                w = WrappedModel(mod_info[2])
                                w.eval()
                                tl = get_target_layers(mod_info[2])
                                c_map, h_img = generate_gradcam(pm["pixel_values"], face_np, w, tl)
                                cam_maps.append(c_map)
                                heatmap_imgs.append(h_img)
                                names.append(pm["model_name"])

                                if pm["model_name"] == best_name:
                                    best_cam_map = c_map
                                    best_heatmap_img = h_img
                            except:
                                pass

                    if heatmap_imgs:
                        with c_heatmaps:
                            h_cols = st.columns(len(heatmap_imgs))
                            for i, (h_img, m_name) in enumerate(zip(heatmap_imgs, names)):
                                h_cols[i].image(h_img, caption=m_name, use_container_width=True)

                    # Explanation
                    if best_cam_map is not None:
                        st.subheader("🤖 AI Explanation")
                        n_fv = sum(1 for r in per_model_results if r["label"] == "FAKE")
                        vote_info = {
                            "n_fake_votes": n_fv,
                            "n_total": len(per_model_results),
                            "vote_ratio": n_fv / len(per_model_results)
                        }

                        if use_gemini and gemini_key:
                            with st.spinner("Generating AI explanation..."):
                                explanation = get_gemini_explanation(
                                    label, best_cam_map, Image.fromarray(best_heatmap_img),
                                    gemini_key, vote_info=vote_info
                                )
                        else:
                            explanation = generate_local_explanation(
                                label, best_cam_map, vote_info=vote_info
                            )
                        st.write(explanation)
                except Exception as e:
                    st.warning(f"Grad-CAM skipped: {e}")

    os.unlink(tmp_path)
