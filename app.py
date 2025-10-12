# app.py — AI Deepfake Detector
# A user-friendly Streamlit application for deepfake classification.

import os, io, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
#      CONFIGURATION
# =========================

# --- Paths & Model Settings ---
# This path must point to your trained model file.
CKPT_PATH = Path(r"resnet50_mixed_export\resnet50_best.pt")
IMG_SIZE = 224
ALLOWED_IMG = {"png", "jpg", "jpeg"}
ALLOWED_VID = {"mp4", "mov", "avi", "mkv"}

# --- Constants ---
# Fixed thresholds replace the "Sensitivity" feature for a simpler UI.
FACE_DETECTION_CONFIDENCE = 0.35  # Confidence for YOLOv8 face detector.
CLASSIFICATION_THRESHOLD = 0.5   # The tipping point for fake vs. real.

# ===> NEW: Video-Specific Tuning Parameters <===
# The minimum size (in pixels) for a detected face to be considered valid for analysis.
MIN_FACE_SIZE = 64
# The average p(fake) score a video must have to be considered a candidate for being fake.
VIDEO_AVG_THRESHOLD = 0.70
# The p(fake) score a single frame must have to be considered a "highly confident" fake.
HIGH_CONF_THRESHOLD = 0.80
# The minimum percentage of "highly confident" frames required to flag a video as fake.
HIGHLY_FAKE_RATE_THRESHOLD = 0.25 # i.e., 10%

# --- Styling ---
PRIMARY = "#1F6FEB"
OK = "#16a34a"
WARN = "#ef4444"
MUTED = "#6b7280"

st.set_page_config(page_title="AI Deepfake Detector", page_icon="🤖", layout="wide")

# =========================
#        STYLING
# =========================
st.markdown(f"""
<style>
/* Core layout and font adjustments */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background: #ffffff;
    color: #111827;
}}
.stButton>button {{ border-radius: 12px; padding: 0.6rem 1.0rem; }}

/* Custom UI Elements */
.hero {{ padding: 1rem 1rem 0rem 1rem; }}
.hero h1 {{ margin: 0; font-size: 32px; letter-spacing: -.5px; }}
.hero .subtitle {{ color: {MUTED}; font-size: 16px; margin-top: 0.25rem; }}
.section-header {{ margin-top: 2rem; margin-bottom: 1rem; }}

/* Verdict Chip (Real/Fake Label) */
.verdict {{
    display: inline-block; padding: 10px 16px; border-radius: 999px;
    font-weight: 700; font-size: 20px; color: #fff !important; line-height: 1;
}}
.verdict-real {{ background: {OK}; }}
.verdict-fake {{ background: {WARN}; }}

/* Confidence Bar */
.confidence-text {{ color: {MUTED}; font-size: 14px; margin-bottom: 0.25rem;}}
.confbar {{ width: 100%; height: 14px; background: #e5e7eb; border-radius: 10px; overflow: hidden; }}
.confbar > div {{ height: 100%; }}
.confbar .real-bar {{ background: {OK}; }}
.confbar .fake-bar {{ background: {WARN}; }}

/* Expander for "Explain the AI" */
.stExpander {{ border: none; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-radius: 12px; }}
.stExpander header {{ font-weight: 600; }}
.explanation-text {{ font-size: 15px; color: #374151; }}
.explanation-text strong {{ color: #111827; }}

/* Hide default Streamlit elements */
[data-testid="stFileUploaderDropzone"] small {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# =========================
#   MODEL & UTILITIES
# =========================
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

@st.cache_resource
def load_model_and_meta(ckpt_path: Path):
    """Loads the ResNet-50 model and associated metadata from a checkpoint."""
    if not ckpt_path.exists():
        st.error(f"FATAL: Checkpoint not found at '{ckpt_path}'. Please place your model file in the correct directory.")
        st.stop()
    try:
        # Use weights_only=False as our checkpoint contains non-tensor data
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        st.stop()

    model = models.resnet50() # Recreate the base architecture
    model.fc = nn.Linear(model.fc.in_features, 2) # Adapt the final layer
    model.load_state_dict(ckpt["model"]) # Load the trained weights
    model.eval()

    class_to_idx = ckpt.get("class_to_idx", {"fake": 0, "real": 1})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, class_to_idx, device

@st.cache_resource
def get_transforms():
    """Returns the required image transformations for the model."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

@st.cache_resource
def get_yolo_face_detector():
    """Loads the YOLOv8-face model for face detection."""
    return YOLO("yolov8n-face.pt")

# ... [previous code] ...

@st.cache_resource
def get_yolo_face_detector():
    """Loads the YOLOv8-face model for face detection."""
    return YOLO("yolov8n-face.pt")

# ===> ADD THIS NEW FUNCTION <===
@st.cache_resource
def get_haar_face_detector():
    """Loads the pre-built Haar Cascade model from OpenCV."""
    # This path points to the XML file included with the cv2 library
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(haar_cascade_path)

def _crop_face(frame_bgr, box, scale=1.3):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = int(max(x2 - x1, y2 - y1) * scale)
    x1_new, y1_new = max(0, cx - side // 2), max(0, cy - side // 2)
    x2_new, y2_new = min(w, x1_new + side), min(h, y1_new + side)
    return frame_bgr[y1_new:y2_new, x1_new:x2_new]

def detect_largest_face(frame_bgr, conf_threshold):
    """
    Detects the largest face in an image.
    Tries YOLOv8-face first, then falls back to Haar Cascade if YOLO finds nothing.
    """
    # --- Primary Method: YOLOv8-face ---
    try:
        yolo = get_yolo_face_detector()
        device = 0 if torch.cuda.is_available() else "cpu"
        res = yolo.predict(frame_bgr[..., ::-1], conf=conf_threshold, verbose=False, device=device)
        
        boxes = []
        if res and res[0].boxes:
            for box in res[0].boxes:
                xyxy = box.xyxy.cpu().numpy().squeeze()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                boxes.append((xyxy, area))
        
        if boxes:
            # If YOLO found faces, pick the largest one and return it
            largest_box, _ = max(boxes, key=lambda item: item[1])
            return _crop_face(frame_bgr, largest_box)
    except Exception as e:
        st.warning(f"YOLO detector failed with an error: {e}")

    # --- Fallback Method: Haar Cascade ---
    # This code only runs if YOLO returned no boxes
    try:
        haar_detector = get_haar_face_detector()
        gray_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # detectMultiScale returns list of (x, y, w, h)
        faces = haar_detector.detectMultiScale(gray_img, 1.1, 4, minSize=(64, 64))
        
        if len(faces) > 0:
            # Find the face with the largest area
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            # Convert (x,y,w,h) to (x1,y1,x2,y2) for the crop function
            return _crop_face(frame_bgr, (x, y, x + w, y + h))
    except Exception as e:
        st.warning(f"Haar Cascade detector failed with an error: {e}")

    # If both detectors fail, return None
    return None
@torch.no_grad()
def predict_pil_images(model, device, class_to_idx, images, tfm, tta=2):
    """Runs prediction on a batch of PIL images."""
    if not images: return []
    batch = torch.stack([tfm(img) for img in images]).to(device)
    
    all_probs = []
    # Perform Test-Time Augmentation (original + horizontal flip)
    for i in range(max(1, tta)):
        current_batch = torch.flip(batch, dims=[-1]) if i > 0 else batch
        logits = model(current_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    
    avg_probs = np.mean(all_probs, axis=0)
    
    i_fake = class_to_idx.get("fake", 0)
    
    results = []
    for p in avg_probs:
        p_fake = float(p[i_fake])
        results.append({"p_fake": p_fake, "p_real": 1.0 - p_fake})
    return results

# ============ Grad-CAM (Heatmap Generation) ============
class GradCAM:
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model; self.model.eval()
        self.gradients = None; self.activations = None
        layer = dict(model.named_modules())[target_layer_name]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)
    def _save_activation(self, module, inp, out): self.activations = out.detach()
    def _save_gradient(self, module, grad_in, grad_out): self.gradients = grad_out[0].detach()
    def __call__(self, x, class_idx=None):
        x = x.requires_grad_(True); out = self.model(x)
        if class_idx is None: class_idx = out.argmax(1).item()
        score = out[:, class_idx]
        self.model.zero_grad(set_to_none=True); score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy(); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam

def overlay_cam(img_pil: Image.Image, cam: np.ndarray, alpha=0.45):
    img = np.array(img_pil).astype(np.float32) / 255.0
    cmap = plt.get_cmap("jet")(cam)[..., :3]
    out = np.clip((1 - alpha) * img + alpha * cmap, 0, 1)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))


# =========================
#      LOAD RESOURCES
# =========================
model, class_to_idx, device = load_model_and_meta(CKPT_PATH)
TFM = get_transforms()
i_fake = class_to_idx.get("fake", 0)

# =========================
#       SIDEBAR UI
# =========================
with st.sidebar:
    st.markdown('<div class="hero"><h1>🤖 AI Deepfake<br>Detector</h1></div>', unsafe_allow_html=True)
    
    with st.expander("About the AI Model", expanded=True):
        st.info(
            """
            This detector uses a **ResNet-50** architecture, a powerful deep learning model
            pre-trained on ImageNet. It was fine-tuned on a diverse dataset including:
            - **WildDeepfake (Images)**
            - **FaceForensics++ (Videos)**
            
            The model analyzes face crops to identify subtle artifacts indicative of manipulation.
            """
        )

# =========================
#        MAIN APP UI
# =========================

st.markdown('<div class="hero"><h1 class="title">Is It Real or AI?</h1><p class="subtitle">Upload an image or a short video to check for signs of deepfake manipulation.</p></div>', unsafe_allow_html=True)

st.markdown('<h3 class="section-header">How It Works</h3>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.info("**1. Upload Media:** Choose an image or video file containing a face.")
col2.info("**2. Detect Face:** The AI automatically finds and crops the largest face.")
col3.info("**3. Classify:** The cropped face is analyzed by the ResNet-50 model to predict if it's real or fake.")
st.write("---")

tabs = st.tabs(["📷 **Image Analysis**", "🎬 **Video Analysis**"])

# =========================
#       IMAGE TAB
# =========================
with tabs[0]:
    files = st.file_uploader("Choose image(s) to analyze", type=list(ALLOWED_IMG), accept_multiple_files=True)
    
    if st.button("Analyze Image(s)", type="primary", disabled=not files, use_container_width=True):
        processed_imgs, face_crops, results = [], [], []
        
        progress_bar = st.progress(0, text="Processing images...")
        for i, f in enumerate(files):
            try:
                img = Image.open(io.BytesIO(f.read())).convert("RGB"); processed_imgs.append(img)
                bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                face_bgr = detect_largest_face(bgr_img, conf_threshold=FACE_DETECTION_CONFIDENCE)
                if face_bgr is not None:
                    face_crops.append(Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)))
                else:
                    face_crops.append(img) # Use full image if no face is found
            except Exception as e:
                st.warning(f"Could not process file '{f.name}': {e}")
            progress_bar.progress((i + 1) / len(files), text=f"Processing '{f.name}'...")
        
        if face_crops:
            results = predict_pil_images(model, device, class_to_idx, face_crops, TFM, tta=2)
        
        progress_bar.empty()

        st.markdown(f'<h3 class="section-header">Analysis Results ({len(results)} image(s))</h3>', unsafe_allow_html=True)
        
        for i, (img, face_crop, res) in enumerate(zip(processed_imgs, face_crops, results)):
            p_fake = res['p_fake']
            is_fake = p_fake >= CLASSIFICATION_THRESHOLD
            
            verdict_class = "verdict-fake" if is_fake else "verdict-real"
            verdict_text = "AI-GENERATED (FAKE)" if is_fake else "LIKELY REAL"
            confidence = p_fake if is_fake else 1 - p_fake
            bar_class = "fake-bar" if is_fake else "real-bar"
            
            st.markdown(f'<span class="verdict {verdict_class}">{verdict_text}</span>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence-text">Confidence: {confidence*100:.1f}%</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="confbar"><div class="{bar_class}" style="width:{confidence*100:.1f}%"></div></div>', unsafe_allow_html=True)

            with st.spinner("Generating AI focus map..."):
                # --- THE FIX IS HERE ---
                # 1. Resize the face_crop to the model's input size (224x224)
                #    This is the image we will use for generating the heatmap and overlaying.
                resized_face_crop = face_crop.resize((IMG_SIZE, IMG_SIZE))
                
                # 2. Create the tensor for the model from the *resized* image
                x = TFM(resized_face_crop).unsqueeze(0).to(device)
                
                cam_generator = GradCAM(model)
                heatmap_data = cam_generator(x, class_idx=i_fake)
                
                # 3. Overlay the heatmap onto the *resized* image
                heatmap_overlay = overlay_cam(resized_face_crop, heatmap_data)

            col_img, col_heatmap = st.columns(2)
            # Display the original, uncropped image for context
            col_img.image(img, caption="Original Image", use_container_width=True)
            # Display the generated heatmap overlay
            col_heatmap.image(heatmap_overlay, caption="AI Focus Heatmap (on Face Crop)", use_container_width=True)

            with st.expander("🧠 Explain the AI's Focus"):
                st.markdown(
                    f"""
                    <div class="explanation-text">
                    <p>The heatmap on the right shows where the AI "looked" to make its decision. It is generated on the cropped face that was fed to the model.</p>
                    <ul>
                    <li><strong>"Hot" areas (Red/Yellow):</strong> These are the regions the AI found most suspicious.</li>
                    <li><strong>"Cool" areas (Blue):</strong> The AI largely ignored these parts.</li>
                    </ul>
                    <p>The model predicted <strong>{verdict_text.split(' ')[-1].replace(')', '')}</strong> because the features in the <strong>hot areas</strong> strongly matched patterns it learned from its training data.</p>
                    </div>
                    """, unsafe_allow_html=True
                )
            st.write("---")

# =========================
#       VIDEO TAB
# =========================
with tabs[1]:
    vf = st.file_uploader("Choose a short video file", type=list(ALLOWED_VID))
    
    if st.button("Analyze Video", type="primary", disabled=(vf is None), use_container_width=True):
        progress_bar = st.progress(0, text="Analyzing video, please wait...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vf.name).suffix) as tmp:
            tmp.write(vf.read()); tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                st.error("Could not open video file.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                stride = max(1, int(fps / 5)) # Aim for ~5 frames per second
                max_frames_to_process = 200
                
                face_crops, frame_indices = [], []
                
                for i in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if i % stride == 0:
                        face_bgr = detect_largest_face(frame, conf_threshold=FACE_DETECTION_CONFIDENCE)
                        # --- [IMPROVEMENT 1] Face Quality Control ---
                        if face_bgr is not None and face_bgr.shape[0] >= MIN_FACE_SIZE and face_bgr.shape[1] >= MIN_FACE_SIZE:
                            face_crops.append(Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)))
                            frame_indices.append(i)
                    
                    if len(face_crops) >= max_frames_to_process: break
                    
                    progress_text = f"Scanning video... Valid faces found: {len(face_crops)}"
                    progress_value = min(1.0, i / min(total_frames, max_frames_to_process * stride))
                    progress_bar.progress(progress_value, text=progress_text)

                cap.release()
                
                if not face_crops:
                    st.warning("Could not detect any high-quality faces in the video. Unable to make a prediction.")
                else:
                    preds = predict_pil_images(model, device, class_to_idx, face_crops, TFM, tta=2)
                    p_fakes = np.array([r["p_fake"] for r in preds])
                    
                    # --- [IMPROVEMENT 2] More Robust, Multi-Factor Verdict Logic ---
                    mean_p_fake = np.mean(p_fakes)
                    max_p_fake = np.max(p_fakes)
                    
                    highly_fake_frames_count = np.sum(p_fakes >= HIGH_CONF_THRESHOLD)
                    highly_fake_rate = highly_fake_frames_count / len(p_fakes)
                    
                    # The video is FAKE if its average score is high AND it contains some "smoking gun" frames.
                    is_fake = (mean_p_fake >= VIDEO_AVG_THRESHOLD) and (highly_fake_rate >= HIGHLY_FAKE_RATE_THRESHOLD)
                    
                    verdict_class = "verdict-fake" if is_fake else "verdict-real"
                    verdict_text = "AI-GENERATED (FAKE)" if is_fake else "LIKELY REAL"
                    
                    st.markdown(f'<h3 class="section-header">Overall Video Verdict</h3>', unsafe_allow_html=True)
                    st.markdown(f'<span class="verdict {verdict_class}">{verdict_text}</span>', unsafe_allow_html=True)

                    # --- [IMPROVEMENT 3] Verdict Rationale ---
                    with st.expander("Why was this verdict reached?", expanded=True):
                        if is_fake:
                            st.error(f"**Verdict: FAKE** because the average fake score ({mean_p_fake:.2f}) was above the threshold of {VIDEO_AVG_THRESHOLD}, AND the rate of highly confident fake frames ({highly_fake_rate:.1%}) was above the threshold of {HIGHLY_FAKE_RATE_THRESHOLD:.0%}.")
                        else:
                            st.success(f"**Verdict: REAL** because the video did not meet the criteria for a fake. The average fake score ({mean_p_fake:.2f}) was below {VIDEO_AVG_THRESHOLD} or the rate of highly confident fakes ({highly_fake_rate:.1%}) was below {HIGHLY_FAKE_RATE_THRESHOLD:.0%}.")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg. Fake Score", f"{mean_p_fake:.3f}", help="The average 'fake' probability across all analyzed frames.")
                    c2.metric("Max Fake Score", f"{max_p_fake:.3f}", help="The highest 'fake' probability found in a single frame.")
                    c3.metric("Confidently Fake Frames", f"{highly_fake_frames_count} / {len(face_crops)} ({highly_fake_rate:.1%})", help=f"The percentage of frames with a fake score > {HIGH_CONF_THRESHOLD:.0%}.")

                    st.markdown(f'<h3 class="section-header">Frame-by-Frame Analysis</h3>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.plot(frame_indices, p_fakes, color=PRIMARY, marker='o', linestyle='-', markersize=4, label='p(fake) Score')
                    ax.axhline(y=HIGH_CONF_THRESHOLD, color=WARN, linestyle='--', label=f'High Confidence Threshold ({HIGH_CONF_THRESHOLD})')
                    ax.set_title("Probability of 'Fake' Over Time")
                    ax.set_xlabel("Frame Number"); ax.set_ylabel("p(fake)"); ax.set_ylim(0, 1); ax.legend()
                    st.pyplot(fig, use_container_width=True)

                    st.markdown(f'<h3 class="section-header">Most Suspicious Frames</h3>', unsafe_allow_html=True)
                    st.caption("These are the frames that received the highest 'fake' scores from the AI.")
                    
                    top_indices = np.argsort(-p_fakes)[:min(9, len(p_fakes))]
                    if p_fakes[top_indices[0]] < CLASSIFICATION_THRESHOLD:
                        st.success("✅ No frames were flagged as suspicious. The video appears to be consistently real.")
                    else:
                        cols = st.columns(3)
                        for j, k in enumerate(top_indices):
                            if p_fakes[k] < CLASSIFICATION_THRESHOLD: continue
                            frame_num = frame_indices[k]; ts = f"{frame_num / fps:.2f}s" if fps > 0 else f"frame {frame_num}"
                            cols[j % 3].image(face_crops[k], caption=f"Time: {ts}\np(fake): {p_fakes[k]:.3f}", use_container_width=True)
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try: os.unlink(tmp_path)
                except Exception as e: print(f"Error deleting temp file {tmp_path}: {e}")
            progress_bar.empty()