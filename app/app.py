"""PentAI goat and cattle disease screening, evaluation and Gemini guidance."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import os
import time

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from inference import load_config, prepare_image, read_probabilities, model_path_for
from edge_cases import assess_image, prediction_warnings  # PENTAI_EDGE_CASES_V1
from guidance import basic_guidance
from google import genai
from google.genai import errors as gemini_errors

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"
DISPLAY_LABELS = {"goatpox": "Goatpox", "healthy": "Healthy (model class)", "lumpy_skin": "Lumpy skin disease"}
MAX_BATCH_FILES = 20
MAX_BYTES_PER_FILE = 12 * 1024 * 1024
# Can be overridden using GEMINI_MODEL in Streamlit secrets/environment.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_RETRY_DELAYS = (1, 2, 4)  # seconds between attempts on temporary server overload

st.set_page_config(page_title="PentAI | Livestock Intelligence", page_icon="🐄", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
:root { --brand:#45dfb4; --ink:#e9f5f2; --muted:#9bb4ac; --line:#263d39; }
html, body, [data-testid="stAppViewContainer"] { font-family:'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse at 85% -15%, #173b36 0%, transparent 44%),
  linear-gradient(145deg,#0b1817 0%,#0e2220 55%,#0c1919 100%);
  color:var(--ink);
}
[data-testid="stSidebar"] { background:#0c1b1a; border-right:1px solid #27443e; }
[data-testid="stSidebar"] * { color:#dcece8; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .3rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: .62rem .55rem; border-radius: .7rem; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#1b3b34; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#214b3e; border:1px solid #428c70; }
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] { color:#a9cabc; }

.block-container { padding-top:6.5rem !important; padding-bottom:3rem; max-width:1420px; }
h1,h2,h3 { font-family:'Outfit', sans-serif !important; letter-spacing:-.025em; }
h1 { font-weight:750 !important; }
[data-testid="stMetric"] { border:1px solid #29473f; background:linear-gradient(130deg,#142c28,#10231f); border-radius:15px; padding:1rem 1.2rem; }
[data-testid="stMetricLabel"] { color:#a9c1b8; }
[data-testid="stMetricValue"] { color:#f1faf6; }
[data-testid="stFileUploader"] { background:#122723; border:1px dashed #47736a; border-radius:16px; padding:.5rem; }
[data-testid="stTabs"] button { font-weight:700; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#45dfb4; }
.stButton > button[kind="primary"] { background:#43d8aa; color:#08211a; border:0; border-radius:11px; font-weight:800; }
.stButton > button[kind="primary"]:hover { background:#6cebc4; color:#08211a; }
.stButton > button { border-radius:11px; }
[data-testid="stAlert"] { border-radius:13px; }
.hero { padding:2rem 2.2rem; border-radius:22px; border:1px solid #315b51;
 background:linear-gradient(112deg,rgba(30,77,65,.96),rgba(17,48,43,.92) 60%,rgba(15,43,39,.90)); margin-bottom:1.2rem; }
.eyebrow { letter-spacing:.15em; text-transform:uppercase; font-size:.72rem; font-weight:800; color:#7af0cb; }
.hero h1 { font-size:clamp(2rem,3.3vw,3.25rem); margin:.5rem 0; line-height:1.1; color:#f1faf7; }
.hero p { color:#c4dbd4; font-size:1.01rem; max-width:720px; margin:.3rem 0 1.1rem; }
.hero-tags span { display:inline-block; margin:.2rem .45rem .1rem 0; font-size:.75rem; padding:.35rem .7rem;
 border:1px solid #47796c; border-radius:99px; color:#c8f4e7; background:#173c32; }
.section-label {font-weight:750; font-family:'Outfit',sans-serif; font-size:1.5rem; padding:.4rem 0 .7rem; color:#ecf7f1;}
.panel { background:linear-gradient(150deg,#142b27,#102420); border:1px solid #2b4840; border-radius:17px; padding:1.2rem 1.4rem; margin-bottom:1rem; }
.panel-title { font-family:'Outfit',sans-serif; font-size:1.1rem; color:#f0f7f3; font-weight:700; margin:0 0 .45rem; }
.panel-desc { font-size:.89rem; color:#a9c4bb; line-height:1.6; }
.kpi-row {display:flex; flex-wrap:wrap; gap:.75rem; margin:1rem 0 .8rem;}
.kpi-chip {display:inline-flex; gap:.35rem; align-items:center; background:#173c32; border:1px solid #35695a; border-radius:9px; padding:.45rem .7rem; font-size:.8rem; color:#d8f6e9;}
.result { background:linear-gradient(120deg,#173c32,#19332d); border:1px solid #4aa987; border-radius:17px; padding:1.25rem 1.4rem; margin:.7rem 0; }
.result .eyebrow {font-size:.71rem;}
.result .big {font-family:'Outfit',sans-serif;font-size:2rem;font-weight:800;color:#f6fff9;line-height:1.25;margin:.35rem 0;}
.result .small {color:#a8d4c5;font-size:.82rem;}
.brand-name {font-family:'Outfit',sans-serif;font-size:1.65rem;font-weight:800;color:#ebfcf4;}
.brand-sub {color:#85bcab;font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;}
.info-dot {padding:.85rem 1rem;border-left:3px solid #53deb4; background:#122c26; color:#bbd6cd; border-radius:0 10px 10px 0; font-size:.88rem;}
footer {visibility:hidden;}
@media(max-width:700px) { .hero{padding:1.35rem;} .hero h1{font-size:2rem;} }

[data-testid="stImage"] img { max-height: 235px; object-fit: contain; border-radius: 12px; }
.thumb-caption { color:#c8dfd6; font-size:.77rem; line-height:1.3; overflow-wrap:anywhere; }
.summary-tag { color:#72e7c2; font-size:.9rem; font-weight:700; }


/* Keep all main page content below the Streamlit toolbar. */
[data-testid="stMainBlockContainer"],
[data-testid="stMain"] .block-container,
.stMain .block-container,
.block-container {
  padding-top: 6.5rem !important;
  scroll-padding-top: 6.5rem;
}
/* Page titles must not slide under the fixed header. */
.section-label {
  line-height: 1.45;
  padding-top: .45rem;
  padding-bottom: .5rem;
  white-space: normal;
  overflow-wrap: break-word;
}
.hero h1, .hero p, .panel-title, .panel-desc {
  white-space: normal;
  overflow-wrap: break-word;
}
@media (max-width: 700px) {
  [data-testid="stMainBlockContainer"],
  [data-testid="stMain"] .block-container,
  .stMain .block-container,
  .block-container {
    padding-top: 7rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }
  .hero { padding: 1.2rem !important; }
  .hero h1 { font-size: 1.65rem !important; line-height: 1.3 !important; }
  .section-label { font-size: 1.35rem !important; }
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource(show_spinner="Loading model...")
def get_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)

@st.cache_data
def get_metadata():
    return load_config()

@st.cache_data
def get_metrics(animal):
    with (EVALUATION_DIR / animal / "metrics.json").open(encoding="utf-8") as f:
        return json.load(f)

def pct(value):
    return f"{value:.1%}"

def infer_one(image, animal, config, labels):
    path = model_path_for(animal, config)
    if not path.is_file():
        raise FileNotFoundError(f"Missing model: {path.name}. Copy it into models/.")
    if config[animal]["preprocessing"] != "inside_model_raw_0_255":
        raise ValueError("Unexpected preprocessing setting; consult the model metadata")
    batch = prepare_image(image, config[animal]["image_size"])
    scores = get_model(str(path)).predict(batch, verbose=0)
    return read_probabilities(scores, labels)

def open_image(data):
    with Image.open(BytesIO(data)) as im:
        im.verify()
    with Image.open(BytesIO(data)) as im:
        return ImageOps.exif_transpose(im).convert("RGB")

def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key and key.strip():
            return key.strip()
    except (FileNotFoundError, KeyError, OSError):
        pass

    return os.getenv("GEMINI_API_KEY", "").strip()
    

def create_gemini_response(api_key, question, prediction_context, history):
    """Generate educational animal-health guidance without sending photos or filenames."""
    client = genai.Client(api_key=api_key)
    recent_history = "\n".join(
        f"{msg['role']}: {msg['content'][:1500]}"
        for msg in history[-6:]
        if msg.get('role') in ('user', 'assistant')
    )
    prompt = (
        "You are PentAI's educational livestock-health assistant, focused on goats "
        "and cattle. Give practical, concise, careful general guidance. "
        "The image classifier only distinguishes goatpox/healthy for goats and "
        "lumpy skin disease/healthy for cattle, and is not clinically validated. "
        "Treat predictions as unverified screening results, not diagnoses. "
        "Explain observation, safe separation, biosecurity, and when to contact "
        "a qualified veterinarian. Do not prescribe drugs, doses, or treatment "
        "plans; advise urgent veterinary care for severe or rapidly worsening signs. "
        "Do not claim to have examined any images. If the question is unrelated to "
        "animal health, politely steer back to livestock guidance.\n\n"
        f"Prediction summary: {prediction_context}\n"
        f"Recent conversation:\n{recent_history}\n"
        f"Current question: {question[:3000]}"
    )
    try:
        model_name = st.secrets.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    except (FileNotFoundError, KeyError, OSError):
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    model_name = str(model_name).strip() or DEFAULT_GEMINI_MODEL

    # Retry only a 503 server overload. Authentication, invalid models,
    # and exhausted quotas should surface immediately to the UI.
    for attempt in range(len(GEMINI_RETRY_DELAYS) + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if not response.text:
                raise ValueError("Gemini returned no text")
            return response.text
        except gemini_errors.ServerError as exc:
            status = getattr(exc, "code", None)
            if str(status) != "503" or attempt == len(GEMINI_RETRY_DELAYS):
                raise
            time.sleep(GEMINI_RETRY_DELAYS[attempt])


def result_context(rows):
    valid = [r for r in rows if r.get("status") == "ok"]
    if not valid:
        return "No successful model predictions available."
    # only aggregate results; never transmit photos or their filenames
    counts = {}
    for row in valid:
        key = row["prediction"]
        counts[key] = counts.get(key, 0) + 1
    return f"Species: {valid[0]['animal']}; {len(valid)} processed images; class counts: {counts}. Predictions are unverified research outputs."

config, label_map = get_metadata()
try:
    metrics = {animal: get_metrics(animal) for animal in ("goat", "cattle")}
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Evaluation metadata missing or invalid: {exc}")
    st.stop()

if "results" not in st.session_state:
    st.session_state.results = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_generation" not in st.session_state:
    st.session_state.upload_generation = 0


def navigate_to(destination):
    """Change sidebar radio selection before the next Streamlit render."""
    st.session_state.active_page = destination


def clear_image_analysis():
    """Reset uploaded photos and related results without affecting Gemini chat."""
    st.session_state.upload_generation += 1
    st.session_state.results = []
    st.session_state.pop("results_signature", None)

with st.sidebar:
    st.markdown('<div class="brand-name">◈ PentAI</div><div class="brand-sub">Livestock intelligence</div>', unsafe_allow_html=True)
    st.caption("Image-based animal health screening")
    st.divider()
    st.markdown("**NAVIGATION**")
    page = st.radio(
        "Navigate",
        ["Home", "Analyze images", "Frequently asked questions", "Guidance assistant", "Model insights", "Useful links", "About PentAI"],
        format_func=lambda name: {
            "Home": "🏠  Home",
            "Analyze images": "📷  Analyze images",
            "Frequently asked questions": "❓  FAQs",
            "Guidance assistant": "💬  Guidance assistant",
            "Model insights": "📊  Model insights",
            "Useful links": "🔗  Useful links",
            "About PentAI": "ℹ️  About PentAI",
        }[name],
        key="active_page",
        label_visibility="collapsed",
        help="Choose a section of PentAI. Your current image results remain available until you clear them or analyze another batch.",
    )
    st.divider()
    st.markdown("**SUPPORTED ANIMALS**")
    st.write("🐐 Goat · Goatpox / Healthy")
    st.write("🐄 Cattle · Lumpy skin / Healthy")
    st.divider()
    st.caption("Decision-support tool. Veterinary assessment is required for diagnosis and treatment.")

if page == "Home":
    st.markdown("""<div class="hero"><div class="eyebrow">PENTAI · LIVESTOCK DISEASE DETECTION</div>
      <h1>Understand animal health,<br>one photo at a time.</h1>
      <p>An image-based screening assistant for recognizing patterns associated with goatpox and cattle lumpy skin disease. Upload one photo or a batch, review the results, and learn what to do next with veterinary guidance.</p>
      <div class="hero-tags"><span>🐐 Goat</span><span>🐄 Cattle</span><span>📷 Batch-ready</span><span>💬 Guidance assistant</span></div></div>""", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="panel"><div class="panel-title">1 · Choose an animal</div><div class="panel-desc">Select goat or cattle. Each animal has a different trained classifier.</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">2 · Upload photos</div><div class="panel-desc">Upload up to 20 images at once. Preview compact thumbnails rather than scrolling through large photos.</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="panel"><div class="panel-title">3 · Review next steps</div><div class="panel-desc">See model predictions and ask for general advice about appropriate veterinary follow-up.</div></div>', unsafe_allow_html=True)
    st.info("This system supports only two classes per animal. A Healthy prediction does not rule out illness. Seek a veterinarian for diagnosis or treatment.")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Species", "2")
    m2.metric("Disease targets", "2")
    m3.metric("Goat test accuracy", pct(metrics["goat"]["accuracy"]),
              help="Evaluation accuracy on 151 held-out goat images, not real-world clinical accuracy.")
    m4.metric("Cattle test accuracy", pct(metrics["cattle"]["accuracy"]),
              help="Evaluation accuracy on 152 held-out cattle images, not real-world clinical accuracy.")
    st.caption("Model accuracy was measured on held-out evaluation images, not independent veterinary field validation.")
    home_analyze, home_faq = st.columns(2)
    with home_analyze:
        st.button("Start analyzing images →", type="primary", use_container_width=True, on_click=navigate_to, args=("Analyze images",), help="Open image upload and disease screening for goats or cattle.")
    with home_faq:
        st.button("Read common questions →", use_container_width=True, on_click=navigate_to, args=("Frequently asked questions",), help="Read answers about supported diseases, image quality, results and next steps.")

if page == "Analyze images":
    st.markdown('<div class="section-label">Analyze livestock images</div>', unsafe_allow_html=True)
    st.caption("Select an animal, upload up to 20 JPEG/PNG images, then click Analyze batch. Photos are processed locally by this app's trained classifiers.")
    species = st.segmented_control(
        "Animal type", ["Goat", "Cattle"], default=None,
        help="Required: choose Goat for goatpox screening or Cattle for lumpy skin disease screening. The app cannot confirm the animal species automatically.",
    )
    if species is None:
        st.info("Choose the animal shown in your photos. PentAI cannot verify species automatically.")
    else:
        st.caption("Confirm all uploaded photos show the selected species, ideally one animal per image. Other species and unknown diseases are not supported.")
    animal = (species or "").lower()
    with st.expander("Photo tips and supported images"):
        st.write("Use a clear, well-lit photo of the selected goat or cow, preferably showing one animal and any relevant skin area. Avoid heavy filters, extreme close-ups, distant or obstructed views.")
        st.caption("This app cannot automatically verify the animal, detect other diseases, or validate multiple animals in an image. Quality warnings are approximate, not guarantees.")
    uploads = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"batch_upload_{st.session_state.upload_generation}",
        help="Select JPEG or PNG photos, up to 20 images per batch and 12 MB per file. Use clear, well-lit photos of the animal selected above.",
    )
    st.button(
        "Clear uploads and results",
        icon=":material/delete_outline:",
        on_click=clear_image_analysis,
        disabled=not (uploads or st.session_state.results),
        help="Clear this batch: remove uploaded photos, thumbnails, predictions and CSV results. This does not delete saved models or your Gemini conversation.",
    )
    if len(uploads) > MAX_BATCH_FILES:
        st.warning(f"Select at most {MAX_BATCH_FILES} files per batch. Only the first {MAX_BATCH_FILES} will be processed.")
    selected = uploads[:MAX_BATCH_FILES]
    prepared = []
    if selected:
        st.markdown(f"**Image thumbnails · {len(selected)} selected**")
        cols = st.columns(4)
        for idx, uploaded in enumerate(selected):
            data = uploaded.getvalue()
            error = None
            im = None
            try:
                if len(data) > MAX_BYTES_PER_FILE:
                    raise ValueError("Image exceeds 12 MB")
                im = open_image(data)
                blocking_error, _ = assess_image(im)
                if blocking_error:
                    raise ValueError(blocking_error)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                error = str(exc)
            prepared.append((uploaded.name, data, im, error))
            with cols[idx % 4]:
                if im is not None:
                    thumb = im.copy()
                    thumb.thumbnail((200, 135))
                    st.image(thumb, width=155)
                else:
                    st.warning("Invalid image")
                st.caption(f"{idx+1}. {uploaded.name[:34]}")
    # A tooltip alone may not appear on disabled buttons or touch screens.
    # Keep the reason visible when uploads exist without a selected species.
    if selected and not species:
        st.warning("Please select Goat or Cattle above before analyzing these photos.")
    analyze_disabled = not (bool(selected) and bool(species))
    if not species:
        analyze_help = "First select Goat or Cattle, then upload photos to enable analysis."
    elif not selected:
        analyze_help = "Upload at least one JPEG or PNG photo to enable analysis."
    else:
        analyze_help = "Classify the selected images with the trained model for your chosen animal."
    if st.button(
        "✦ Analyze batch", type="primary", disabled=analyze_disabled,
        use_container_width=True, help=analyze_help,
    ):
        records = []
        seen_hashes = {}
        progress = st.progress(0, text="Processing images...")
        for index, (filename, data, im, error) in enumerate(prepared):
            row = {"image": filename, "animal": species, "prediction": "", "score": None, "status": "ok"}
            digest = hashlib.sha256(data).hexdigest()
            if error:
                row.update(status=f"error: {error}")
            elif digest in seen_hashes:
                row.update(status="duplicate", warnings=f"Duplicate of image {seen_hashes[digest]}: prediction skipped")
            else:
                seen_hashes[digest] = index + 1
                try:
                    _, quality_warnings = assess_image(im)
                    scores = infer_one(im, animal, config, label_map[animal])
                    best_label, score = scores[0]
                    row.update(prediction=DISPLAY_LABELS.get(best_label, best_label), score=round(float(score), 4))
                    for label, value in scores:
                        row[f"prob_{label}"] = round(float(value), 4)
                    notices = quality_warnings + prediction_warnings(best_label, float(score))
                    if notices:
                        row["warnings"] = " | ".join(notices)
                except Exception:
                    row["status"] = "error: Could not analyze this image; try another photo."
            records.append(row)
            progress.progress((index + 1) / len(prepared), text=f"Processed {index + 1}/{len(prepared)}")
        st.session_state.results = records
        st.session_state.results_signature = hashlib.sha256((animal + "|" + "|".join(hashlib.sha256(d).hexdigest() for _, d, _, _ in prepared)).encode()).hexdigest()
        st.session_state.messages = []
        progress.empty()
    rows = st.session_state.results
    current_signature = hashlib.sha256((animal + "|" + "|".join(hashlib.sha256(d).hexdigest() for _, d, _, _ in prepared)).encode()).hexdigest() if prepared else ""
    if rows and st.session_state.get("results_signature") != current_signature:
        st.info("The selected animal or uploaded files have changed. Click Analyze batch to refresh the results.")
        rows = []
    if rows:
        st.markdown('<div class="section-label">Batch results</div>', unsafe_allow_html=True)
        successful = [r for r in rows if r["status"] == "ok"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Images processed", len(rows))
        m2.metric("Successful predictions", len(successful))
        m3.metric("Errors / skipped", len(rows) - len(successful))
        warnings_count = sum(bool(r.get("warnings")) for r in rows)
        if warnings_count:
            st.warning(f"{warnings_count} image(s) have quality, duplicate, or interpretation notices. Read the warnings column before relying on results.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config={"score": st.column_config.NumberColumn("Model score", format="%.2f")})
        csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇ Download results (CSV)", csv_bytes, file_name="pentai_batch_predictions.csv",
            mime="text/csv",
            help="Save all batch results, including predictions, model scores, warnings, skipped duplicates and individual errors.",
        )
        st.caption("Scores are uncalibrated model outputs, not disease probabilities. Image-quality checks are heuristic and cannot prove the right species or detect unknown diseases. Photos remain in session memory; predictions and warnings appear in the CSV.")
        st.warning("Any animal showing illness, unusual skin lesions, fever, or rapid worsening should be assessed by a veterinarian. Do not use model results alone to decide treatment.")
        if successful:
            with st.expander("Inspect individual class scores"):
                choices = [i for i,r in enumerate(rows) if r["status"] == "ok"]
                idx = st.selectbox(
                    "Image", choices, format_func=lambda i: f"{i+1}. {rows[i]['image']}",
                    help="Select a successfully analyzed photo to review scores for all supported classes. Scores are not clinical probabilities.",
                )
                for label in label_map[rows[idx]["animal"].lower()]:
                    probability = rows[idx].get(f"prob_{label}", 0.0)
                    st.progress(float(probability), text=f"{DISPLAY_LABELS.get(label,label)} · {probability:.1%}")

if page == "Frequently asked questions":
    st.markdown('<div class="section-label">Frequently asked questions</div>', unsafe_allow_html=True)
    st.write("Quick answers to common questions about checking your goats and cattle with PentAI.")
    faqs = [
        (
            "What does PentAI do?",
            "PentAI checks a photo of a goat or cow for patterns linked to the conditions it was trained to recognize. "
            "It provides a screening result, not a confirmed veterinary diagnosis.",
        ),
        (
            "Which animals and diseases can I check?",
            "PentAI currently supports goats (goatpox or healthy model class) and cattle "
            "(lumpy skin disease or healthy model class). It does not assess chickens, other animals, or every possible illness.",
        ),
        (
            "How do I check an animal?",
            "Open Analyze images, choose Goat or Cattle, upload a clear JPEG or PNG photo, and select Analyze batch. "
            "You can upload up to 20 photos at a time.",
        ),
        (
            "What makes a good photo?",
            "Use a clear, well-lit photo of the animal or affected skin area. Keep the area in focus and avoid heavy shadows "
            "or filters. Poor-quality images can reduce the usefulness of a screening result.",
        ),
        (
            "What does a 'Healthy' result mean?",
            "It means the model selected its healthy class from the limited classes it knows. "
            "It does not prove the animal is healthy or rule out other diseases. Contact a veterinarian if the animal seems unwell.",
        ),
        (
            "What if PentAI identifies a possible disease?",
            "Treat it as a reason to seek veterinary advice, not a final diagnosis. Record symptoms and when they began, "
            "limit unnecessary animal movement and contact, and ask a veterinarian about examination and testing.",
        ),
        (
            "Can I use PentAI to choose medicine or treatment?",
            "No. Do not start medication or make treatment decisions based only on PentAI. "
            "A qualified veterinarian should decide what tests or treatment may be needed.",
        ),
        (
            "What if my animal is sick but the result says Healthy?",
            "Do not ignore the symptoms. PentAI recognizes only a few classes and may miss disease. "
            "Seek a veterinary assessment, especially if the animal has fever, skin lesions, poor appetite, or worsening symptoms.",
        ),
        (
            "Is PentAI free to use?",
            "The image-screening feature does not require you to buy an AI API key. "
            "The optional guidance assistant may sometimes be busy or unavailable, but basic guidance remains available.",
        ),
        (
            "Are my uploaded photos sent to the AI assistant?",
            "No. PentAI processes uploaded photos through its image-classification models. "
            "If you use the guidance assistant, it sends your chat text and a summary of prediction labels to the AI service, "
            "not the uploaded photos or filenames.",
        ),
    ]
    for question, answer in faqs:
        with st.expander(question):
            st.write(answer)
    st.divider()
    st.markdown("### Still have a question?")
    st.write("Your Livestock Health Assistant is ready to help you understand results, recognize warning signs, "
             "and prepare questions for a veterinarian.")
    st.button("Ask the Livestock Health Assistant →", type="primary", use_container_width=True, on_click=navigate_to, args=("Guidance assistant",), help="Open Gemini-powered guidance. Answers are educational and do not replace a veterinarian.")
    st.caption("Answers are general information and cannot replace a veterinarian's assessment.")

if page == "Guidance assistant":
    st.markdown('<div class="section-label">Guidance assistant</div>', unsafe_allow_html=True)
    st.write("Ask about screening results, symptoms, animal care, or preparing for a veterinary visit. This information is not a diagnosis or prescription.")
    key = get_api_key()
    st.markdown("### Your Livestock Health Assistant Is Ready")
    st.write("Ask questions about animal health, understand disease symptoms, and learn recommended next steps for your goats and cattle.")
    if not key:
        st.caption("Answers are currently based on built-in guidance. Live assistant responses may be unavailable.")
    successful_context = [r for r in st.session_state.results if r.get("status") == "ok"]
    if successful_context:
        with st.expander("Previous image analysis", expanded=False):
            st.caption(result_context(successful_context))
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Example: What should I do if a cow has suspicious skin lumps?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        if key:
            try:
                answer = create_gemini_response(key, question, result_context(st.session_state.results), st.session_state.messages[-7:-1])
            except Exception as exc:
                # Show a helpful status without revealing sensitive API response details.
                error_code = getattr(exc, "code", None)
                if str(error_code) == "503":
                    st.warning("Gemini is busy (503). Automatic retries were unsuccessful; showing offline guidance.")
                elif str(error_code) == "429":
                    st.warning("Gemini usage limit reached (429); showing offline guidance.")
                elif str(error_code) in ("401", "403"):
                    st.error("Gemini authorization failed. Check the API key and project access.")
                elif str(error_code) == "404":
                    st.error("Gemini model unavailable (404). Check GEMINI_MODEL in Streamlit secrets.")
                else:
                    st.error(f"Gemini request failed ({type(exc).__name__}); showing offline guidance.")
                answer = (
                    "The AI service is temporarily unavailable. "
                    + basic_guidance(question)
                )
        else:
            answer = "**Basic guidance (not AI):** " + basic_guidance(question)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    st.caption("Only your chat text and an aggregate of prediction labels are sent to Google Gemini when AI is enabled. Uploaded photos and filenames are not sent. Avoid entering private identifying details.")

if page == "Model insights":
    st.markdown('<div class="section-label">Model performance</div>', unsafe_allow_html=True)
    st.caption("These results summarize performance on held-out test images, separate from newly uploaded photos. They are not a substitute for veterinary field validation.")
    insight_view = st.radio(
        "View results", ["Overview", "Goat", "Cattle"], horizontal=True, key="insight_view",
        help="Overview compares models; Goat and Cattle show class-level evaluation and confusion matrices from held-out tests.",
    )

    if insight_view == "Overview":
        st.subheader("Evaluation overview")
        a, b = st.columns(2)
        with a:
            st.metric("Goat test accuracy", pct(metrics["goat"]["accuracy"]),
                      help="Share of 151 held-out goat test images classified correctly (148/151). Not a field diagnosis rate.")
            st.caption("151 test images · 148 correct")
        with b:
            st.metric("Cattle test accuracy", pct(metrics["cattle"]["accuracy"]),
                      help="Share of 152 held-out cattle test images classified correctly (130/152). Disease recall is different from accuracy.")
            st.caption("152 test images · 130 correct")
        st.markdown("#### Key findings")
        st.write("The goat model missed 3 of 60 goatpox images. The cattle model missed 18 of 48 lumpy skin disease images.")
        st.warning("Cattle lumpy skin disease recall is 62.5%. A Healthy result must not be used to rule out illness.")
        st.caption("Choose Goat or Cattle above for confusion matrices and detailed class-level scores.")
    else:
        animal = insight_view.lower()
        data = metrics[animal]
        st.subheader(("🐐 " if animal == "goat" else "🐄 ") + insight_view + " model")
        a, b, c = st.columns(3)
        a.metric("Accuracy", pct(data["accuracy"]),
                 help="Overall proportion of held-out test images classified correctly.")
        b.metric("Balanced accuracy", pct(data["balanced_accuracy"]),
                 help="Average recall across the two classes, giving both classes equal weight.")
        c.metric("Macro F1", pct(data["macro_f1"]),
                 help="Average of class-specific F1-scores; balances precision and recall equally across classes.")
        st.caption(f'{data["test_samples"]} held-out test images')
        st.markdown("#### Confusion matrix")
        matrix_path = EVALUATION_DIR / animal / "confusion_matrix.png"
        if matrix_path.exists():
            st.image(str(matrix_path), caption="Rows: actual class; columns: predicted class", use_container_width=True)
        else:
            st.info("Confusion matrix not available in the evaluation folder.")
        if animal == "goat":
            st.info("Goatpox: 57 of 60 detected; 3 incorrectly predicted healthy.")
        else:
            st.warning("Lumpy skin disease: 30 of 48 detected (62.5% recall); 18 incorrectly predicted healthy.")
        with st.expander("Detailed classification report"):
            report_path = EVALUATION_DIR / animal / "classification_report.csv"
            if report_path.exists():
                report = pd.read_csv(report_path)
                report = report.rename(columns={"Unnamed: 0": "Disease class"})
                if "Disease class" not in report.columns and len(report.columns):
                    report = report.rename(columns={report.columns[0]: "Disease class"})
                names = ["goatpox", "healthy"] if animal == "goat" else ["healthy", "lumpy_skin"]
                if "Disease class" in report.columns:
                    report = report[report["Disease class"].astype(str).isin(names)].copy()
                    report["Disease class"] = report["Disease class"].replace({"goatpox": "Goatpox", "healthy": "Healthy", "lumpy_skin": "Lumpy skin disease"})
                for col in ("precision", "recall", "f1-score"):
                    if col in report.columns:
                        report[col] = pd.to_numeric(report[col], errors="coerce").map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
                if "support" in report.columns:
                    report["support"] = pd.to_numeric(report["support"], errors="coerce").astype("Int64")
                report = report.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-score", "support": "Test images"})
                st.dataframe(report, hide_index=True, use_container_width=True)
            else:
                st.info("Classification report not available in the evaluation folder.")
        st.caption("These figures are evaluation statistics, not clinical validation or calibrated disease probabilities.")

if page == "Useful links":
    st.markdown('<div class="section-label">Useful veterinary resources in Kenya</div>', unsafe_allow_html=True)
    st.write("Access trusted Kenyan veterinary services, professional registers and livestock research information. These resources can support informed follow-up after a PentAI screening result.")
    st.info("For a sick animal, consult a licensed veterinary professional. PentAI does not diagnose disease or prescribe treatment.")
    resource_groups = [
        ("Veterinary professionals and services", [
            ("🩺 Kenya Veterinary Board", "Find information about registered veterinary practitioners and professional standards in Kenya.", "https://kenyavetboard.or.ke/en"),
            ("🔎 Kenya Veterinary Board — Registers", "Check official professional registration information before seeking veterinary services.", "https://kenyavetboard.or.ke/en/registers"),
            ("🏛️ Directorate of Veterinary Services", "Government veterinary services, including information on diagnostic and veterinary public-health services.", "https://dvs.ecitizen.go.ke/"),
        ]),
        ("Research and professional information", [
            ("🌱 Kenya Agricultural and Livestock Research Organization", "Kenyan livestock research, knowledge and agricultural innovations.", "https://kalrodri.org/"),
            ("👩‍⚕️ Kenya Veterinary Association", "Professional information and veterinary-sector initiatives in Kenya.", "https://www.kenyavetassociation.com/"),
        ]),
    ]
    for group_title, resources in resource_groups:
        st.subheader(group_title)
        for name, description, url in resources:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.write(description)
                st.link_button(
                    "Visit official website ↗", url, use_container_width=False,
                    help=f"Open the external website for {name}. You will leave the PentAI application.",
                )
    st.caption("External websites are operated by their respective organizations. Links were checked during the preparation of this page.")

if page == "About PentAI":
    st.markdown('<div class="section-label">About PentAI</div>', unsafe_allow_html=True)
    st.write("PentAI is a research-based livestock disease screening application developed to support early identification of visual patterns associated with goatpox in goats and lumpy skin disease in cattle. Using separate MobileNetV2 image-classification models, the platform enables users to analyze photographs, review model-generated screening results and access educational veterinary guidance. Its purpose is to demonstrate practical applications of machine learning in animal-health decision support, while emphasizing responsible interpretation and qualified veterinary follow-up.")
    st.markdown("**Prediction flow:** Choose species → upload image(s) → RGB conversion and resize to 224×224 → saved species-specific model (internal preprocessing) → class scores → result and optional guidance.")
    st.markdown("**Limitations:** Only two classes per species; no automatic species verification, multiple-animal localization, or unknown-disease detection. Quality checks and low-confidence alerts are heuristics only; scores are uncalibrated and dataset performance may not generalize to field conditions.")
    st.markdown("**Veterinary advice:** Seek qualified veterinary assessment for clinical signs, testing and decisions. Do not medicate animals based solely on app predictions.")
    st.caption("PentAI currently focuses on image-based screening for goatpox in goats and lumpy skin disease in cattle.")

st.divider()
st.caption("© PentAI · Livestock Disease Detection | Predictions support screening only; veterinary confirmation required.")
