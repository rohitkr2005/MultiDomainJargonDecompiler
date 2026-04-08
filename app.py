"""
Multi-Domain Jargon Decompiler
Simplifies complex legal and medical documents for everyday users.
Built with Streamlit, PyTesseract, OpenCV, and OpenRouter AI.
"""

import streamlit as st
import cv2
import numpy as np
import pytesseract
import json
import re
import os
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Jargon Decompiler",
    page_icon="🔍",
    layout="wide",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('[https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap)');

    /* Global */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.35);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #c4b5fd;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }

    /* Clause Cards */
    .clause-card {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.15s ease;
    }
    .clause-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
    }
    .clause-card.predatory {
        background: linear-gradient(135deg, #fef2f2, #fff1f2);
        border-color: #ef4444;
    }
    .clause-card.warning {
        background: linear-gradient(135deg, #fffbeb, #fef9c3);
        border-color: #f59e0b;
    }
    .clause-card.normal {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-color: #22c55e;
    }

    .clause-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .severity-badge {
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-predatory { background: #ef4444; color: white; }
    .badge-warning   { background: #f59e0b; color: white; }
    .badge-normal    { background: #22c55e; color: white; }

    .clause-original {
        font-size: 0.88rem;
        color: #374151;
        line-height: 1.6;
        margin-bottom: 0.6rem;
        font-style: italic;
    }
    .clause-translation {
        font-size: 0.95rem;
        color: #111827;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Medical Cards */
    .bio-card {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.15s ease;
    }
    .bio-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
    }
    .bio-card.status-high, .bio-card.status-low {
        background: linear-gradient(135deg, #fef2f2, #fff1f2);
        border-color: #ef4444;
    }
    .bio-card.status-normal {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-color: #22c55e;
    }
    .bio-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.3rem;
    }
    .bio-status-badge {
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .bio-badge-high, .bio-badge-low { background: #ef4444; color: white; }
    .bio-badge-normal { background: #22c55e; color: white; }
    .bio-values {
        font-size: 0.9rem;
        color: #374151;
        line-height: 1.7;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e7ff !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# OCR Processing
# ──────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to grayscale and apply binary thresholding."""
    img_array = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_text_ocr(pil_image: Image.Image) -> str:
    """Preprocess image and extract text using PyTesseract."""
    cleaned = preprocess_image(pil_image)
    text = pytesseract.image_to_string(Image.fromarray(cleaned))
    return text.strip()


# ──────────────────────────────────────────────
# OpenRouter AI Helpers
# ──────────────────────────────────────────────
def _parse_json_from_response(text: str) -> list | dict:
    """Extract and parse JSON from the AI's response text."""
    # Try to find JSON in code fences first
    # Using `{3} to match triple backticks so it doesn't break the UI!
    fence_match = re.search(r"`{3}(?:json)?\s*\n?([\s\S]*?)`{3}", text)
    if fence_match:
        return json.loads(fence_match.group(1).strip())
    # Otherwise try the whole text
    return json.loads(text.strip())


def call_openrouter_legal(text: str, api_key: str) -> list[dict]:
    """Send OCR text to OpenRouter for legal clause analysis."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:8501", 
            "X-Title": "Jargon Decompiler App"
        }
    )

    prompt = f"""You are a legal document analyst. Analyze the following text extracted via OCR from a legal document.
Extract every distinct clause or meaningful provision from the text.

For EACH clause, return a JSON object with these exact keys:
- "original_clause": the exact clause text from the document
- "layman_translation": a plain-English explanation a non-lawyer can understand
- "severity": one of "Normal", "Warning", or "Predatory"
  - "Normal" = standard boilerplate, nothing concerning
  - "Warning" = worth paying attention to, slightly unusual
  - "Predatory" = potentially exploitative or harmful to the signer

Return ONLY a valid JSON array. No explanations outside the JSON.

OCR TEXT:
\"\"\"
{text}
\"\"\"
"""
    response = client.chat.completions.create(
        model="openrouter/free", 
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json_from_response(response.choices[0].message.content)


def call_openrouter_medical(text: str, api_key: str) -> list[dict]:
    """Send OCR text to OpenRouter for medical biomarker analysis."""
    client = OpenAI(
        base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)",
        api_key=api_key,
    )

    prompt = f"""You are a medical report analyst. Analyze the following text extracted via OCR from a medical lab report.
Extract every biomarker / lab test result mentioned.

For EACH biomarker, return a JSON object with these exact keys:
- "biomarker": the name of the biomarker or lab test
- "observed_value": the value found in the report (include units if present)
- "reference_range": the normal/reference range (include units if present)
- "status": one of "Normal", "High", or "Low"
- "definition": a brief plain-English explanation of what this biomarker measures and why it matters

Return ONLY a valid JSON array. No explanations outside the JSON.

OCR TEXT:
\"\"\"
{text}
\"\"\"
"""
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_json_from_response(response.choices[0].message.content)


# ──────────────────────────────────────────────
# Rendering Functions
# ──────────────────────────────────────────────
def render_legal_results(results: list[dict]):
    """Render color-coded legal clause cards."""
    st.markdown("### 📜 Clause-by-Clause Breakdown")

    for i, clause in enumerate(results):
        severity = clause.get("severity", "Normal")
        sev_lower = severity.lower()
        badge_class = f"badge-{sev_lower}"
        card_class = sev_lower

        icon_map = {"predatory": "🚨", "warning": "⚠️", "normal": "✅"}
        icon = icon_map.get(sev_lower, "📄")

        html = f"""
        <div class="clause-card {card_class}">
            <div class="clause-header">
                <span style="font-size:1.2rem">{icon}</span>
                <span class="severity-badge {badge_class}">{severity}</span>
            </div>
            <div class="clause-original">"{clause.get('original_clause', 'N/A')}"</div>
            <div class="clause-translation">💬 {clause.get('layman_translation', 'N/A')}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


def render_medical_results(results: list[dict]):
    """Render medical biomarker cards with status indicators."""
    st.markdown("### 🩺 Biomarker Analysis")

    for i, bio in enumerate(results):
        status = bio.get("status", "Normal")
        status_lower = status.lower()

        badge_class = f"bio-badge-{status_lower}"
        card_class = f"status-{status_lower}"

        icon_map = {"high": "🔴", "low": "🔵", "normal": "🟢"}
        icon = icon_map.get(status_lower, "⚪")

        definition = bio.get("definition", "")

        html = f"""
        <div class="bio-card {card_class}">
            <div class="bio-name">{icon} {bio.get('biomarker', 'N/A')}</div>
            <span class="bio-status-badge {badge_class}">{status}</span>
            <div class="bio-values">
                <strong>Observed:</strong> {bio.get('observed_value', 'N/A')}<br>
                <strong>Reference Range:</strong> {bio.get('reference_range', 'N/A')}
            </div>
            {"<div style='margin-top:0.5rem; font-size:0.85rem; color:#6b7280;'>ℹ️ " + definition + "</div>" if definition else ""}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PDF Helper
# ──────────────────────────────────────────────
def pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    """Convert PDF bytes to a list of PIL Images."""
    try:
        from pdf2image import convert_from_bytes
        
        # 1. Define your local Windows path
        local_poppler = r'C:\poppler-25.12.0\Library\bin'  # <-- Update this to your actual Poppler path
        
        # 2. Check if that local path actually exists
        if os.path.exists(local_poppler):
            return convert_from_bytes(pdf_bytes, poppler_path=local_poppler)
        else:
            # 3. If it doesn't exist (like on a cloud server), just run it normally
            # The server will find Poppler via the 'packages.txt' installation
            return convert_from_bytes(pdf_bytes)
            
    except Exception as e:
        st.error(f"❌ Failed to convert PDF: {e}")
        return []


# ──────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Multi-Domain Jargon Decompiler</h1>
        <p>Upload legal contracts or medical reports — get instant plain-English explanations powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load API key from environment ──
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🔍 About")
        st.markdown("""
        This tool helps you understand complex
        documents written in legal or medical jargon.
        """)
        st.divider()
        st.markdown("### 📖 How It Works")
        st.markdown("""
        1. **Upload** a document image or PDF
        2. **OCR** extracts text automatically
        3. **AI** analyzes and simplifies jargon
        4. **Results** are color-coded for quick reading
        """)
        st.divider()
        st.markdown("### 🎨 Color Guide")
        st.markdown("""
        **Legal Documents:**
        - 🟢 **Normal** — Standard clause
        - 🟡 **Warning** — Pay attention
        - 🔴 **Predatory** — Potentially harmful

        **Medical Reports:**
        - 🟢 **Normal** — Within range
        - 🔴 **High** — Above normal
        - 🔵 **Low** — Below normal
        """)
        st.divider()
        st.markdown(
            "<p style='font-size:0.75rem; color:#94a3b8; text-align:center;'>"
            "Built with Streamlit · OpenRouter AI · PyTesseract</p>",
            unsafe_allow_html=True,
        )

    # ── Tabs ──
    tab_legal, tab_medical = st.tabs(["📑 Legal Document", "🧬 Medical Report"])

    for tab, doc_type in [(tab_legal, "legal"), (tab_medical, "medical")]:
        with tab:
            label = "Legal Document" if doc_type == "legal" else "Medical Report"

            uploaded_file = st.file_uploader(
                f"Upload your {label}",
                type=["png", "jpg", "jpeg", "pdf"],
                key=f"uploader_{doc_type}",
                help=f"Supported formats: PNG, JPG, PDF",
            )

            if uploaded_file is None:
                st.info(f"👆 Upload a {label.lower()} to get started.")
                continue

            # ── Load images from file ──
            images: list[Image.Image] = []
            if uploaded_file.type == "application/pdf":
                with st.spinner("📄 Converting PDF pages to images..."):
                    images = pdf_to_images(uploaded_file.read())
                if not images:
                    continue
            else:
                images = [Image.open(uploaded_file)]

            # Show uploaded image preview
            with st.expander("🖼️ Uploaded Document Preview", expanded=False):
                for idx, img in enumerate(images):
                    st.image(img, caption=f"Page {idx + 1}", use_container_width=True)

            # ── OCR ──
            raw_texts = []
            with st.spinner("🔎 Extracting text via OCR..."):
                try:
                    for img in images:
                        raw_texts.append(extract_text_ocr(img))
                except Exception as e:
                    st.error(f"❌ OCR failed: {e}")
                    st.info("💡 Make sure Tesseract OCR is installed on your system.")
                    continue

            combined_text = "\n\n".join(raw_texts)

            if not combined_text.strip():
                st.warning("⚠️ OCR did not extract any text. The image may be too blurry or not contain readable text.")
                continue

            with st.expander("📝 Raw Extracted Text", expanded=False):
                st.code(combined_text, language=None)

            # ── AI Analysis ──
            if not api_key:
                st.error("⚙️ AI analysis is currently unavailable. The server administrator needs to configure the OPENROUTER_API_KEY in the environment variables.")
                continue

            if st.button(f"🚀 Analyze {label}", key=f"analyze_{doc_type}", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your document..."):
                    try:
                        if doc_type == "legal":
                            results = call_openrouter_legal(combined_text, api_key)
                        else:
                            results = call_openrouter_medical(combined_text, api_key)

                        if not results:
                            st.warning("The AI returned no results. Try a clearer document image.")
                            continue

                    except json.JSONDecodeError:
                        st.error("❌ Failed to parse AI response. The model returned invalid JSON. Please try again.")
                        continue
                    except Exception as e:
                        st.error(f"❌ AI analysis failed: {e}")
                        continue

                st.divider()

                # ── Render results ──
                if doc_type == "legal":
                    # Summary counters
                    severities = [c.get("severity", "Normal") for c in results]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🟢 Normal", severities.count("Normal"))
                    col2.metric("🟡 Warning", severities.count("Warning"))
                    col3.metric("🔴 Predatory", severities.count("Predatory"))
                    st.divider()
                    render_legal_results(results)
                else:
                    # Summary counters
                    statuses = [b.get("status", "Normal") for b in results]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🟢 Normal", statuses.count("Normal"))
                    col2.metric("🔴 High", statuses.count("High"))
                    col3.metric("🔵 Low", statuses.count("Low"))
                    st.divider()
                    render_medical_results(results)

if __name__ == "__main__":
    main()