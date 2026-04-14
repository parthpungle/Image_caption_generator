"""
Image Caption Generator (Offline)
Uses the BLIP model from Hugging Face to generate captions locally.
No API key needed. No cost. Fully offline after first download.
"""

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


@st.cache_resource
def load_model():
    """Load the BLIP model and processor (cached so it only loads once)."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


def generate_caption(image: Image.Image, processor, model, conditional_text: str = "") -> str:
    """Generate a caption for the given image."""
    if conditional_text:
        inputs = processor(image, conditional_text, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def main():
    st.set_page_config(
        page_title="Image Caption Generator",
        page_icon="📷",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # ── Custom CSS ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* Page background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
        }

        /* Hide default Streamlit header / footer */
        #MainMenu, footer, header { visibility: hidden; }

        /* Hero section */
        .hero {
            text-align: center;
            padding: 2.5rem 1rem 1.5rem;
        }
        .hero-icon {
            font-size: 3.5rem;
            line-height: 1;
        }
        .hero h1 {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0.4rem 0 0.2rem;
        }
        .hero p {
            color: #94a3b8;
            font-size: 1.05rem;
            margin: 0;
        }

        /* Badge pills */
        .badges {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin: 1rem 0 2rem;
        }
        .badge {
            background: rgba(167,139,250,0.15);
            border: 1px solid rgba(167,139,250,0.3);
            color: #c4b5fd;
            border-radius: 999px;
            padding: 0.25rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }

        /* Card containers */
        .card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(10px);
        }
        .card-title {
            color: #e2e8f0;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.75rem;
        }

        /* Caption result box */
        .caption-box {
            background: linear-gradient(135deg, rgba(167,139,250,0.12), rgba(96,165,250,0.12));
            border: 1px solid rgba(167,139,250,0.35);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            text-align: center;
        }
        .caption-text {
            color: #f1f5f9;
            font-size: 1.2rem;
            font-style: italic;
            font-weight: 500;
            line-height: 1.6;
            margin: 0;
        }

        /* Tip box */
        .tip-box {
            background: rgba(251,191,36,0.08);
            border-left: 3px solid #fbbf24;
            border-radius: 0 8px 8px 0;
            padding: 0.65rem 1rem;
            color: #fde68a;
            font-size: 0.88rem;
            margin-top: 1rem;
        }

        /* Streamlit widget label overrides */
        .stFileUploader label, .stCheckbox label, .stTextInput label {
            color: #cbd5e1 !important;
            font-weight: 500 !important;
        }

        /* Divider */
        hr { border-color: rgba(255,255,255,0.08) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero">
            <div class="hero-icon">📷</div>
            <h1>Image Caption Generator</h1>
            <p>AI-powered captions — fully offline, no API key required</p>
        </div>
        <div class="badges">
            <span class="badge">⚡ Offline</span>
            <span class="badge">🔒 Private</span>
            <span class="badge">🤗 BLIP Model</span>
            <span class="badge">🆓 Free</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model loading ────────────────────────────────────────────────────────
    with st.spinner("Loading BLIP model — first run may take a minute..."):
        processor, model = load_model()

    st.success("Model ready", icon="✅")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload card ──────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">📁 Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Guided prompt card ───────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">✏️ Guided Prompt (optional)</div>', unsafe_allow_html=True)
    use_prompt = st.toggle("Steer the caption with a starting phrase")
    prompt_text = ""
    if use_prompt:
        prompt_text = st.text_input(
            "Start the caption with…",
            placeholder="e.g. 'a photo of' or 'this image shows'",
            label_visibility="collapsed",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Preview + generate ───────────────────────────────────────────────────
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col_img, _ = st.columns([3, 1])
        with col_img:
            st.markdown('<div class="card"><div class="card-title">🖼️ Preview</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        btn_col, _ = st.columns([2, 3])
        with btn_col:
            generate = st.button("✨ Generate Caption", type="primary", use_container_width=True)

        if generate:
            with st.spinner("Generating caption…"):
                caption = generate_caption(image, processor, model, prompt_text)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">💬 Generated Caption</div>
                    <div class="caption-box">
                        <p class="caption-text">"{caption}"</p>
                    </div>
                    <div class="tip-box">
                        💡 <strong>Tip:</strong> Enable the guided prompt above to steer the caption style or focus.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
