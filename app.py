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
    st.set_page_config(page_title="Image Caption Generator", page_icon="📷", layout="centered")

    st.title("📷 Image Caption Generator")
    st.write("Upload an image and get an AI-generated caption — runs 100% offline, no API key needed.")

    # Load model
    with st.spinner("Loading BLIP model (first time may take a minute)..."):
        processor, model = load_model()

    st.success("Model loaded!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

    # Optional: guided captioning
    use_prompt = st.checkbox("Use a guided prompt (optional)")
    prompt_text = ""
    if use_prompt:
        prompt_text = st.text_input(
            "Start the caption with...",
            placeholder="e.g. 'a photo of' or 'this image shows'",
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Caption", type="primary"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(image, processor, model, prompt_text)

            st.subheader("Caption")
            st.info(caption)

            st.write("---")
            st.write("💡 *Tip: Try the guided prompt option to steer the caption style.*")


if __name__ == "__main__":
    main()
