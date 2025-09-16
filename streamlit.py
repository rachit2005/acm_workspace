import streamlit as st

# Set page config
st.set_page_config(page_title="Skin Cancer Detection", layout="wide")

# Title
st.title("🩺 Skin Cancer Detection Web App")

# Layout: Two columns
col1, col2 = st.columns([1, 2])  # make preview column wider

with col1:
    uploaded_file = st.file_uploader(
        "📤 Upload a skin image for analysis",
        type=["jpg", "jpeg", "png"],
        help="Upload an image in JPG, JPEG, or PNG format."
    )

with col2:
    if uploaded_file is not None:
        st.image(uploaded_file, caption='✅ Uploaded Image Preview', use_container_width=True)
    else:
        st.warning("⚠️ Please upload an image file to continue.")

# Divider
st.markdown("""---""")

# Footer with fixed position
st.markdown(
    """
    <style>
    .bottom-text {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 10px;
        background-color: #f0f2f6;
        border-top: 1px solid #ccc;
        font-size: 14px;
        color: #333;
    }
    </style>
    <div class="bottom-text">
        ⚡ This is a demo web app for <b>Skin Cancer Detection</b> using Streamlit.<br>
        👨‍💻 Created by <b>Rachit Rastogi</b>.
    </div>
    """,
    unsafe_allow_html=True
)
