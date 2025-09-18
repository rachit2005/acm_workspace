import streamlit as st
from torchvision import transforms
from PIL import Image
import torch
from acm_submission import Model_Skin

# Set page config
st.set_page_config(page_title="Skin Cancer Detection", layout="wide")

# Title
st.title("🩺 Skin Cancer Detection Web App")

# Layout: Two columns
col1, col2 = st.columns([1, 2])  # make preview column wider

img = None
tensor_img = None

with col1:
    uploaded_file = st.file_uploader(
        "📤 Upload a skin image for analysis",
        type=["jpg", "jpeg", "png"],
        help="Upload an image in JPG, JPEG, or PNG format."
    )

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image: {e}")

with col2:
    if uploaded_file is not None and img is not None:
        st.image(img, caption='✅ Uploaded Image Preview', use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tensor_img = transform(img)
        tensor_img = tensor_img.unsqueeze(0)
        st.write(f"Tensor shape: {tensor_img.shape}")
    else:
        st.warning("⚠️ Please upload an image file to continue.")

# Model loading and prediction
if tensor_img is not None:
    model = Model_Skin(128, 7, 64)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(tensor_img)
        pred_class = torch.argmax(output, dim=1).item()
        class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        st.success(f"You have been diagnosed as: **{class_names[pred_class]}**")
        st.write(f"Model raw output: {output}")
else:
    st.info("Upload an image to get a diagnosis.")
