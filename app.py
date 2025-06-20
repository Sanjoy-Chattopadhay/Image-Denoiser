import streamlit as st
import requests
import os.path
import torch
from PIL import Image
from utils import utils_model
from utils import utils_image as util
import numpy as np
import cv2

# Page configuration
st.set_page_config(page_title="Real Image Denoising", layout="wide")

# Title section
st.markdown("<h1 style='text-align: center;'>🧼 Real Image Denoising Demo</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Remove real-world noise from your images using deep learning.</p>", unsafe_allow_html=True)
st.write("---")

# Sidebar
st.sidebar.header("🔧 Upload and Settings")
upfile = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Main layout
if upfile is not None:
    img = Image.open(upfile).convert("RGB")
    img.save("test1.png")
    st.sidebar.success("Image uploaded successfully!")

    # Display columns
    col1, col2 = st.columns(2)

    # Original image
    with col1:
        st.subheader("📸 Original Noisy Image")
        st.image(img, width=400)

    # Load model
    model_name = 'team15_SAKDNNet'
    model_path = os.path.join('model_zoo', model_name + '.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.team15_SAKDNNet import SAKDNNet as net
    model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model.to(device)

    for _, param in model.named_parameters():
        param.requires_grad = False

    # Denoising with spinner message
    with st.spinner("🔄 Denoising in progress... Please hold on."):
        img_U = util.imread_uint("test1.png", n_channels=3)
        img_N = util.uint2tensor4(img_U).to(device)
        img_DN = model(img_N)
        img_DN = util.tensor2uint(img_DN)

    # Show result
    with col2:
        st.subheader("✨ Denoised Output")
        st.image(img_DN, width=400)

    st.success("✅ Denoising complete. Scroll to see results.")
else:
    st.info("📤 Please upload an image from the sidebar to begin.")


# with st.expander("🧠 About the Model"):
st.header("Model Information")
st.markdown("""
**SAKDNNet** (Self-Attention Kernel Denoising Network) is a deep CNN-based architecture designed for real-world image denoising.  
It integrates **convolutional blocks** and **transformer-based attention mechanisms** for both local and global feature learning.

- 📦 **STCB (Spatial Transformer-Convolutional Block)** combines local CNN features with global Self-Attention (SAST).
- 🔁 **SAST (Shifted Attention Spatial Transformer)** uses window-based multi-head attention with relative positional encoding, similar to Swin Transformers.
- 🧠 **DRFE (Denoising Residual Feedforward Encoder)** applies residual connections and LayerNorm to stabilize learning.
- 🧱 The overall architecture follows a **U-Net-like encoder-decoder** pattern, enabling multi-scale feature extraction and fusion.
- 💡 DropPath regularization and GELU activations are used for robustness and generalization.

The model is designed to handle noisy real-world photos where traditional Gaussian assumptions do not hold.
""")

