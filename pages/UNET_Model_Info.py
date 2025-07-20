"""
UNET Model Information Page

Educational content about Deep Neural Networks, CNNs, and UNET architecture
for blood vessel segmentation applications.

Based on segRetino by Srijarko Roy (https://github.com/srijarkoroy/segRetino)
Theory from Wang Xiancheng et al. research paper.
Licensed under MIT License.
"""

import streamlit as st
from PIL import Image
import os

st.set_page_config(
    page_title="UNET Model Information",
    page_icon="📚",
)

st.title("📚 Understanding UNET for Blood Vessel Segmentation")

st.markdown("""
This page provides an overview of the neural network architectures used in medical image segmentation, 
culminating in the UNET architecture employed in this application for retinal blood vessel segmentation.
""")

# Deep Neural Networks (DNN)
st.header("Deep Neural Networks (DNN)")

if os.path.exists('./dnn.png'):
    dnn_img = Image.open('./dnn.png')
    st.image(dnn_img, caption="Deep Neural Network Architecture", use_container_width=True)

st.markdown("""
**Deep Neural Networks (DNN)** are the foundation of modern machine learning:

- **(a) Network Architecture**: Shows a fully connected neural network with:
  - **Input Layer**: Receives raw data (x₁, x₂, ..., xₙ)
  - **Hidden Layers**: Multiple layers of interconnected neurons that learn complex patterns
  - **Output Layer**: Produces the final prediction (ŷ)
  - **Connections**: Every neuron connects to every neuron in the next layer

- **(b) Activation Function**: The ReLU (Rectified Linear Unit) function:
  - **f(x) = x for x ≥ 0, 0 for x < 0**
  - Introduces non-linearity, allowing the network to learn complex relationships
  - Helps solve the vanishing gradient problem in deep networks

**Limitations for Image Processing:**
- Fully connected layers lose spatial information
- Massive number of parameters for high-resolution images
- Not translation-invariant (doesn't recognize patterns regardless of position)
""")

# Convolutional Neural Networks (CNN)
st.header("Convolutional Neural Networks (CNN)")

if os.path.exists('./cnn.jpeg'):
    cnn_img = Image.open('./cnn.jpeg')
    st.image(cnn_img, caption="Convolutional Neural Network Architecture", use_container_width=True)

st.markdown("""
**Convolutional Neural Networks (CNN)** revolutionized image processing:

**Key Components:**

1. **Convolutional Layers (Conv_1, Conv_2)**:
   - Use **5×5 kernels** to detect local features (edges, textures)
   - **Valid padding** maintains spatial relationships
   - Extract hierarchical features from simple to complex

2. **Max-Pooling Layers**:
   - **2×2 pooling** reduces spatial dimensions
   - Provides translation invariance and reduces computation
   - Retains the most important features

3. **Fully-Connected Layers (fc_3, fc_4)**:
   - **ReLU activation** for non-linearity
   - **Dropout** prevents overfitting
   - Final classification into multiple classes (0, 1, 2, ..., 9)

**Architecture Flow:**
- **Input**: 28×28×1 (e.g., handwritten digit)
- **Conv_1**: 24×24×n1 → **Max-Pool**: 12×12×n1
- **Conv_2**: 8×8×n2 → **Max-Pool**: 4×4×n2
- **Flattened** → **FC layers** → **Output**: n3 classes

**Advantages for Medical Imaging:**
- Preserves spatial relationships
- Reduces parameters compared to fully connected networks
- Learns translation-invariant features
""")

# UNET Architecture
st.header("UNET Architecture")

if os.path.exists('./unet.png'):
    unet_img = Image.open('./unet.png')
    st.image(unet_img, caption="UNET Architecture for Image Segmentation", use_container_width=True)

st.markdown("""
**UNET** is specifically designed for **pixel-level image segmentation**, making it ideal for medical applications like blood vessel segmentation:

**Architecture Overview:**

1. **Encoder (Contracting Path - Left Side)**:
   - **Downsampling**: 2×2 Max Pooling reduces spatial dimensions
   - **Feature Extraction**: 3×3 convolutions with Batch Normalization and ELU activation
   - **Progressive Reduction**: 8 → 16 → 32 → 64 channels
   - **Context Capture**: Learns "what" is in the image

2. **Decoder (Expansive Path - Right Side)**:
   - **Upsampling**: 2×2 Up-convolution increases spatial resolution
   - **Feature Refinement**: 3×3 convolutions restore fine details
   - **Progressive Expansion**: 64 → 32 → 16 → 8 channels
   - **Localization**: Learns "where" features are located

3. **Skip Connections (Horizontal Arrows)**:
   - **Element-wise Sum**: Combines encoder and decoder features at same resolution
   - **Information Preservation**: Retains fine-grained spatial details
   - **Gradient Flow**: Helps training of very deep networks

4. **Output**:
   - **Probability Map**: Each pixel gets a probability of being a blood vessel
   - **1×1 Convolution + Sigmoid**: Final activation for binary segmentation

**Why UNET Excels at Blood Vessel Segmentation:**

- **Precise Localization**: Skip connections preserve vessel boundaries
- **Multi-Scale Features**: Captures both thick and thin vessels
- **Context Awareness**: Understands vessel branching patterns
- **End-to-End Training**: Optimized specifically for segmentation tasks
- **Efficient**: Relatively few parameters compared to classification networks

**Medical Imaging Applications:**
- Retinal vessel segmentation (this application)
- Cell segmentation in microscopy
- Organ segmentation in CT/MRI scans
- Tumor detection and delineation
""")

# Technical Details
st.header("Technical Implementation")

st.markdown("""
**In This Application:**

- **Input**: 512×512 RGB fundus images (retinal photographs)
- **Training**: DRIVE dataset with manual annotations
- **Data Augmentation**: Rotation, flipping, scaling for robustness
- **Loss Function**: Dice coefficient and IoU for handling class imbalance
- **Output**: Binary mask highlighting blood vessels

**Performance Metrics:**
- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index for segmentation quality
- **Precision/Recall**: Balance between detecting vessels and avoiding false positives
- **Visual Comparison**: Color-coded agreement/disagreement maps

The UNET's ability to maintain spatial precision while capturing global context makes it the gold standard for medical image segmentation tasks.
""")

# Footer
st.markdown("---")
st.markdown("*Navigate back to the main page to try blood vessel segmentation with the trained UNET model.*")

# Attribution footer
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2em;'>
    <p>Based on <a href='https://github.com/srijarkoroy/segRetino' target='_blank'>segRetino</a> by Srijarko Roy (MIT License)<br>
    Theory from <a href='https://researchbank.swinburne.edu.au/file/fce08160-bebd-44ff-b445-6f3d84089ab2/1/2018-xianchneng-retina_blood_vessel.pdf' target='_blank'>Wang Xiancheng et al.</a> research paper</p>
</div>
""", unsafe_allow_html=True)