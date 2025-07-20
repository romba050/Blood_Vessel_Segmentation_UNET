"""
Blood Vessel Segmentation Streamlit Application

A web interface for retinal blood vessel segmentation using UNET architecture.
Based on segRetino by Srijarko Roy (https://github.com/srijarkoroy/segRetino)
Licensed under MIT License.

Features:
- Interactive image segmentation
- Real-time comparison with ground truth
- Performance metrics visualization
- Educational content about neural networks
"""

import streamlit as st
from PIL import Image
import os
import time
import numpy as np
from comparison_utils import (
    load_and_preprocess_mask, 
    calculate_segmentation_metrics, 
    create_comparison_visualization, 
    get_manual_annotation_path
)

# use this for titles and icons in multipage apps, see:
# https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app
st.set_page_config(
    page_title="Blood Vessel Segmentation",
    page_icon="👁",
)

st.title("👁 Retinal Vessel Segmentation")

# st.info("💡 **New to UNET?** Check out the **UNET Model Info** page in the sidebar to learn about the neural network architecture used for this segmentation!")

from segRetino.inference import SegRetino

st.write("Select a fundus image or upload your own to get the blood vessel segmentation.")
# Choice between predefined images or upload
image_source = st.radio(
    "Choose image source:",
    ["Select from predefined images", "Upload your own image (512×512 RGB)"]
)

input_path = None
image_number = None

if image_source == "Select from predefined images":
    # Dropdown for image selection
    image_options = {f"Input Image {i}": f"{i:02d}_test_0.png" for i in range(1, 21)}
    
    selected_image = st.selectbox("Choose an input image:", list(image_options.keys()))
    image_filename = image_options[selected_image]
    image_number = image_filename.split('_')[0]  # Extract number for output naming
    input_path = f'./segRetino/DRIVE_augmented/test/image/{image_filename}'
    
    # Check if input file exists
    if not os.path.exists(input_path):
        st.error(f"Input file {input_path} not found!")
        input_path = None

else:  # Upload your own image
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a fundus image for blood vessel segmentation"
    )
    
    # Check if we have a previous upload that completed segmentation
    if 'last_upload_info' in st.session_state:
        upload_info = st.session_state.last_upload_info
        # Check if the segmentation results exist for this upload
        upload_output = f'./segRetino/results/output/my_output_{upload_info["image_number"]}.png'
        upload_blend = f'./segRetino/results/blend/my_blend_{upload_info["image_number"]}.png'
        
        if os.path.exists(upload_output) and os.path.exists(upload_blend) and os.path.exists(upload_info['input_path']):
            input_path = upload_info['input_path']
            image_number = upload_info['image_number']
    
    if uploaded_file is not None:
        # Only create new timestamp if this is a truly new upload
        if ('last_upload_info' not in st.session_state or 
            st.session_state.last_upload_info.get('filename') != uploaded_file.name):
            
            # Save uploaded file temporarily
            upload_dir = './segRetino/results/input/uploads'
            os.makedirs(upload_dir, exist_ok=True)
            
            input_path = os.path.join(upload_dir, uploaded_file.name)
            
            # Save uploaded file
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Use timestamp or file name for unique output naming
            timestamp = int(time.time())
            image_number = f"upload_{timestamp}"
            
            # Store in session state to persist across reruns
            st.session_state.last_upload_info = {
                'input_path': input_path,
                'image_number': image_number,
                'filename': uploaded_file.name
            }

# Process image if we have a valid input path
if input_path is not None:
    # Load the input image
    img = Image.open(input_path)

    # Display images
    cols = st.columns(4)
        
    with cols[0]:
        st.write("Input image")
        st.image(img)
    
    # Load manual annotation for comparison (only for predefined images)
    manual_annotation_path = None
    if image_source == "Select from predefined images" and image_number:
        manual_annotation_path = get_manual_annotation_path(image_number)
        if os.path.exists(manual_annotation_path):
            with cols[1]:
                st.write("Manual annotation")
                manual_img = Image.open(manual_annotation_path)
                st.image(manual_img)
    
    # Construct output paths
    if image_number is not None:
        output_path = f'./segRetino/results/output/my_output_{image_number}.png'
        blend_path = f'./segRetino/results/blend/my_blend_{image_number}.png'
        
        # Ensure output directories exist
        os.makedirs('./segRetino/results/output', exist_ok=True)
        os.makedirs('./segRetino/results/blend', exist_ok=True)
        
        # Run inference button
        if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
            start_time = time.time()
            
            # Create progress display
            progress_bar = st.progress(0)
            time_display = st.empty()
            status_display = st.empty()
            
            try:
                # Step 1: Initialize model
                status_display.info("🔄 Initializing segmentation model...")
                progress_bar.progress(20)
                time_display.text(f"Elapsed: {time.time() - start_time:.1f}s")
                
                seg_model = SegRetino(input_path)
                
                # Step 2: Run inference (main processing)
                status_display.info("🔄 Processing image and running segmentation...")
                progress_bar.progress(50)
                time_display.text(f"Elapsed: {time.time() - start_time:.1f}s")
                
                seg_model.inference(
                    set_weight_dir='unet.pth', 
                    path=output_path, 
                    blend_path=blend_path
                )
                
                # Step 3: Complete
                progress_bar.progress(100)
                elapsed_time = time.time() - start_time
                time_display.success(f"✅ Completed in {elapsed_time:.1f} seconds")
                status_display.empty()  # Clear the processing message
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                time_display.error(f"❌ Error after {elapsed_time:.1f} seconds")
                status_display.error(f"Error during segmentation: {str(e)}")
                progress_bar.empty()
        
        # Display results (outside button click so they persist)
        # Determine column indices based on whether manual annotation exists
        seg_col_idx = 2 if manual_annotation_path and os.path.exists(manual_annotation_path) else 1
        blend_col_idx = 3 if manual_annotation_path and os.path.exists(manual_annotation_path) else 2
        
        with cols[seg_col_idx]:
            st.write("Model segmentation")
            if os.path.exists(output_path):
                seg_result = Image.open(output_path)
                st.image(seg_result)
            else:
                st.write("Run segmentation to see results")
        
        with cols[blend_col_idx]:
            st.write("Segmentation blend")
            if os.path.exists(blend_path):
                blend = Image.open(blend_path)
                st.image(blend)
            else:
                st.write("Run segmentation to see results")
        
        # Download buttons (persist after segmentation)
        if os.path.exists(output_path) and os.path.exists(blend_path):
            st.markdown("### 📥 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Segmentation",
                        data=file,
                        file_name=f"segmentation_{image_number}.png",
                        mime="image/png"
                    )
            
            with col2:
                with open(blend_path, "rb") as file:
                    st.download_button(
                        label="Download Blend",
                        data=file,
                        file_name=f"blend_{image_number}.png",
                        mime="image/png"
                    )
                    
        # Comparison analysis (only if manual annotation exists and segmentation is done)
        if (manual_annotation_path and os.path.exists(manual_annotation_path) and 
            os.path.exists(output_path)):
            
            st.subheader("📊 Segmentation Comparison Analysis")
            
            try:
                # Load and preprocess images
                predicted_mask = load_and_preprocess_mask(output_path)
                manual_mask = load_and_preprocess_mask(manual_annotation_path)
                
                # Calculate metrics
                metrics = calculate_segmentation_metrics(predicted_mask, manual_mask)
                
                # Display metrics in columns
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Dice Coefficient", f"{metrics['dice_coefficient']:.3f}")
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                
                with metric_cols[1]:
                    st.metric("IoU (Jaccard)", f"{metrics['iou']:.3f}")
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                
                with metric_cols[2]:
                    st.metric("Recall (Sensitivity)", f"{metrics['recall']:.3f}")
                    st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
                
                with metric_cols[3]:
                    st.metric("Specificity", f"{metrics['specificity']:.3f}")
                
                # Pixel-level statistics in separate row
                st.markdown("**Pixel-Level Statistics:**")
                pixel_cols = st.columns(3)
                
                with pixel_cols[0]:
                    st.metric("True Positive Pixels", f"{metrics['tp']:,}")
                
                with pixel_cols[1]:
                    st.metric("False Positive Pixels", f"{metrics['fp']:,}")
                
                with pixel_cols[2]:
                    st.metric("False Negative Pixels", f"{metrics['fn']:,}")
                
                # Create and display comparison visualization
                st.subheader("🎨 Visual Comparison")
                comparison_img = create_comparison_visualization(predicted_mask, manual_mask)
                
                # Convert to PIL Image for display
                comparison_pil = Image.fromarray(comparison_img)
                
                comparison_cols = st.columns(2)
                with comparison_cols[0]:
                    st.write("**Agreement/Disagreement Map**")
                    st.image(comparison_pil, caption="Green: Correct vessels, Red: False positives, Blue: Missed vessels, Black: Correct background")
                
                with comparison_cols[1]:
                    st.write("**Legend:**")
                    st.write("🟢 **Green**: True Positives (Correctly detected vessels)")
                    st.write("🔴 **Red**: False Positives (Incorrectly detected vessels)")
                    st.write("🔵 **Blue**: False Negatives (Missed vessels)")
                    st.write("⚫ **Black**: True Negatives (Correctly detected background)")
                    
                    # Summary statistics
                    total_pixels = predicted_mask.size
                    vessel_coverage = (metrics['tp'] + metrics['fn']) / total_pixels * 100
                    st.write(f"**Vessel coverage:** {vessel_coverage:.1f}% of image")
                    
            except Exception as e:
                st.error(f"Error in comparison analysis: {str(e)}")

else:
    st.info("Please select or upload an image to proceed.")

# Attribution footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>Based on <a href='https://github.com/srijarkoroy/segRetino' target='_blank'>segRetino</a> by Srijarko Roy (MIT License)<br>
    Theory from <a href='https://researchbank.swinburne.edu.au/file/fce08160-bebd-44ff-b445-6f3d84089ab2/1/2018-xianchneng-retina_blood_vessel.pdf' target='_blank'>Wang Xiancheng et al.</a> research paper</p>
</div>
""", unsafe_allow_html=True)