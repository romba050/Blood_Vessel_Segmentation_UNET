# Blood Vessel Segmentation UNET

A Streamlit web application for retinal blood vessel segmentation using UNET deep learning architecture.

Hosted under https://basile-rommes.com/BVS

![UNET Architecture](unet.png)

## Features

- **Interactive Web Interface**: User-friendly Streamlit application
- **Real-time Segmentation**: Process fundus images with trained UNET model
- **Comparison Analysis**: Compare model predictions with manual annotations
- **Educational Content**: Learn about DNN, CNN, and UNET architectures
- **Performance Metrics**: Comprehensive evaluation with Dice, IoU, Precision, Recall
- **Visual Comparison**: Color-coded agreement/disagreement maps

## Model Performance

The UNET model provides:
- Precise vessel boundary detection
- Multi-scale feature extraction
- Skip connections for fine detail preservation
- Binary segmentation with probability maps

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Blood_Vessel_Segmentation_UNET
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Run the Streamlit application:
```bash
uv run streamlit run Vessel_Segmentation.py
```

## Usage

1. **Select Image Source**: Choose from 20 predefined DRIVE dataset images or upload your own
2. **Run Segmentation**: Click the "Run Segmentation" button
3. **View Results**: See the original image, segmentation mask, and blend overlay
4. **Compare with Ground Truth**: For predefined images, view detailed comparison metrics
5. **Learn More**: Visit the "UNET Model Info" page to understand the architecture

## Project Structure

```
Blood_Vessel_Segmentation_UNET/
├── Vessel_Segmentation.py          # Main Streamlit application
├── comparison_utils.py              # Segmentation comparison utilities
├── pages/
│   └── UNET_Model_Info.py          # Educational page about neural networks
├── segRetino/                       # Core segmentation module
│   ├── inference.py                # Model inference and processing
│   ├── config/
│   │   └── weights_download.json   # Google Drive model weights config
│   ├── segretino/                  # UNET model architecture
│   │   ├── unet.py                 # UNET implementation
│   │   ├── loss.py                 # Loss functions
│   │   └── training_utils.py       # Training utilities
│   ├── DRIVE_augmented/            # DRIVE dataset (test images)
│   │   └── test/
│   │       ├── image/              # Input fundus images (20 images)
│   │       └── 1st_manual/         # Manual annotations for comparison
│   ├── weights/                    # Model weights (auto-downloaded)
│   │   └── unet.pth               # Trained UNET model (119MB)
│   └── results/                    # Generated outputs (ignored by git)
│       ├── output/                 # Segmentation masks
│       ├── blend/                  # Overlay visualizations
│       └── input/uploads/          # User uploaded images
├── pyproject.toml                  # UV project configuration
├── uv.lock                         # Dependency lock file
├── LICENSE                         # MIT license
├── README.md                       # Project documentation
└── assets/                         # Educational images
    ├── unet.png                    # UNET architecture diagram
    ├── cnn.jpeg                    # CNN explanation
    └── dnn.png                     # Deep neural network diagram
```

## Dataset

This project uses the **DRIVE (Digital Retinal Images for Vessel Extraction)** dataset:
- 20 test images (512x512 RGB)
- Manual annotations by medical experts
- Augmented versions for robust training

## Technical Details

- **Architecture**: UNET with skip connections
- **Input**: 512x512 RGB fundus images
- **Output**: Binary vessel segmentation masks
- **Framework**: PyTorch + Streamlit
- **Metrics**: Dice coefficient, IoU, Precision, Recall, F1-score

## References

### Original Implementation
This project is based on the excellent work by **Srijarko Roy**:
- **GitHub**: [segRetino](https://github.com/srijarkoroy/segRetino)
- **License**: MIT License
- **Original Author**: Srijarko Roy (2021)

### Scientific Background
The theoretical foundation is based on:
- **Paper**: "Retinal Blood Vessel Segmentation Using Convolutional Neural Networks"
- **Authors**: Wang Xiancheng et al.
- **URL**: [Research Paper](https://researchbank.swinburne.edu.au/file/fce08160-bebd-44ff-b445-6f3d84089ab2/1/2018-xianchneng-retina_blood_vessel.pdf)

## Enhancements Added

This implementation extends the original work with:

- **Interactive Streamlit Web Interface**
- **Real-time Comparison with Ground Truth**
- **Educational Content about Neural Networks**
- **Enhanced Visualization and Metrics**
- **Improved User Experience**
- **Comprehensive Documentation**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The project maintains the original MIT license from the segRetino implementation while adding attribution for enhancements.

## Acknowledgments

- **Srijarko Roy** for the original segRetino implementation
- **Wang Xiancheng et al.** for the theoretical framework
- **DRIVE Dataset** creators for providing the medical imaging data
- **PyTorch** and **Streamlit** communities for excellent frameworks

## Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share your results

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

*This project demonstrates the power of UNET architecture for medical image segmentation with an accessible web interface for education and research.*