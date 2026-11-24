# BrandRestoreAI

**AI-powered 4× image super-resolution for low-resolution brand and creative assets.**

BrandRestoreAI is an image restoration application that enhances low-resolution product images, logos, advertisements and creative assets using deep learning. It uses a pretrained Swin2SR model for real-world image super-resolution and includes a custom CNN developed for experimentation and evaluation.

## Features

- 4× image upscaling using pretrained Swin2SR.
- Real-world image restoration.
- Image analysis and optional enhancement.
- Before-and-after comparison.
- Custom CNN training and evaluation.
- PSNR and SSIM quality metrics.
- Interactive Streamlit interface.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · Swin2SR · OpenCV · Streamlit · NumPy · Pillow

## Models

**Pretrained model:** [Swin2SR Real-World X4](https://huggingface.co/caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr)

Used for the application's primary 4× image restoration pipeline.

**Custom CNN:** A residual CNN with convolutional layers, residual blocks and PixelShuffle upsampling, trained using L1 loss and Adam optimizer.

## Dataset

The custom CNN experiment used selected images from:

| Dataset | Training | Validation |
|---|---:|---:|
| ABO | 1,800 | 200 |
| LLD | 450 | 50 |
| **Total** | **2,250** | **250** |

Images were prepared as 128×128 HR targets and 32×32 LR inputs for the ×4 training experiment.

## Experimental Results

Custom CNN compared with bicubic upscaling:

| Metric | Bicubic | CNN |
|---|---:|---:|
| PSNR | 22.7277 | 24.7688 |
| SSIM | 0.7708 | 0.8227 |

The custom CNN achieved improved PSNR and SSIM on the validation experiment.

## Installation

```bash
git clone https://github.com/AyeeshaShaik984/BrandRestoreAI.git
cd BrandRestoreAI
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
BrandRestoreAI/
├── app.py
├── restoration.py
├── requirements.txt
├── models/
├── training/
├── results/
└── datasets/  (excluded from GitHub)
```

## Limitations

- AI may generate details not present in the original.
- Text and logos may not be reconstructed perfectly.
- Experimental metrics apply to the custom CNN validation experiment, not the pretrained Swin2SR model.

## Future Work

- Improve text and logo restoration.
- Benchmark additional restoration models.
- Optimize inference speed.
- Explore domain-specific training.

## Acknowledgements

- [Swin2SR](https://huggingface.co/caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr)
- Amazon Berkeley Objects (ABO)
- Low-Level Image Dataset (LLD)
