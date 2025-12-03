# Denoising Diffusion Probabilistic Models 🖼️

A PyTorch implementation of **Denoising Diffusion Probabilistic Models (DDPM)** for unconditional image generation. This project demonstrates how diffusion models progressively denoise random noise into coherent images through a learned reverse diffusion process.

## Overview 📋 

This implementation provides a complete training pipeline for generating images using diffusion models. The model learns to reverse a gradual noising process, enabling it to generate high-quality images from pure Gaussian noise.

### Key Features ✨ 

- **UNet Architecture** with self-attention mechanisms
- **Training with checkpointing** - Resume training from saved checkpoints
- **Configurable diffusion parameters** - Customize noise schedules and timesteps
- **TensorBoard logging** - Track training progress in real-time
- **Image generation** - Sample new images during and after training
- **Google Colab compatible** - Train on free GPU resources

## Architecture 🏗️ 

The implementation consists of several key components:

### Core Modules

- **`Diffusion`** - Manages the forward and reverse diffusion processes
  - Noise scheduling with linear beta schedule
  - Forward process: adds noise to images
  - Reverse process: denoises images step-by-step

- **`UNet`** - The backbone neural network
  - Encoder-decoder architecture with skip connections
  - Self-attention layers at multiple resolutions
  - Time embedding for diffusion timestep conditioning

- **Supporting Modules**
  - `DoubleConv` - Double convolution blocks with GroupNorm
  - `Down` - Downsampling blocks with time embedding
  - `Up` - Upsampling blocks with skip connections
  - `SelfAttention` - Multi-head self-attention mechanism
  - `EMA` - Exponential Moving Average for model weights (optional)

## Getting Started 🚀

### Prerequisites

```bash
pip install torch torchvision matplotlib tqdm tensorboard pillow
```

### Dataset Structure 📂

Organize your images in the following structure:
```
dataset/
└── class_folder/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Training 🎮 

The notebook `Diffusion_Models_PyTorch_Implementation_YT.ipynb` contains the complete training pipeline.

#### Configuration Parameters

Edit the `launch()` function to customize training:

```python
args.run_name = "DDPM_Unconditional"  # Experiment name
args.epochs = 500                      # Total training epochs
args.batch_size = 8                    # Batch size
args.image_size = 64                   # Image resolution (64x64)
args.dataset_path = r"path/to/dataset" # Path to your dataset
args.lr = 3e-4                         # Learning rate
```

#### Training Features

- ✅ **Automatic checkpointing** - Training resumes from the last saved epoch
- ✅ **Progress tracking** - MSE loss logged to TensorBoard
- ✅ **Sample generation** - Images generated after each epoch
- ✅ **Model saving** - Checkpoints include model, optimizer state, and epoch number

### Sampling/Inference 🎯

Once trained, the model can generate new images by sampling from random noise:

```python
# Load trained model
model = UNet().to(device)
checkpoint = torch.load("models/DDPM_Unconditional/ckpt.pt", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Generate images
diffusion = Diffusion(img_size=64, device=device)
sampled_images = diffusion.sample(model, n=16)
plot_images(sampled_images)
```

## Training Process 📊

1. **Forward Diffusion**: Random noise is gradually added to training images over T timesteps
2. **Model Training**: UNet learns to predict and remove the noise at each timestep
3. **Reverse Diffusion**: Start from pure noise and iteratively denoise to generate images

The model uses:
- **Loss Function**: MSE between predicted and actual noise
- **Optimizer**: AdamW with learning rate 3e-4
- **Noise Schedule**: Linear schedule from β_start=1e-4 to β_end=0.02
- **Timesteps**: 1000 diffusion steps

## Output Structure 📁

```
models/
└── DDPM_Unconditional/
    └── ckpt.pt              # Model checkpoint

results/
└── DDPM_Unconditional/
    ├── 0.jpg                # Samples from epoch 0
    ├── 1.jpg                # Samples from epoch 1
    └── ...

runs/
└── DDPM_Unconditional/      # TensorBoard logs
```

## Troubleshooting 🔧

### PyTorch 2.6+ Checkpoint Loading

If you encounter `UnpicklingError` when loading checkpoints, add `weights_only=False`:

```python
checkpoint = torch.load(model_path, weights_only=False)
```

> ⚠️ **Note**: Only use `weights_only=False` with checkpoints from trusted sources.

### CUDA Out of Memory

- Reduce `batch_size` in `launch()` configuration
- Reduce `image_size` (e.g., from 64 to 32)
- Use gradient checkpointing (requires modification)

### Dataset Not Found

Ensure your dataset path is correctly set and the directory structure matches the expected format with subdirectories containing images.

## References 📚

* [Denoising Diffusion Probabilistic Models — Ho et al. (2020)](https://arxiv.org/pdf/2006.11239)
* [High-Resolution Image Synthesis with Latent Diffusion Models (LDM) — Rombach et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
* [Progressive Distillation for Fast Sampling — Salimans & Ho (2022)](https://arxiv.org/abs/2202.00512)
* [DPM-Solver / DPM-Solver++ (Fast ODE Solvers) — Lu et al. (2022)](https://arxiv.org/abs/2206.00927)
* [Structural Pruning for Diffusion Models (Diff-Pruning) — Fang et al. (2023)](https://arxiv.org/abs/2305.10924)
* [Denoising Diffusion Implicit Models (DDIM) — Song et al. (2020)](https://arxiv.org/pdf/2010.02502)
* [Improved DDPMs — Nichol & Dhariwal (2021)](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)
* [Elucidating the Design Space of Diffusion Models (EDM) — Karras et al. (2022)](https://arxiv.org/abs/2206.00364)
* [Denoising Student / Single-step Distillation — Luhman & Luhman (2021)](https://arxiv.org/abs/2101.02388)
* [Post-training Quantization for Diffusion (PTQ4DiT / Q-DiT / PQD, 2023–2025)](https://openreview.net/forum?id=NLmAGkN6nn)

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments ⭐

- PyTorch team for the deep learning framework
- Authors of the DDPM paper for the groundbreaking research
- Google Colab for providing free GPU resources

## Support & Contact 📞

- 🐛 **Issues**: [GitHub Issues](https://github.com/DewduSendanayake/Seasonal_Travel_Recommender/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/DewduSendanayake/Seasonal_Travel_Recommender/discussions)
- 👥 **Authors**: [DewduSendanayake](https://github.com/DewduSendanayake), [dulhara79](https://github.com/dulhara79), [UVINDUSEN](https://github.com/UVINDUSEN), [SENUVI20](https://github.com/SENUVI20)

---

**Happy Generating! 🎨✨**
