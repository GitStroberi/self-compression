# Agentic Workflow Prompt: Self-Compressing Neural Networks (SCNN) Project

**Role:** Senior Computer Vision Engineer & Model Optimization Specialist  
**System Environment:** Windows 10/11, AMD Radeon RX 9070 XT (ROCm), Conda env: `selfcomp`  
**Objective:** Architect and implement a maintainable, extensible library for "Self-Compressing Neural Networks" (SCNN) [arXiv:2301.13142].

## 1. Project Scope & Philosophy
*   **Core Goal:** Implement differentiable bit-width quantization where the model learns its own optimal size during training.
*   **Simplicity & Maintainability:** Avoid over-engineering. Stick to standard PyTorch inheritance patterns. The directory structure must be modular to support adding new backbones (YOLO, MobileNet) later without refactoring core logic.
*   **Hardware Constraint:** All training must run on an AMD GPU via ROCm. Ensure code uses `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` and avoids any CUDA-specific (NVIDIA-only) kernels unless a portable ROCm alternative is confirmed.

## 2. Technical Specification

### A. The Custom Layer (`SCNNConv2d`)
You must implement a custom convolution layer that replaces `nn.Conv2d`.
*   **Reference:** [benearnthof/self_compressing](https://github.com) (GitHub) - Use this for the quantization logic structure but **modify the clamping mechanism** as defined below.
*   **Learnable Parameter:** A scalar parameter `theta` (or `b`) representing the bit-width for the layer.
*   **Novelty (Critical):** Implement but do NOT use the standard hard clamp  (as a possible fallback or grounds for comparison). Implement this **smooth soft-clamp** function to improve gradient flow and convergence speed:
    \[\text{out} = 2^{b-1} \cdot \frac{\frac{x}{2^{b-1}}}{1 + \frac{\vert{}x\vert{}}{2^{b-1}}}\]
    *Where \(x = 2^{-e} \cdot w\) (scaled weights) and \(b\) is the current bit-width.*

### B. Loss Function
Implement the compound loss function:
\[\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \cdot \mathcal{L}_{compression}\]
*   **\(\mathcal{L}_{task}\)**: Standard CrossEntropy (for classification) or Object Detection loss (for YOLO).
*   **\(\mathcal{L}_{compression}\)**: A penalty term derived from the sum of bit-widths across all layers, encouraging the model to shrink \(b\) where possible.

## 3. Implementation Roadmap

### Phase 1: Core Library Structure
Create a modular package structure:
```text
self_compression/
├── layers/
│   ├── __init__.py
│   └── scnn_conv.py      # The custom layer with Smooth Soft-Clamp
├── models/
│   ├── __init__.py
│   ├── resnet_scnn.py    # ResNet20 implementation using SCNN layers
│   └── factory.py        # Helper to swap standard Conv2d with SCNNConv2d
├── utils/
│   └── loss.py           # SelfCompressionLoss
├── runs/                 # Timestamped training runs
└── train.py              # Main training loop
```

### Phase 2: Proof of Concept (PoC)
*   **Target Model:** ResNet20
*   **Dataset:** CIFAR-10
*   **Execution:** Run the training loop on the RX 9070 XT.
*   **Output required:** Into a `runs` folder save timestamped folders that include periodic checkpoints of the models during training, along with relevant metrics at the end (plot out accuracy and model size, like benearnthof/self_compressing)

### Phase 3: Extensibility Hooks
*   Write the code in `models/factory.py` such that applying this to `torchvision.models.resnet18` or `yolov5` later is as simple as:
    ```python
    model = torchvision.models.resnet18(pretrained=True)
    scnn_model = convert_to_scnn(model) # Automates layer replacement
    ```

## 4. Immediate Action Items (Agent Instructions)
1.  **Environment Check:** verify `torch.cuda.is_available()` returns True on the `selfcomp` environment.
2.  **Scaffold:** Create the directory structure above.
3.  **Code:** Write `scnn_conv.py` implementing the specific smooth soft-clamp formula provided.
4.  **Test:** Create a unit test `tests/test_layer.py` that passes a random tensor through `SCNNConv2d` and verifies the output shape and gradient availability.


