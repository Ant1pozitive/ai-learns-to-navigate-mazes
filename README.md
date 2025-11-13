# Reinforcement Learning Maze Explorer  
*A visual and interactive exploration of how AI learns to navigate complex environments.*

---

## Overview
This project demonstrates how a reinforcement learning (RL) agent learns to navigate through a maze environment.  
It combines **deep Q-learning**, **GPU acceleration**, and **real-time visualization** using Streamlit.

Users can observe how the agent improves over time - from random wandering to near-optimal navigation - with a clear “thinking overlay” that reveals the agent’s internal decision probabilities.

---

## Quick Start

### 1️. Clone the repository
```bash
git clone https://github.com/Ant1pozitive/ai-learns-to-navigate-mazes.git
cd ai-learns-to-navigate-mazes
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
⚡ For GPU acceleration, ensure that PyTorch with CUDA is installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Run the app
```bash
streamlit run RL.py -- --web
```

---

## Highlights

- Procedural maze generation (perfect maze by DFS).
- DQN agent with prioritized replay (PER), dueling network architecture and target network updates.
- Observation modalities: `vector` (classic DQN state representation), `grid` (vision-based observation with local field of view), and `egocentric` (local patch + goal heatmap).
- Smooth final video export (MP4/GIF) with annotated overlays: agent sprite, path gradient, per-action probabilities (thinking overlay) and Q-value bars.
- GPU-aware implementation (uses CUDA if available).
- Streamlit interactive UI for fast demos and real-time previews.

---

## Important notes & tips

* GPU: If you have NVIDIA GPU and CUDA installed, the code will automatically use cuda via PyTorch. For best performance, install a PyTorch build that matches your CUDA driver.
* ffmpeg: MP4 encoding requires ffmpeg for best results. The code attempts to use imageio-ffmpeg and imageio to write MP4. If you see MP4 failures, install ffmpeg and make sure it's available in PATH.
* Performance: The training loop avoids heavy I/O during the loop and delays the creation of high-resolution smooth frames until after training - this optimizes training speed while preserving final visual quality.
* Reproducibility: The code sets seeds for Python, NumPy, and PyTorch; however, exact bitwise reproducibility can still vary with different hardware and library versions.

---

## Configurable options (CLI / Streamlit)

* --size: Maze size (odd integer).
* --episodes: Number of training episodes.
* Observation type: selected in Streamlit - vector, grid, or egocentric (local patch centered on the agent).
* Other hyperparameters are defined in-code with sensible defaults. If you want to expose more knobs, modify Config inside main.py or use Advanced settings in web.

---

## Outputs

* outputs/dqn_model.pth - trained model weights.
* outputs/rewards.png - training reward curve.
* outputs/final_animation_30fps.mp4 - high-quality final video (if ffmpeg available).
* outputs/training_preview.gif - compact preview GIF created during training.
* outputs/train.log - logging output for debugging/analysis.
* outputs/frames/ - optional saved frames (if enabled).

---

## Recommended enhancements (next steps)

* Add curriculum learning for progressively larger mazes.
* Compare DQN variants (Double DQN, Rainbow components, distributional RL).
* Add unit/integration tests for environment and replay buffer.
* Add CI workflow (GitHub Actions) to run basic smoke tests.
