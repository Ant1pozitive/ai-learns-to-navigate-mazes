# AI Learns to Navigate in Mazes

Deep Q-Network project for learning to navigate mazes with high-quality visualization:
- Procedural maze generation.
- Custom Gym-like Maze environment.
- DQN with optional PER, dueling network, GPU support.
- Egocentric / grid / vector observations.
- Smooth final video export (MP4 / GIF) with thinking overlay.
- Streamlit UI for interactive demos.

## Quick start

1. Create venv and install:
```bash
python -m venv .venv
source .venv/bin/activate   # windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run Streamlit UI:
```bash
streamlit run main.py -- --web
```

OR

Run CLI training:
```bash
python main.py --size 21 --episodes 300
```

Artifacts (GIF/MP4, rewards plot, model) saved to outputs/.

ToDo:
  * advanced settings in web
  * smoothing training video
  * adaptive frames_per_step
  * write readme
  * files structure
