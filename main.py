import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import random
import time
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable, Dict, Any
import io
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import imageio
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# Logging / outputs
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, "train.log")
logger = logging.getLogger("ai_maze")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
ch.setLevel(logging.INFO)
logger.addHandler(fh); logger.addHandler(ch)


# Configuration
@dataclass
class Config:
    maze_size: int = 21
    episodes: int = 300
    obs_type: str = 'vector'  # 'vector','grid','egocentric'

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 64
    lr: float = 5e-4
    gamma: float = 0.99
    seed: int = 42
    target_update_steps: int = 1000
    replay_capacity: int = 30000
    per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000
    min_replay_size: int = 1000
    max_grad_norm: float = 10.0

    render_interval: int = 10
    train_frames_per_step: int = 2
    final_frames_per_step: int = 12
    preview_width: int = 800
    final_width: int = 1024
    save_frames_during_training: bool = False
    save_final_frames: bool = True
    save_mp4: bool = True

    sprite_agent: bool = True
    show_q_bars: bool = True

    output_dir: str = OUTPUT_DIR
    tensorboard_logdir: Optional[str] = None

    max_steps_multiplier: int = 3

    epsilon_start: float = 1.0
    epsilon_final: float = 0.02
    epsilon_decay_frames: int = 40000

    @classmethod
    def from_args(cls, args):
        cfg = cls(maze_size=args.size, episodes=args.episodes)
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.tensorboard_logdir = os.path.join(cfg.output_dir, "tb")
        return cfg


# Utilities
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def softmax_safe_np(x: Optional[np.ndarray], temp: float = 1.0) -> Optional[np.ndarray]:
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    t = max(1e-6, float(temp))
    x = x / t
    x = x - x.max()
    exps = np.exp(x)
    s = exps.sum()
    if s <= 0:
        return np.ones_like(x, dtype=np.float32) / float(x.size)
    return (exps / s).astype(np.float32)


# Thoughts generator
def generate_thoughts(q_values: Optional[np.ndarray],
                      recent_cache: deque,
                      rng: random.Random,
                      temperature: float = 1.0,
                      max_attempts: int = 12) -> str:
    if q_values is None:
        base = "No action probabilities available yet."
        recent_cache.append(base)
        return base

    probs = softmax_safe_np(np.asarray(q_values, dtype=np.float32), temp=temperature)
    if probs is None:
        base = "No action probabilities available yet."
        recent_cache.append(base)
        return base

    actions = ["right", "down", "left", "up"]
    idx_sorted = list(np.argsort(probs)[::-1])
    top_idx = int(idx_sorted[0])
    top_p = float(probs[top_idx])
    second_idx = int(idx_sorted[1])
    second_p = float(probs[second_idx])
    entropy = -np.sum([p * math.log(p+1e-12) for p in probs])
    prob_std = float(np.std(probs))

    openers = ["Thinking out loud,", "Hmm—", "I estimate", "My thought:", "I lean toward", "Tentative plan:", "Observation:", "Quick read:"]
    confidences = ["I have strong confidence", "I'm somewhat confident", "I'm uncertain", "It's ambiguous", "The model favors", "The model slightly prefers"]
    connectors = ["—", ":", ";", ", but", ", however", "."]
    endings = ["", "Let's try that.", "Worth trying.", "I'll explore that.", "I'll give it a shot."]
    alt_templates = ["Alternative is {alt} ({alt_p}%).", "Second choice: {alt} ({alt_p}%).", "Also considering {alt} ({alt_p}%)."]

    templates = [
        lambda: f"{rng.choice(openers)} go {actions[top_idx]} ({int(100*top_p)}%).",
        lambda: f"{rng.choice(openers)} {actions[top_idx]} ({int(100*top_p)}%) over {actions[second_idx]} ({int(100*second_p)}%).",
        lambda: f"{rng.choice(openers)} I'm inclined to go {actions[top_idx]} — {int(100*top_p)}% confidence{rng.choice(connectors)} {rng.choice(endings)}",
        lambda: f"{rng.choice(openers)} probabilities are close; {actions[top_idx]} ~{int(100*top_p)}%, {actions[second_idx]} ~{int(100*second_p)}%.",
        lambda: f"{rng.choice(confidences)} to choose {actions[top_idx]} ({int(100*top_p)}%). {rng.choice(endings)}",
        lambda: f"{rng.choice(openers)} results are ambiguous: {', '.join([f'{actions[i]} {int(100*probs[i])}%' for i in range(len(actions))])}.",
        lambda: f"I'd try {actions[top_idx]} (p={top_p:.2f}). {rng.choice(alt_templates).format(alt=actions[second_idx], alt_p=int(100*second_p))}"
    ]

    adjectives = ["likely", "plausible", "promising", "risky", "reasonable", "uncertain"]
    hedges = ["maybe", "perhaps", "might", "could be", "I guess", "I suspect"]

    candidate = None
    for attempt in range(max_attempts):
        tmpl = rng.choice(templates)
        s = tmpl()
        if rng.random() < 0.25:
            s = s.rstrip('.') + f", {rng.choice(hedges)}."
        if rng.random() < 0.15:
            s = s + f" [{rng.choice(adjectives)}]"
        s = s.replace("  ", " ").strip()

        if s in recent_cache:
            s2 = s + (" " + rng.choice(["Let's test it.", "I'll check that.", "Confirming..."]))
            if s2 not in recent_cache:
                s = s2
            else:
                continue
        candidate = s
        break

    if candidate is None:
        candidate = f"Prefer {actions[top_idx]} ({int(100*top_p)}%)."

    recent_cache.append(candidate)
    return candidate


# Maze generator
class MazeGenerator:
    @staticmethod
    def generate(size:int, seed:Optional[int]=None) -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]]:
        if size % 2 == 0:
            size += 1
        if seed is not None:
            random.seed(seed)
        maze = np.ones((size, size), dtype=np.uint8)
        def dfs(x,y):
            maze[x,y] = 0
            dirs = [(0,2),(2,0),(0,-2),(-2,0)]
            random.shuffle(dirs)
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if 0 < nx < size-1 and 0 < ny < size-1 and maze[nx,ny] == 1:
                    maze[x+dx//2, y+dy//2] = 0
                    dfs(nx,ny)
        start = (1,1); goal = (size-2, size-2)
        dfs(start[0], start[1])
        maze[start] = 0; maze[goal] = 0
        maze[0,1] = 0; maze[size-1,size-2] = 0
        return maze, start, goal


# Environment + render
class MazeEnv:
    def __init__(self, maze:np.ndarray, start:Tuple[int,int], goal:Tuple[int,int]):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.state = start
        self.actions = [(0,1),(1,0),(0,-1),(-1,0)]
        self.size = maze.shape[0]

    def reset(self)->Tuple[int,int]:
        self.state = self.start
        return self.state

    def step(self, action:int)->Tuple[Tuple[int,int], float, bool]:
        dx,dy = self.actions[action]
        nx, ny = self.state[0]+dx, self.state[1]+dy
        prev = self.state
        moved = False
        if 0 <= nx < self.size and 0 <= ny < self.size and self.maze[nx,ny] == 0:
            self.state = (nx, ny)
            moved = True
        reward = -0.02
        done = self.state == self.goal
        if done:
            return self.state, 100.0, True
        if not moved:
            reward -= 0.12
        old_d = abs(prev[0] - self.goal[0]) + abs(prev[1] - self.goal[1])
        new_d = abs(self.state[0] - self.goal[0]) + abs(self.state[1] - self.goal[1])
        reward += (old_d - new_d) * 0.5
        return self.state, reward, done

    @staticmethod
    def ease(t:float)->float:
        return 0.5*(1 - math.cos(math.pi*t))

    def render_frame(self,
                     interp_pos:Tuple[float,float],
                     path:List[Tuple[int,int]],
                     q_values:Optional[np.ndarray],
                     action:int,
                     collision:bool,
                     overlay_text:Optional[str],
                     width:int,
                     show_q_bars:bool=True,
                     sprite_agent:bool=True
                     ) -> np.ndarray:
        n = self.size
        cell_px = max(4, width // n)
        W = n*cell_px; H = n*cell_px

        wall = (20,20,20)
        tile_a = (245,245,240)
        tile_b = (235,235,230)
        start_c = (34,139,34)
        goal_c = (178,34,34)
        agent_c = (30,116,200)

        img = Image.new("RGB", (W,H), color=tile_a)
        draw = ImageDraw.Draw(img)

        for i in range(n):
            y0 = i*cell_px
            row = self.maze[i]
            for j in range(n):
                x0 = j*cell_px
                if row[j] == 1:
                    draw.rectangle([x0,y0,x0+cell_px,y0+cell_px], fill=wall)
                else:
                    draw.rectangle([x0,y0,x0+cell_px,y0+cell_px], fill=(tile_a if (i+j)%2==0 else tile_b))

        if path and len(path) > 1:
            overlay = Image.new("RGBA", (W,H), (0,0,0,0)); od = ImageDraw.Draw(overlay)
            coords = [((y+0.5)*cell_px,(x+0.5)*cell_px) for (x,y) in path]
            L = len(coords); lw = max(1, cell_px//3)
            for k in range(len(coords)-1):
                frac = k / max(1, L-2)
                r = int(40 + 200*frac); g = int(120 + 100*frac); b = int(200 - 170*frac)
                od.line([coords[k], coords[k+1]], fill=(r,g,b,200), width=lw)
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        def badge(cell, color, label=None):
            cx = int((cell[1]+0.5)*cell_px); cy = int((cell[0]+0.5)*cell_px)
            r = max(4, cell_px//2)
            sh = Image.new("RGBA",(W,H),(0,0,0,0)); sd = ImageDraw.Draw(sh)
            sd.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(0,0,0,140))
            sh = sh.filter(ImageFilter.GaussianBlur(radius=max(1, cell_px//10)))
            base = Image.alpha_composite(img.convert("RGBA"), sh)
            bd = ImageDraw.Draw(base)
            bd.ellipse([cx-r,cy-r,cx+r,cy+r], fill=color+(255,), outline=(255,255,255,200), width=1)
            if label:
                try:
                    fnt = ImageFont.load_default(); bd.text((cx+r+2, cy-r), label, fill=(255,255,255,220), font=fnt)
                except Exception:
                    pass
            return base.convert("RGB")

        img = badge(self.start, start_c, label="Start")
        img = badge(self.goal, goal_c, label="Goal")

        cx = int((interp_pos[1]+0.5)*cell_px); cy = int((interp_pos[0]+0.5)*cell_px)
        r = max(3, cell_px//3)
        sprite = Image.new("RGBA",(W,H),(0,0,0,0)); sd = ImageDraw.Draw(sprite)
        sd.ellipse([cx-r,cy-r,cx+r,cy+r], fill=agent_c+(255,))
        if sprite_agent and (q_values is not None):
            act_dirs = [(0,1),(1,0),(0,-1),(-1,0)]
            ad = act_dirs[action]
            ex = cx + int(0.25*r * ad[1]); ey = cy + int(0.25*r * ad[0])
            pr = max(1, r//4)
            sd.ellipse([ex-pr-3,ey-pr,ex+pr-3,ey+pr], fill=(10,10,10,255))
            sd.ellipse([ex-pr+3,ey-pr,ex+pr+3,ey+pr], fill=(10,10,10,255))
        if collision:
            flash = Image.new("RGBA",(W,H),(255,60,60,80))
            img = Image.alpha_composite(img.convert("RGBA"), flash).convert("RGB")
        img = Image.alpha_composite(img.convert("RGBA"), sprite).convert("RGB")

        if show_q_bars and (q_values is not None):
            try:
                bar_w = max(3, cell_px//8); bar_h = int(cell_px*1.0)
                bx0 = cx - int(cell_px*1.6); by0 = cy + int(cell_px*0.9)
                qv = np.asarray(q_values, dtype=np.float32)
                qv = np.nan_to_num(qv, nan=qv.min() if np.isfinite(qv).any() else 0.0)
                vmin = float(qv.min()); vmax = float(qv.max()); denom = vmax - vmin + 1e-12
                for i in range(4):
                    h_val = int(bar_h * ((qv[i]-vmin)/denom if denom>0 else 0.0))
                    x0 = bx0 + i*(bar_w+3); y1 = by0 + bar_h; y0 = y1 - h_val
                    draw.rectangle([x0,y0,x0+bar_w,y1], fill=(80,160,220))
                    draw.rectangle([x0,by0,x0+bar_w,y1], outline=(220,220,220), width=1)
            except Exception:
                pass

        if overlay_text:
            try:
                d2 = ImageDraw.Draw(img); fnt = ImageFont.load_default()
                pad = 6
                wtxt, htxt = d2.textsize(overlay_text, font=fnt)
                d2.rectangle([6,6, 6+wtxt+pad*2, 6+htxt+pad*2], fill=(0,0,0,160))
                d2.text((6+pad, 6+pad), overlay_text, fill=(255,255,255,230), font=fnt)
            except Exception:
                pass

        return np.array(img, dtype=np.uint8)


# Replay buffers & models
class PrioritizedReplayBuffer:
    def __init__(self, capacity:int, alpha:float=0.6):
        self.capacity = int(capacity); self.alpha = float(alpha)
        self.buffer = [None]*self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0; self.size = 0; self.default_priority = 1.0
    def add(self, transition):
        self.buffer[self.pos] = transition
        if self.size == 0:
            self.priorities[self.pos] = self.default_priority
        else:
            mp = self.priorities.max()
            self.priorities[self.pos] = mp if mp>0 else self.default_priority
        self.pos = (self.pos+1) % self.capacity
        self.size = min(self.size+1, self.capacity)
    def sample(self, batch_size:int, beta:float):
        if self.size == 0: raise ValueError("Sampling empty replay")
        pr = self.priorities[:self.size].astype(np.float64)
        if pr.sum() <= 0: pr = np.ones_like(pr)
        probs = pr ** self.alpha; probs = probs / (probs.sum() + 1e-12)
        idx = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[i] for i in idx]
        weights = (self.size * probs[idx]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        return samples, idx.tolist(), weights.astype(np.float32)
    def update_priorities(self, indices:List[int], priorities:List[float]):
        for i,p in zip(indices, priorities):
            pv = float(p) if p is not None else 1e-6
            if pv <= 0: pv = 1e-6
            self.priorities[i] = pv
    def __len__(self): return self.size

class ReplayBuffer:
    def __init__(self, capacity:int):
        self.capacity = int(capacity); self.buf = deque(maxlen=self.capacity)
    def add(self, t): self.buf.append(t)
    def sample(self, n): return random.sample(self.buf, n)
    def __len__(self): return len(self.buf)

class DuelingMLP(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden:int=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        v = self.value(x); a = self.adv(x)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

class CNNQ(nn.Module):
    def __init__(self, in_ch:int, action_dim:int, dueling:bool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self._out = None; self.dueling = dueling; self.action_dim = action_dim; self.fc_shared = None
    def forward(self, x):
        h = self.conv(x)
        if self._out is None:
            self._out = h.shape[1]
            self.fc_shared = nn.Linear(self._out, 256).to(h.device)
            if self.dueling:
                self.value = nn.Linear(256,1).to(h.device); self.adv = nn.Linear(256, self.action_dim).to(h.device)
            else:
                self.out = nn.Linear(256, self.action_dim).to(h.device)
        z = F.relu(self.fc_shared(h))
        if self.dueling:
            v = self.value(z); a = self.adv(z); q = v + (a - a.mean(dim=1, keepdim=True)); return q
        else:
            return self.out(z)

class Agent:
    def __init__(self, action_dim:int, cfg:Config, obs_shape:Dict[str,int]):
        self.cfg = cfg; self.device = torch.device(cfg.device); self.action_dim = action_dim
        self.obs_type = cfg.obs_type
        if cfg.obs_type == 'vector':
            self.policy = DuelingMLP(obs_shape['state_dim'], action_dim).to(self.device)
            self.target = DuelingMLP(obs_shape['state_dim'], action_dim).to(self.device)
        else:
            in_ch = obs_shape.get('in_channels', 2)
            self.policy = CNNQ(in_ch, action_dim, dueling=True).to(self.device)
            self.target = CNNQ(in_ch, action_dim, dueling=True).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.step_count = 0

    def act(self, obs:torch.Tensor, epsilon:float)->int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        self.policy.eval()
        with torch.no_grad():
            if obs.dim() == 1:
                obs_t = obs.to(self.device).unsqueeze(0)
            else:
                obs_t = obs.to(self.device).unsqueeze(0)
            q = self.policy(obs_t)
            return int(q.argmax(dim=1).item())

    def q_values(self, obs:torch.Tensor)->np.ndarray:
        self.policy.eval()
        with torch.no_grad():
            if obs.dim() == 1:
                obs_t = obs.to(self.device).unsqueeze(0)
            else:
                obs_t = obs.to(self.device).unsqueeze(0)
            q = self.policy(obs_t).cpu().numpy().squeeze(0)
            return q

    def update(self, batch, weights=None, gamma=0.99):
        states, actions, rewards, next_states, dones = zip(*batch)
        s_t = torch.stack(states).to(self.device); ns_t = torch.stack(next_states).to(self.device)
        a_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_vals = self.policy(s_t).gather(1, a_t).squeeze(1)
        with torch.no_grad():
            na = self.policy(ns_t).argmax(dim=1, keepdim=True)
            nq = self.target(ns_t).gather(1, na).squeeze(1)
            targets = r_t + gamma * nq * (1.0 - d_t)

        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32, device=self.device)
            loss = (F.smooth_l1_loss(q_vals, targets, reduction='none') * w).mean()
        else:
            loss = F.smooth_l1_loss(q_vals, targets)

        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.cfg.target_update_steps == 0:
            self.target.load_state_dict(self.policy.state_dict())
        td_err = (q_vals - targets).detach().abs().cpu().numpy()
        return float(loss.item()), td_err


# Observation
def build_obs_vector(state:Tuple[int,int], goal:Tuple[int,int], size:int)->torch.Tensor:
    sx, sy = state; gx, gy = goal
    return torch.tensor([sx/(size-1), sy/(size-1), gx/(size-1), gy/(size-1)], dtype=torch.float32)

def build_obs_grid(maze:np.ndarray, state:Tuple[int,int], goal:Tuple[int,int])->torch.Tensor:
    ch_maze = maze.astype(np.float32)
    ch_agent_goal = np.zeros_like(ch_maze, dtype=np.float32)
    ch_agent_goal[state[0], state[1]] = 1.0
    ch_agent_goal[goal[0], goal[1]] = -1.0
    arr = np.stack([ch_maze, ch_agent_goal], axis=0)
    return torch.tensor(arr, dtype=torch.float32)

class EgocentricBuilder:
    def __init__(self, maze:np.ndarray, goal:Tuple[int,int], patch_radius:int):
        self.maze = maze.astype(np.float32)
        self.goal = goal
        self.patch_radius = patch_radius
        pad = patch_radius
        self.padded_maze = np.pad(self.maze, pad_width=pad, mode='constant', constant_values=1.0)
        rr = np.arange(self.maze.shape[0])[:,None]; cc = np.arange(self.maze.shape[1])[None,:]
        gr, gc = goal
        dist2 = (rr - gr)**2 + (cc - gc)**2
        sigma = max(1.0, self.maze.shape[0]/8.0)
        heat = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
        if heat.max() > 0:
            heat = heat / float(heat.max())
        self.padded_heat = np.pad(heat, pad_width=pad, mode='constant', constant_values=0.0)

    def get(self, state:Tuple[int,int])->torch.Tensor:
        r,c = state
        pr = self.patch_radius
        H = 2*pr + 1
        patch_maze = self.padded_maze[r:r+H, c:c+H].copy()
        patch_heat = self.padded_heat[r:r+H, c:c+H].copy()
        ch = np.stack([patch_maze, patch_heat], axis=0)
        return torch.tensor(ch, dtype=torch.float32)


# Training loop
ProgressCB = Callable[[int,int,float,float,float,int,Optional[np.ndarray], str], None]

def train(cfg:Config, progress_cb:Optional[ProgressCB]=None)->Dict[str,str]:
    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    frames_dir = os.path.join(cfg.output_dir, "frames")
    if cfg.save_frames_during_training and not os.path.exists(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
    if cfg.tensorboard_logdir is None:
        cfg.tensorboard_logdir = os.path.join(cfg.output_dir, "tb")
    tb = SummaryWriter(cfg.tensorboard_logdir)

    maze, start, goal = MazeGenerator.generate(cfg.maze_size, seed=cfg.seed)
    env = MazeEnv(maze, start, goal)

    if cfg.obs_type == 'vector':
        obs_shape = {'state_dim':4}
        egobuilder = None
    elif cfg.obs_type == 'grid':
        obs_shape = {'in_channels':2, 'H':maze.shape[0], 'W':maze.shape[1]}
        egobuilder = None
    else:
        patch_radius = max(3, min(7, maze.shape[0]//6))
        obs_shape = {'in_channels':2, 'H':2*patch_radius+1, 'W':2*patch_radius+1, 'patch_radius':patch_radius}
        egobuilder = EgocentricBuilder(maze, goal, patch_radius)

    agent = Agent(4, cfg, obs_shape)
    replay = PrioritizedReplayBuffer(cfg.replay_capacity, cfg.per_alpha) if cfg.per else ReplayBuffer(cfg.replay_capacity)

    preview_frames_small = []
    last_preview_frame = None
    rewards = []; losses = []
    total_steps = 0; per_beta = cfg.per_beta_start; start_time = time.time()
    max_steps_episode = cfg.maze_size**2 * cfg.max_steps_multiplier
    global_step = 0

    recent_thoughts = deque(maxlen=60)

    eps_iter = trange(cfg.episodes, desc="Training", unit="ep", ncols=80)
    for ep in eps_iter:
        s = env.reset(); done=False; ep_reward=0.0; ep_steps=0; path=[s]; prev_cell = s
        while not done and ep_steps < max_steps_episode:
            if cfg.obs_type == 'vector':
                obs = build_obs_vector(s, goal, maze.shape[0])
            elif cfg.obs_type == 'grid':
                obs = build_obs_grid(maze, s, goal)
            else:
                obs = egobuilder.get(s)

            eps = max(cfg.epsilon_final, cfg.epsilon_start + (cfg.epsilon_final - cfg.epsilon_start) * (global_step / max(1, cfg.epsilon_decay_frames)))
            try:
                qvals = agent.q_values(obs)
            except Exception:
                qvals = None
            action = agent.act(obs, eps)
            nxt, r, done = env.step(action)
            moved = (nxt != s); collision = not moved
            ep_reward += float(r); ep_steps += 1

            if cfg.obs_type == 'vector':
                obs_next = build_obs_vector(nxt, goal, maze.shape[0])
            elif cfg.obs_type == 'grid':
                obs_next = build_obs_grid(maze, nxt, goal)
            else:
                obs_next = egobuilder.get(nxt)

            replay.add((obs, action, float(r), obs_next, float(done)))

            prev_r, prev_c = prev_cell; next_r, next_c = nxt
            for f_idx in range(cfg.train_frames_per_step):
                t_lin = f_idx / max(1, cfg.train_frames_per_step-1)
                t = env.ease(t_lin)
                cur_r = prev_r + (next_r - prev_r) * t
                cur_c = prev_c + (next_c - prev_c) * t
                overlay = f"Ep {ep+1}/{cfg.episodes}  Step {ep_steps}  Reward {ep_reward:.2f}"
                frame = env.render_frame(
                    interp_pos=(cur_r, cur_c),
                    path=path,
                    q_values=qvals if cfg.show_q_bars else None,
                    action=action,
                    collision=(collision and f_idx==0),
                    overlay_text=overlay,
                    width=cfg.preview_width,
                    show_q_bars=cfg.show_q_bars,
                    sprite_agent=cfg.sprite_agent
                )
                last_preview_frame = frame

            if (ep % max(1, cfg.render_interval) == 0) and (ep_steps % 1 == 0):
                if last_preview_frame is not None:
                    try:
                        pil = Image.fromarray(last_preview_frame)
                        small = pil.resize((min(320, pil.size[0]), int(pil.size[1]*min(320/pil.size[0],1.0))), Image.LANCZOS)
                        preview_frames_small.append(np.array(small))
                    except Exception:
                        pass

            prev_cell = nxt; s = nxt; path.append(s)
            global_step += 1; total_steps += 1

            if cfg.per:
                per_beta = min(1.0, cfg.per_beta_start + (global_step / cfg.per_beta_frames) * (1.0 - cfg.per_beta_start))

            can_update = (cfg.per and len(replay) >= cfg.min_replay_size) or (not cfg.per and len(replay) >= cfg.min_replay_size)
            if can_update:
                try:
                    if cfg.per:
                        batch, idxs, weights = replay.sample(cfg.batch_size, per_beta)
                        batch = [b for b in batch if b is not None]
                        if len(batch) >= cfg.batch_size:
                            loss_val, td_err = agent.update(batch, weights=weights, gamma=cfg.gamma)
                            replay.update_priorities(idxs, (np.abs(td_err)+1e-6).tolist())
                            losses.append(loss_val)
                    else:
                        batch = replay.sample(cfg.batch_size)
                        loss_val, td_err = agent.update(batch, weights=None, gamma=cfg.gamma)
                        losses.append(loss_val)
                except Exception:
                    logger.exception("Update failed")

            thought_text = generate_thoughts(qvals, recent_thoughts, rng, temperature=1.0)

            if progress_cb is not None and (ep_steps % 5 == 0):
                avg10 = float(np.mean(rewards[-10:])) if len(rewards) >= 10 else (float(np.mean(rewards)) if rewards else 0.0)
                try:
                    progress_cb(ep+1, cfg.episodes, ep_reward, avg10, eps, ep_steps, last_preview_frame, thought_text)
                except Exception:
                    logger.exception("progress_cb error")

        rewards.append(ep_reward)
        tb.add_scalar("train/episode_reward", float(ep_reward), ep)
        eps_iter.set_postfix({"ep_reward":f"{ep_reward:.2f}", "eps":f"{eps:.3f}"})

    duration = time.time() - start_time
    logger.info("Training finished: episodes=%d total_steps=%d duration=%.2fs", cfg.episodes, total_steps, duration)
    tb.add_scalar("train/duration_sec", duration, 0)

    preview_gif = os.path.join(cfg.output_dir, "preview.gif")
    try:
        if preview_frames_small:
            imageio.mimsave(preview_gif, preview_frames_small, fps=6)
            logger.info("Saved preview GIF: %s", preview_gif)
    except Exception:
        logger.exception("Failed to save preview GIF")

    final_artifacts = final_render_and_save(agent, env, cfg, egobuilder)

    plot_path = os.path.join(cfg.output_dir, "rewards.png")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4)); plt.plot(rewards, label='reward')
        if len(rewards) > 50:
            ma = np.convolve(rewards, np.ones(50)/50, mode='valid')
            plt.plot(range(49,49+len(ma)), ma, label='MA50', linewidth=2)
        plt.xlabel("Episode"); plt.ylabel("Total reward"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(plot_path, dpi=150); plt.close()
    except Exception:
        logger.exception("Failed to save reward plot")

    model_path = os.path.join(cfg.output_dir, "dqn_model.pth")
    try:
        torch.save(agent.policy.state_dict(), model_path)
    except Exception:
        logger.exception("Failed to save model")

    tb.close()
    return {
        "preview_gif": preview_gif if os.path.exists(preview_gif) else "",
        "final_mp4": final_artifacts.get("mp4",""),
        "final_gif": final_artifacts.get("gif",""),
        "plot": plot_path,
        "model": model_path,
        "log": LOG_PATH
    }


# Final rendering (same)
def final_render_and_save(agent:Agent, env:MazeEnv, cfg:Config, egobuilder:Optional[EgocentricBuilder]):
    try:
        s = env.reset(); path=[s]; done=False; steps=0; max_eval = cfg.maze_size**2 * 5
        while not done and steps < max_eval:
            if cfg.obs_type == 'vector':
                obs = build_obs_vector(s, env.goal, env.maze.shape[0])
            elif cfg.obs_type == 'grid':
                obs = build_obs_grid(env.maze, s, env.goal)
            else:
                obs = egobuilder.get(s)
            a = agent.act(obs, 0.0)
            s,_,done = env.step(a)
            path.append(s); steps += 1
    except Exception:
        logger.exception("Final deterministic rollout failed")
        return {}

    final_frames_dir = os.path.join(cfg.output_dir, "final_frames")
    if cfg.save_final_frames:
        os.makedirs(final_frames_dir, exist_ok=True)

    frames = []
    for i in range(len(path)-1):
        prev = path[i]; nxt = path[i+1]
        if cfg.obs_type == 'vector':
            obs = build_obs_vector(prev, env.goal, env.maze.shape[0])
        elif cfg.obs_type == 'grid':
            obs = build_obs_grid(env.maze, prev, env.goal)
        else:
            obs = egobuilder.get(prev)
        qvals = agent.q_values(obs) if hasattr(agent, 'q_values') else None
        for f_idx in range(cfg.final_frames_per_step):
            t_lin = f_idx / max(1, cfg.final_frames_per_step - 1)
            t = env.ease(t_lin)
            cur_r = prev[0] + (nxt[0] - prev[0]) * t
            cur_c = prev[1] + (nxt[1] - prev[1]) * t
            overlay = f"Final render  Step {i+1}/{len(path)-1}"
            frame = env.render_frame(
                interp_pos=(cur_r, cur_c),
                path=path[:i+1],
                q_values=qvals,
                action=0,
                collision=False,
                overlay_text=overlay,
                width=cfg.final_width,
                show_q_bars=cfg.show_q_bars,
                sprite_agent=cfg.sprite_agent
            )
            frames.append(frame)
            if cfg.save_final_frames:
                try:
                    Image.fromarray(frame).save(os.path.join(final_frames_dir, f"frame_{len(frames):06d}.png"))
                except Exception:
                    logger.exception("Failed saving final frame")

    gif_path = os.path.join(cfg.output_dir, "final_animation.gif")
    mp4_path = os.path.join(cfg.output_dir, "final_animation_30fps.mp4")
    try:
        if frames:
            imageio.mimsave(gif_path, frames, fps=8)
            logger.info("Saved final GIF: %s", gif_path)
    except Exception:
        logger.exception("Failed to save final GIF")
    if cfg.save_mp4:
        try:
            if frames:
                with imageio.get_writer(mp4_path, fps=30, codec='libx264', quality=8) as writer:
                    for f in frames:
                        writer.append_data(f)
                logger.info("Saved final MP4: %s", mp4_path)
        except Exception:
            logger.exception("Failed to save final MP4 (ffmpeg may be missing)")

    return {"gif": gif_path if os.path.exists(gif_path) else "", "mp4": mp4_path if os.path.exists(mp4_path) else ""}


# Streamlit UI with right-side thoughts
def run_streamlit():
    import streamlit as st
    st.set_page_config(page_title="AI Learns to Navigate Mazes", layout="wide")
    st.title("AI Learns to Navigate Mazes")

    tabs = st.tabs(["Main", "Advanced settings"])
    main_tab = tabs[0]; adv_tab = tabs[1]

    with main_tab:
        left_col, center_col, right_col = st.columns([1,3,1])
        with left_col:
            size = st.slider("Maze size (odd)", 11, 41, 21, step=2)
            episodes = st.slider("Episodes", 50, 1000, 300, step=50)
            obs_type = st.selectbox("Observation type", ['vector','grid','egocentric'], index=0)
            start_btn = st.button("🚀 Start training")
            st.markdown("**Note:** preview updates during training; final smooth video is produced after training.")
        with center_col:
            status = st.empty()
            preview = st.empty()
            chart = st.empty()
            artifacts = st.empty()
        with right_col:
            st.markdown("### Agent thoughts")
            thoughts_box = st.empty()
            st.caption("Latest agent thoughts (newest on top)")

    with adv_tab:
        st.subheader("Advanced settings (override defaults)")
        col1, col2 = st.columns(2)
        with col1:
            device = st.selectbox("Device", ['auto','cuda','cpu'], index=0)
            batch_size = st.number_input("Batch size", min_value=8, max_value=2048, value=64, step=8)
            lr = st.number_input("Learning rate", value=5e-4, format="%.6f")
            gamma = st.number_input("Gamma (discount)", value=0.99, format="%.4f")
            seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
            target_update_steps = st.number_input("Target update steps", min_value=1, max_value=100000, value=1000, step=1)
        with col2:
            replay_capacity = st.number_input("Replay capacity", min_value=1000, max_value=200000, value=30000, step=1000)
            use_per = st.checkbox("Use PER (Prioritized Replay)", value=True)
            per_alpha = st.slider("PER alpha", 0.0, 1.0, 0.6)
            per_beta_start = st.slider("PER beta start", 0.0, 1.0, 0.4)
            min_replay_size = st.number_input("Min replay size before updates", min_value=128, max_value=100000, value=1000, step=128)
            max_grad_norm = st.number_input("Max grad norm", min_value=0.1, max_value=100.0, value=10.0, step=0.1)

    # UI-side recent thoughts (display buffer)
    ui_recent_thoughts = deque(maxlen=10)

    reward_history = []; avg_history=[]

    def progress_cb(ep, total, ep_reward, avg10, eps, steps, frame, thought_text):
        try:
            if frame is not None:
                pil = Image.fromarray(frame)
                preview.image(pil, width='content')
            reward_history.append(ep_reward)
            avg = float(np.mean(reward_history[-50:])) if reward_history else ep_reward
            avg_history.append(avg)
            chart.line_chart({"reward": reward_history, "avg50": avg_history})
            status.markdown(f"**Ep**: {ep}/{total}  **Reward**: {ep_reward:.2f}  **Avg50**: {avg:.2f}  **Eps**: {eps:.3f}  **Steps**: {steps}")

            # maintain UI recent thoughts (newest first)
            if thought_text is not None:
                ts = time.strftime("%H:%M:%S")
                entry = f"{ts} — {thought_text}"
                ui_recent_thoughts.appendleft(entry)  # newest at left
                md = "\n".join([f"- {e}" for e in ui_recent_thoughts])
                thoughts_box.markdown(md)
        except Exception:
            logger.exception("progress_cb failed")

    if start_btn:
        cfg = Config.from_args(argparse.Namespace(size=size, episodes=episodes))
        cfg.obs_type = obs_type
        if device != 'auto':
            cfg.device = device
        else:
            cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.batch_size = int(batch_size)
        cfg.lr = float(lr)
        cfg.gamma = float(gamma)
        cfg.seed = int(seed)
        cfg.target_update_steps = int(target_update_steps)
        cfg.replay_capacity = int(replay_capacity)
        cfg.per = bool(use_per)
        cfg.per_alpha = float(per_alpha)
        cfg.per_beta_start = float(per_beta_start)
        cfg.min_replay_size = int(min_replay_size)
        cfg.max_grad_norm = float(max_grad_norm)
        cfg.tensorboard_logdir = os.path.join(cfg.output_dir, "tb")

        status.info(f"Starting training on device `{cfg.device}` — preview will update.")
        artifacts_info = train(cfg, progress_cb=progress_cb)
        status.success("Training finished.")
        try:
            c1, c2 = artifacts.columns(2)
            with c1:
                if artifacts_info.get("preview_gif") and os.path.exists(artifacts_info["preview_gif"]):
                    c1.image(artifacts_info["preview_gif"], caption="Training preview GIF", use_container_width=True)
                if artifacts_info.get("plot") and os.path.exists(artifacts_info["plot"]):
                    c1.image(artifacts_info["plot"], caption="Rewards", use_container_width=True)
            with c2:
                if artifacts_info.get("final_mp4") and os.path.exists(artifacts_info["final_mp4"]):
                    c2.video(artifacts_info["final_mp4"])
                if artifacts_info.get("final_gif") and os.path.exists(artifacts_info["final_gif"]):
                    c2.image(artifacts_info["final_gif"], caption="Final smooth GIF", use_container_width=True)
            artifacts.markdown(f"- Model: `{artifacts_info.get('model','')}`  \n- Log: `{artifacts_info.get('log','')}`")
        except Exception:
            logger.exception("Failed showing artifacts in Streamlit UI")
            artifacts.error("Failed to show artifacts; check outputs/ and train.log")


# CLI entry
def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=21, help='Maze size (odd)')
    parser.add_argument('--episodes', type=int, default=300, help='Training episodes')
    parser.add_argument('--web', action='store_true', help='Run Streamlit UI')
    args = parser.parse_args()
    cfg = Config.from_args(args)
    if args.web:
        run_streamlit()
    else:
        result = train(cfg)
        logger.info("Training completed, artifacts: %s", result)

if __name__ == "__main__":
    main_cli()
