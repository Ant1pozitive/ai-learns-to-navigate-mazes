"""
Run web:
    streamlit run main.py -- --web
or CLI:
    python main.py --size 21 --episodes 300
"""
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

import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import matplotlib
import matplotlib.pyplot as plt
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
logger = logging.getLogger("rl_opt")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
ch.setLevel(logging.INFO)
logger.addHandler(fh); logger.addHandler(ch)


# Config
@dataclass
class Config:
    maze_size: int = 21
    episodes: int = 300

    output_dir: str = OUTPUT_DIR
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    replay_capacity: int = 30000
    target_update_steps: int = 1000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.02
    epsilon_decay_steps: int = 40000
    seed: int = 42
    render_interval: int = 10
    max_steps_multiplier: int = 3

    # obs types: 'vector', 'grid', 'egocentric'
    obs_type: str = 'vector'

    # replay / PER
    per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000

    dueling: bool = True
    min_replay_size: int = 2000
    max_grad_norm: float = 10.0

    # visualization: different values for training vs final video
    train_frames_per_step: int = 1         # small -> training fast
    final_frames_per_step: int = 16        # large -> very smooth final video
    preview_width: int = 800               # preview width in Streamlit during training (smaller -> faster)
    final_width: int = 1024                # final high-res width for video
    save_frames_during_training: bool = False  # do NOT save thousands of frames during training by default
    save_final_frames: bool = True         # save frames when doing final rendering
    save_mp4: bool = True

    thinking_overlay: bool = True
    sprite_agent: bool = True
    show_q_bars: bool = True

    tensorboard_logdir: str = os.path.join(OUTPUT_DIR, "tb")

    @classmethod
    def from_args(cls, args):
        c = cls(maze_size=args.size, episodes=args.episodes)
        c.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return c


# Determinism
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            maze[x,y]=0
            dirs=[(0,2),(2,0),(0,-2),(-2,0)]
            random.shuffle(dirs)
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if 0 < nx < size-1 and 0 < ny < size-1 and maze[nx,ny]==1:
                    maze[x+dx//2, y+dy//2]=0
                    dfs(nx,ny)
        start=(1,1); goal=(size-2,size-2)
        dfs(start[0], start[1])
        maze[start]=0; maze[goal]=0
        # open borders for aesthetics
        maze[0,1]=0; maze[size-1,size-2]=0
        return maze, start, goal


# Env + renderer
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
        moved=False
        if 0<=nx<self.size and 0<=ny<self.size and self.maze[nx,ny]==0:
            self.state=(nx,ny); moved=True
        reward=-0.02
        done = self.state == self.goal
        if done: return self.state, 100.0, True
        old_d = abs(prev[0]-self.goal[0]) + abs(prev[1]-self.goal[1])
        new_d = abs(self.state[0]-self.goal[0]) + abs(self.state[1]-self.goal[1])
        reward += (old_d - new_d)*0.5
        return self.state, reward, done

    @staticmethod
    def ease(t:float)->float:
        return 0.5*(1 - math.cos(math.pi * t))

    # frame render
    def render_frame(self,
                     interp_pos:Tuple[float,float],
                     path:List[Tuple[int,int]],
                     q_values:Optional[np.ndarray],
                     action:int,
                     collision:bool,
                     overlay_text:Optional[str],
                     width:int,
                     show_q_bars:bool=True,
                     thinking_overlay:bool=True,
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

        img = Image.new("RGB",(W,H), color=tile_a)
        draw = ImageDraw.Draw(img)

        # tiles
        for i in range(n):
            row_y = i*cell_px
            for j in range(n):
                x0 = j*cell_px; y0 = row_y
                if self.maze[i,j]==1:
                    draw.rectangle([x0,y0,x0+cell_px,y0+cell_px], fill=wall)
                else:
                    draw.rectangle([x0,y0,x0+cell_px,y0+cell_px], fill=(tile_a if (i+j)%2==0 else tile_b))

        # path drawn with precomputed palette
        if path and len(path)>1:
            overlay = Image.new("RGBA",(W,H),(0,0,0,0)); od = ImageDraw.Draw(overlay)
            coords = [((y+0.5)*cell_px,(x+0.5)*cell_px) for (x,y) in path]
            L = len(coords)
            # linear color interpolation
            for k in range(len(coords)-1):
                frac = k / max(1, L-2)
                r = int(30 + 225*frac); g = int(120 + 120*frac); b = int(200 - 150*frac)
                od.line([coords[k], coords[k+1]], fill=(r,g,b,200), width=max(1,cell_px//3))
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        # badges
        def badge(cell, color, label=None):
            cx=int((cell[1]+0.5)*cell_px); cy=int((cell[0]+0.5)*cell_px)
            r=max(4, cell_px//2)
            sh=Image.new("RGBA",(W,H),(0,0,0,0)); sd=ImageDraw.Draw(sh)
            sd.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(0,0,0,140))
            sh = sh.filter(ImageFilter.GaussianBlur(radius=max(1, cell_px//10)))
            base = Image.alpha_composite(img.convert("RGBA"), sh)
            bd = ImageDraw.Draw(base)
            bd.ellipse([cx-r,cy-r,cx+r,cy+r], fill=color+(255,), outline=(255,255,255,200), width=1)
            if label:
                try:
                    fnt = ImageFont.load_default()
                    bd.text((cx+r+2, cy-r), label, fill=(255,255,255,220), font=fnt)
                except Exception:
                    pass
            return base.convert("RGB")
        img = badge(self.start, start_c, label="Start")
        img = badge(self.goal, goal_c, label="Goal")

        cx = int((interp_pos[1]+0.5)*cell_px); cy=int((interp_pos[0]+0.5)*cell_px)
        # thinking overlay: simple wedges based on softmax-like normalization of q_values
        if thinking_overlay and q_values is not None:
            qv = np.array(q_values, dtype=np.float32)
            qv = qv - qv.max()
            exps = np.exp(qv)
            probs = exps / (exps.sum() + 1e-12)
            overlay = Image.new("RGBA",(W,H),(0,0,0,0)); od = ImageDraw.Draw(overlay)
            dir_angles = [(-30,30),(60,120),(150,210),(-150,-120)]
            colors = [(70,130,180),(60,180,75),(200,80,70),(180,80,200)]
            for i in range(4):
                p = float(probs[i])
                if p < 0.001: continue
                ang0, ang1 = dir_angles[i]
                alpha = int(180*(0.2 + 0.8*p))
                col = colors[i] + (alpha,)
                pts=[(cx,cy)]
                for a in np.linspace(math.radians(ang0), math.radians(ang1), num=6):
                    x = cx + int(cell_px * 1.8 * math.cos(a)); y = cy + int(cell_px * 1.8 * math.sin(a))
                    pts.append((x,y))
                od.polygon(pts, fill=col)
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        # agent sprite
        r = max(3, cell_px//3)
        sprite = Image.new("RGBA",(W,H),(0,0,0,0)); sd=ImageDraw.Draw(sprite)
        sd.ellipse([cx-r,cy-r,cx+r,cy+r], fill=agent_c+(255,))
        if sprite_agent and q_values is not None:
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

        # q bars
        if show_q_bars and q_values is not None:
            bar_w = max(3, cell_px//8); bar_h = int(cell_px*1.0)
            bx0 = cx - int(cell_px*1.6); by0 = cy - int(cell_px*1.1)
            qv = np.array(q_values, dtype=np.float32)
            qv = np.nan_to_num(qv, nan=qv.min() if np.isfinite(qv).any() else 0.0)
            vmin = float(qv.min()); vmax = float(qv.max()); denom = vmax - vmin + 1e-12
            for i in range(4):
                h_val = int(bar_h * ((qv[i]-vmin)/denom if denom>0 else 0.0))
                x0 = bx0 + i*(bar_w+3); y1 = by0 + bar_h; y0 = y1 - h_val
                draw.rectangle([x0,y0,x0+bar_w,y1], fill=(80,160,220))
                draw.rectangle([x0,by0,x0+bar_w,y1], outline=(220,220,220), width=1)

        # overlay text box
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


# Replay buffer & models
class PrioritizedReplayBuffer:
    def __init__(self, capacity:int, alpha:float=0.6):
        self.capacity = int(capacity); self.alpha = float(alpha)
        self.buffer = [None]*self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0; self.size=0; self.default_priority=1.0
    def add(self, transition):
        self.buffer[self.pos] = transition
        if self.size==0:
            self.priorities[self.pos] = self.default_priority
        else:
            mp = self.priorities.max()
            self.priorities[self.pos] = mp if mp>0 else self.default_priority
        self.pos = (self.pos+1) % self.capacity
        self.size = min(self.size+1, self.capacity)
    def sample(self, batch_size:int, beta:float):
        if self.size==0: raise ValueError("Sampling empty replay")
        pr = self.priorities[:self.size].astype(np.float64)
        if pr.sum() <=0: pr = np.ones_like(pr)
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
        self.capacity=int(capacity); self.buf = deque(maxlen=self.capacity)
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
    def forward(self,x):
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
            self.policy = CNNQ(in_ch, action_dim, dueling=cfg.dueling).to(self.device)
            self.target = CNNQ(in_ch, action_dim, dueling=cfg.dueling).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.step_count = 0

    def act(self, obs:torch.Tensor, epsilon:float)->int:
        if random.random() < epsilon: return random.randrange(self.action_dim)
        self.policy.eval()
        with torch.no_grad():
            if obs.dim()==1:
                obs_t = obs.to(self.device).unsqueeze(0)
            else:
                obs_t = obs.to(self.device).unsqueeze(0)
            q = self.policy(obs_t)
            return int(q.argmax(dim=1).item())

    def q_values(self, obs:torch.Tensor)->np.ndarray:
        self.policy.eval()
        with torch.no_grad():
            if obs.dim()==1:
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


# Observations
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

# Precompute padded arrays for egocentric to avoid expensive ops each step
class EgocentricBuilder:
    def __init__(self, maze:np.ndarray, goal:Tuple[int,int], patch_radius:int):
        self.maze = maze.astype(np.float32)
        self.goal = goal
        self.patch_radius = patch_radius
        pad = patch_radius
        self.padded_maze = np.pad(self.maze, pad_width=pad, mode='constant', constant_values=1.0)
        # precompute full goal heatmap
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
        # slice from padded arrays
        patch_maze = self.padded_maze[r:r+H, c:c+H].copy()
        patch_heat = self.padded_heat[r:r+H, c:c+H].copy()
        ch = np.stack([patch_maze, patch_heat], axis=0)
        return torch.tensor(ch, dtype=torch.float32)


# Training loop
ProgressCB = Callable[[int,int,float,float,float,int,Optional[np.ndarray]], None]

def train(cfg:Config, progress_cb:Optional[ProgressCB]=None)->Dict[str,str]:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    frames_dir = os.path.join(cfg.output_dir, "frames")
    if cfg.save_frames_during_training and not os.path.exists(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)

    tb = SummaryWriter(cfg.tensorboard_logdir)

    maze, start, goal = MazeGenerator.generate(cfg.maze_size, seed=cfg.seed)
    env = MazeEnv(maze, start, goal)

    # prepare obs shape and egocentric builder if needed
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

    # storage for preview frames to show a GIF during training
    preview_frames_small = []
    last_preview_frame = None

    frames_for_final = []

    rewards = []; losses = []
    total_steps = 0; per_beta = cfg.per_beta_start; start_time = time.time()
    max_steps_episode = cfg.maze_size**2 * cfg.max_steps_multiplier
    global_step = 0

    eps_iter = trange(cfg.episodes, desc="Training", unit="ep", ncols=80)
    for ep in eps_iter:
        s = env.reset(); done=False; ep_reward=0.0; ep_steps=0; path=[s]; prev_cell = s
        while not done and ep_steps < max_steps_episode:
            # build obs
            if cfg.obs_type == 'vector':
                obs = build_obs_vector(s, goal, maze.shape[0])
            elif cfg.obs_type == 'grid':
                obs = build_obs_grid(maze, s, goal)
            else:
                obs = egobuilder.get(s)
            epsilon = cfg.epsilon_final if global_step >= cfg.epsilon_decay_steps else (cfg.epsilon_start + (cfg.epsilon_final - cfg.epsilon_start) * (global_step / cfg.epsilon_decay_steps))
            qvals = agent.q_values(obs) if hasattr(agent, 'q_values') else None
            action = agent.act(obs, epsilon)
            nxt, r, done = env.step(action)
            moved = (nxt != s); collision = not moved
            ep_reward += float(r); ep_steps += 1

            # next obs
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
                    q_values=qvals if cfg.thinking_overlay else None,
                    action=action,
                    collision=(collision and f_idx==0),
                    overlay_text=overlay,
                    width=cfg.preview_width,
                    show_q_bars=cfg.show_q_bars,
                    thinking_overlay=cfg.thinking_overlay,
                    sprite_agent=cfg.sprite_agent
                )
                last_preview_frame = frame

            # save one small preview frame every render_interval episodes
            if (ep % max(1, cfg.render_interval) == 0) and (ep_steps % 1 == 0):
                if last_preview_frame is not None:
                    # store scaled-down preview for GIF
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

            # training update
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

            # progress callback
            if progress_cb is not None and (ep_steps % 5 == 0):
                avg10 = float(np.mean(rewards[-10:])) if len(rewards) >= 10 else (float(np.mean(rewards)) if rewards else 0.0)
                try:
                    progress_cb(ep+1, cfg.episodes, ep_reward, avg10, epsilon, ep_steps, last_preview_frame)
                except Exception:
                    logger.exception("progress_cb error")

        rewards.append(ep_reward)
        tb.add_scalar("train/episode_reward", float(ep_reward), ep)
        tb.add_scalar("train/epsilon", float(epsilon), ep)
        eps_iter.set_postfix({"ep_reward":f"{ep_reward:.2f}", "eps":f"{epsilon:.3f}"})

    duration = time.time() - start_time
    logger.info("Training finished: episodes=%d total_steps=%d duration=%.2fs", cfg.episodes, total_steps, duration)
    tb.add_scalar("train/duration_sec", duration, 0)

    # Save light GIF from preview_frames_small
    gif_path = os.path.join(cfg.output_dir, "training_preview.gif")
    try:
        if preview_frames_small:
            with io.StringIO() as s_out:
                imageio.mimsave(gif_path, preview_frames_small, fps=6)
            logger.info("Saved preview GIF: %s", gif_path)
    except Exception:
        logger.exception("Failed to save preview GIF")

    # After training: produce full-quality final video by running deterministic evaluation
    final_artifacts = final_render_and_save(agent, env, cfg, egobuilder)

    # save rewards plot
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

    # save model
    model_path = os.path.join(cfg.output_dir, "dqn_model.pth")
    try:
        torch.save(agent.policy.state_dict(), model_path)
    except Exception:
        logger.exception("Failed to save model")

    tb.close()
    # return summary, prefer final artifacts (mp4/gif/plot)
    res = {
        "preview_gif": gif_path if os.path.exists(gif_path) else "",
        "final_mp4": final_artifacts.get("mp4",""),
        "final_gif": final_artifacts.get("gif",""),
        "plot": plot_path,
        "model": model_path,
        "log": LOG_PATH
    }
    return res


# Final rendering (after training) - deterministic trajectory and high-quality frames
def final_render_and_save(agent:Agent, env:MazeEnv, cfg:Config, egobuilder:Optional[EgocentricBuilder]):
    # Run deterministic episode (greedy) and generate smooth frames with final_frames_per_step
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

    # create frames dir for final frames
    final_frames_dir = os.path.join(cfg.output_dir, "final_frames")
    if cfg.save_final_frames:
        os.makedirs(final_frames_dir, exist_ok=True)

    # generate interpolated frames along the deterministic path
    frames = []
    for i in range(len(path)-1):
        prev = path[i]; nxt = path[i+1]
        if cfg.obs_type == 'vector':
            obs = build_obs_vector(prev, env.goal, env.maze.shape[0])
        elif cfg.obs_type == 'grid':
            obs = build_obs_grid(env.maze, prev, env.goal)
        else:
            # egocentric
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
                q_values=qvals if cfg.thinking_overlay else None,
                action=0,
                collision=False,
                overlay_text=overlay,
                width=cfg.final_width,
                show_q_bars=cfg.show_q_bars,
                thinking_overlay=cfg.thinking_overlay,
                sprite_agent=cfg.sprite_agent
            )
            frames.append(frame)
            if cfg.save_final_frames:
                idx = len(frames)
                try:
                    Image.fromarray(frame).save(os.path.join(final_frames_dir, f"frame_{idx:06d}.png"))
                except Exception:
                    logger.exception("Failed to write final frame to disk")
    # save final GIF and MP4
    gif_path = os.path.join(cfg.output_dir, "final_animation.gif")
    mp4_path = os.path.join(cfg.output_dir, "final_animation_30fps.mp4")
    try:
        if frames:
            imageio.mimsave(gif_path, frames, fps=8)
            logger.info("Saved final GIF to %s", gif_path)
    except Exception:
        logger.exception("Failed to save final GIF")
    try:
        if frames:
            with imageio.get_writer(mp4_path, fps=30, codec='libx264', quality=8) as writer:
                for f in frames:
                    writer.append_data(f)
            logger.info("Saved final MP4 to %s", mp4_path)
    except Exception:
        logger.exception("Failed to save final MP4 (ffmpeg may be missing)")

    return {"gif": gif_path if os.path.exists(gif_path) else "", "mp4": mp4_path if os.path.exists(mp4_path) else ""}


# Streamlit UI
def run_streamlit():
    import streamlit as st
    st.set_page_config(page_title="RL Maze", layout="wide")
    st.title("AI learns to navigate mazes")

    left, right = st.columns([1,2])
    with left:
        size = st.slider("Maze size (odd)", 11, 41, 21, step=2)
        episodes = st.slider("Episodes", 50, 1000, 300, step=50)
        obs = st.selectbox("Observation type", ['vector','grid','egocentric'], index=0)
        start_btn = st.button("Start training")
    with right:
        st.caption("Training uses faster preview rendering; final smooth video is generated after training.")
        status = st.empty(); preview = st.empty(); chart = st.empty(); artifacts = st.empty()

    reward_history=[]; avg_history=[]

    def progress_cb(ep, total, ep_reward, avg10, eps, steps, frame):
        try:
            if frame is not None:
                pil = Image.fromarray(frame)
                if pil.size[0] != 800:
                    new_h = int(pil.size[1] * (800 / pil.size[0]))
                    pil = pil.resize((800, new_h), Image.LANCZOS)
                preview.image(pil, use_column_width=False)
            reward_history.append(ep_reward)
            avg = float(np.mean(reward_history[-50:])) if reward_history else ep_reward
            avg_history.append(avg)
            chart.line_chart({"reward": reward_history, "avg50": avg_history})
            status.markdown(f"**Ep**: {ep}/{total}  **Reward**: {ep_reward:.2f}  **Avg50**: {avg:.2f}  **Eps**: {eps:.3f}  **Steps**: {steps}")
        except Exception:
            logger.exception("progress_cb failed")

    if start_btn:
        cfg = Config.from_args(argparse.Namespace(size=size, episodes=episodes))
        cfg.obs_type = obs
        status.info(f"Starting training on device `{cfg.device}` — preview will update.")
        artifacts_info = train(cfg, progress_cb=progress_cb)
        status.success("Training finished.")
        try:
            c1, c2 = artifacts.columns(2)
            with c1:
                if artifacts_info.get("preview_gif") and os.path.exists(artifacts_info["preview_gif"]):
                    c1.image(artifacts_info["preview_gif"], caption="Training preview GIF")
                if artifacts_info.get("plot") and os.path.exists(artifacts_info["plot"]):
                    c1.image(artifacts_info["plot"], caption="Rewards")
            with c2:
                if artifacts_info.get("final_mp4") and os.path.exists(artifacts_info["final_mp4"]):
                    c2.video(artifacts_info["final_mp4"])
                if artifacts_info.get("final_gif") and os.path.exists(artifacts_info["final_gif"]):
                    c2.image(artifacts_info["final_gif"], caption="Final smooth GIF")
            artifacts.markdown(f"- Model: `{artifacts_info.get('model','')}`  \n- Log: `{artifacts_info.get('log','')}`")
        except Exception:
            logger.exception("Failed showing artifacts")
            artifacts.error("Failed to show artifacts; check outputs/ and train.log")


# CLI entry
def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=21)
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--web', action='store_true')
    args = parser.parse_args()
    cfg = Config.from_args(args)
    if args.web:
        run_streamlit()
    else:
        res = train(cfg)
        logger.info("Artifacts: %s", res)

if __name__ == "__main__":
    main_cli()
