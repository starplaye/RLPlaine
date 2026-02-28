import gymnasium as gym
import pandas as pd
import torch
from PIL import Image
import numpy as np
from math import *
import pygame

CELL_SIZE = 2
MAZE_SIZE = 501

class MazeEnv(gym.Env) :
    
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, in_eval=False, in_debug=False, device="cuda"):
        super().__init__()
        
        self.device = device
        self.dt = 0.01
        self.show_viewer = show_viewer
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.runner = None
        
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        
        # optimisation
        self.batch_idx = torch.arange(self.num_envs, device=self.device) # remplace les ":" car le GPU les gères moins bien que des tenseurs
        
        # mazes
        self.csv_data = pd.read_csv(env_cfg["maze_data"])
        self.csv_size = self.csv_data.shape[0] - 1
        self.img_path = env_cfg["maze_folder"]
        self.mazes = torch.ones((num_envs, MAZE_SIZE, MAZE_SIZE), device=device, dtype=torch.uint8)
        self.sizes = torch.zeros((num_envs, 2), device=device, dtype=torch.uint16)
        
        # players
        self.pos = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        self.init_pos = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        self.last_pos = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        self.target = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        
        self.rel_pos_to_target = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        self.last_rel_pos_to_target = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        
        self.actions = torch.zeros((num_envs, 1), device=device, dtype=torch.long)
        self.memory = torch.zeros((num_envs, env_cfg["memory_size"], 2), device=device, dtype=torch.int)
        self.vision = torch.zeros((num_envs, 3, 3), device=device, dtype=torch.uint8)
        self.vision_range = env_cfg["vision_range"]
        
        # time
        self.episode_length_buf = torch.zeros((self.num_envs,1), device=self.device, dtype=torch.int)
        self.max_episode_length = reward_cfg["timeout"] * (1 - (reward_cfg["base_difficulty"]/reward_cfg["max_difficulty"]))
        
        # buffer
        self.win_condition = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.time_condition = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.already_move = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        
        self.load_mazes_in_idx(range(num_envs))
        
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float16)
            
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        
        if self.show_viewer : 
            self.setup_viewer()
            self.is_update = False
        
        # print(self.mazes[0, self.pos[0][0], self.pos[0][1]])
        # print(sqrt((self.pos[0][0].item() - self.target[0][0].item())**2 + (self.pos[0][1].item() - self.target[0][1].item())**2))
    
    
    def find_target_at_distance(self,idx,distance):
        already = set()
        to_view = [(self.pos[idx,0].item(), self.pos[idx,1].item(), 0)] # set of triplet (x,y,distance)
        quat = [(-1,0),(1,0),(0,-1),(0,1)]
        
        while to_view :
            x,y,d = to_view.pop()
            already.add((x,y))
            for (vx, vy) in quat :
                nx, ny = x + vx, y + vy
                t = (nx,ny)
                
                if self.mazes[idx, ny, nx] == 0 and ny >= 0 and ny < MAZE_SIZE and nx >= 0 and nx < MAZE_SIZE and t not in already:
                    if (d + 1) >= distance :
                        return (nx,ny)
                    
                    to_view.append((nx,ny,d+1))
                    
        return None
            
    
    def load_mazes_in_idx(self, env_idx):
        
        lst_random_data_idx = np.random.randint(1,self.csv_size,len(env_idx))
        
        for i in range(len(env_idx)):
            idx = env_idx[i]
            random_data_idx = lst_random_data_idx[i]
            
            image_id = self.csv_data.iloc[random_data_idx, 0].astype(str)
            width = self.csv_data.iloc[random_data_idx, 1].astype(int)
            height = self.csv_data.iloc[random_data_idx, 2].astype(int)
            self.sizes[idx, 0], self.sizes[idx, 1] = width, height # stocke w/h
            
            img = Image.open(f"{self.img_path}{image_id}.png").convert('L')
            img = np.array(img, dtype=np.uint8)          
            img = torch.from_numpy(img).to(self.device)     
            img = img != 0 

            self.mazes[idx] = 1
            self.mazes[idx, :img.shape[0], :img.shape[1]] = img
            
            coords = torch.nonzero(img == 0, as_tuple=False)
            coords = coords[:, [1, 0]] # swap (y,x) to (x,y)
            self.init_pos[idx] = coords[torch.randint(0,coords.shape[0],(1,)).item()]
            self.pos[idx] = self.init_pos[idx]
            self.last_pos[idx] = self.init_pos[idx]
            
            target = self.find_target_at_distance(idx, np.random.randint(1,50))
            if not target : 
                self.target[idx] = coords[torch.randint(0,coords.shape[0],(1,)).item()]
            else :
                x,y = target
                self.target[idx, 0], self.target[idx, 1] = x,y
                
            self.mazes[idx, self.target[idx,1], self.target[idx,0]] = -1
            
            # self.rel_pos_to_target[idx] = self.target[idx] - self.pos[idx]
            
            self.memory[idx,:,:] = 0
            # self.memory[idx,0,:] = self.rel_pos_to_target[idx]
            self.memory[idx,0,:] = self.init_pos[idx]
            
            if self.show_viewer and idx == 0 :
                self.is_update = True
            
    
    def setup_viewer(self):
        pygame.init()
        self.display = pygame.display.set_mode((CELL_SIZE*MAZE_SIZE, CELL_SIZE*MAZE_SIZE))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Maze Viewer")
            
    def update_viewer(self):
        
        # Draw Maze
        # if self.is_update :
        self.is_update = False
        
        self.display.fill((0,0,0))
        
        maze = self.mazes[0].cpu()
        maze[self.target[0,1], self.target[0,0]] = 0
        
        img = (1 - maze) * 255  # vide = blanc, mur = noir   
        rgb = np.stack([img, img, img], axis=-1).astype(np.uint8) # RGB (H,W) -> (H,W,3)  
        rgb = np.transpose(rgb, (1, 0, 2))# pygame veut (W,H,3)

        surf = pygame.surfarray.make_surface(rgb)
        
        scaled = pygame.transform.scale(surf, (MAZE_SIZE * CELL_SIZE, MAZE_SIZE * CELL_SIZE))
        self.display.blit(scaled, (0, 0))
    
        
        # Draw player    
        x,y = self.last_pos[0]
        x,y = x.item(), y.item()
        pygame.draw.rect(
                    self.display,
                    (255,255,255),
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )
        
        x,y = self.pos[0]
        x,y = x.item(), y.item()
        pygame.draw.rect(
                    self.display,
                    (255,0,0),
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )
        r = 10
        pygame.draw.rect(self.display,
                    (255,0,0),
                    ((x-r) * CELL_SIZE, (y-r) * CELL_SIZE, CELL_SIZE*r*2, CELL_SIZE)
        )
        pygame.draw.rect(self.display,
                    (255,0,0),
                    ((x-r) * CELL_SIZE, (y-r) * CELL_SIZE, CELL_SIZE, CELL_SIZE*r*2)
        )
        pygame.draw.rect(self.display,
                    (255,0,0),
                    ((x+r) * CELL_SIZE, (y-r) * CELL_SIZE, CELL_SIZE, CELL_SIZE*r*2)
        )
        pygame.draw.rect(self.display,
                    (255,0,0),
                    ((x-r) * CELL_SIZE, (y+r) * CELL_SIZE, CELL_SIZE*r*2, CELL_SIZE)
        )
        
        # Draw target
        x,y = self.target[0]
        x,y = x.item(), y.item()
        pygame.draw.rect(
                    self.display,
                    (0,0,255),
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

        pygame.display.flip()
    
            
    def compute_vision_patch(self):
        for y in range(-self.vision_range,self.vision_range+1):
            for x in range(-self.vision_range,self.vision_range+1):
                current_x, current_y = self.pos[:,0] + x, self.pos[:,1] + y
                envs_idx = (current_x < MAZE_SIZE) & (current_x >= 0) & (current_y < MAZE_SIZE) & (current_y >= 0)
                self.vision[envs_idx,y,x] = self.mazes[envs_idx, current_y[envs_idx], current_x[envs_idx]]
                self.vision[~envs_idx,y,x] = 1
              
            
    def step(self, actions):
        # Move player
        self.last_pos[self.batch_idx] = self.pos[self.batch_idx]
        
        # self.actions = torch.clip(torch.round(actions),-1,1)
        self.actions[self.batch_idx] = torch.argmax(actions, dim=1).unsqueeze(-1)
        self.already_move[self.batch_idx] = False
        # print(self.actions)
        
        # 1 case : up
        up = ((self.actions[self.batch_idx,0] == 0) 
              & ((self.pos[self.batch_idx, 1] + 1).long() < MAZE_SIZE) 
              & (self.mazes[self.batch_idx, (self.pos[self.batch_idx, 1] + 1).long(), self.pos[self.batch_idx, 0].long()] != 1))
        
        idx = up.nonzero(as_tuple=True)[0]
        self.already_move = up
        self.pos[idx, 1] += 1
        
        # -------------------------
        
        # 2 case : down
        down = ((self.actions[self.batch_idx,0] == 1) 
              & (~self.already_move)
              & ((self.pos[self.batch_idx, 1] - 1).long() >= 0) 
              & (self.mazes[self.batch_idx, (self.pos[self.batch_idx, 1] - 1).long(), self.pos[self.batch_idx, 0].long()] != 1))
        
        idx = down.nonzero(as_tuple=True)[0]
        self.already_move |= down
        self.pos[idx, 1] -= 1
        
        # -------------------------
        
        # 3 case : right
        right = ((self.actions[self.batch_idx,0] == 2) 
              & (~self.already_move)
              & ((self.pos[self.batch_idx, 0] + 1).long() < MAZE_SIZE) 
              & (self.mazes[self.batch_idx, self.pos[self.batch_idx, 1].long(), (self.pos[self.batch_idx, 0] + 1).long()] != 1))
        
        idx = right.nonzero(as_tuple=True)[0]
        self.already_move |= right
        self.pos[idx, 0] += 1
        
        # -------------------------
        
        # 4 case : left
        left = ((self.actions[self.batch_idx,0] == 3) 
              & (~self.already_move)
              & ((self.pos[self.batch_idx, 0] - 1).long() >= 0) 
              & (self.mazes[self.batch_idx, self.pos[self.batch_idx, 1].long(), (self.pos[self.batch_idx, 0] - 1).long()] != 1))
        
        idx = left.nonzero(as_tuple=True)[0]
        self.already_move |= left
        self.pos[idx, 0] -= 1
        
        # --------- Update buffer -----------
        
        self.rel_pos_to_target = self.target - self.pos
        self.last_rel_pos_to_target = self.target - self.last_pos
        self.episode_length_buf[self.batch_idx,0] += 1
        
        # ---------- Update viewer ------------
        
        if self.show_viewer : self.update_viewer()
        
        # -----------------------------------
    
        self.win_condition = (self.pos == self.target).all(dim=1)
        self.time_condition = (self.episode_length_buf[self.batch_idx,0] > self.max_episode_length)
        
        self.reset_buf = self.time_condition | self.win_condition
        
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        
        # ----------------------------------
        
        # compute vision
        self.compute_vision_patch()
        vision_flat = self.vision.reshape(self.num_envs,-1)
        
        # roll up the memory
        self.memory[self.batch_idx,1:,:] = self.memory[self.batch_idx,:-1,:].clone()
        self.memory[self.batch_idx,0,:] = self.pos[self.batch_idx]
        memory_flat = self.memory.reshape(self.num_envs,-1)
        
        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        # observations normalized (-1,1)
        self.obs_buf = torch.cat(
            [
                (self.pos.float() / (MAZE_SIZE-1)) * 2 - 1, # 2
                (self.rel_pos_to_target.float() / (MAZE_SIZE-1)) * 2 - 1, # 2
                vision_flat.float(), # 9
                (memory_flat.float() / (MAZE_SIZE-1)) * 2 - 1, # 20
                self.actions.float()/3 * 2 - 1,  # 1
                self.episode_length_buf.float()/self.max_episode_length, # 1
            ],
            axis=-1,
        )
        
        self.extras["observations"]["critic"] = self.obs_buf
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras      
    
    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf 
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        
        self.episode_length_buf[envs_idx,0] = 0
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.max_episode_length
            )
            self.episode_sums[key][envs_idx] = 0.0

        self.load_mazes_in_idx(envs_idx)
        

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(range(self.num_envs))
        return self.obs_buf, None

    # ------------ reward functions----------------

    def _reward_distance(self): 
        distance_rew = torch.sum(torch.square(self.last_rel_pos_to_target), dim=1) - torch.sum(torch.square(self.rel_pos_to_target), dim=1) 
        distance_rew = torch.clamp(distance_rew,min=-1) 
        return distance_rew
    
    def _reward_movement(self):
        move_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        exists = (self.memory == self.pos.unsqueeze(1)).all(dim=2).any(dim=1)
        move_rew[exists] = -1
        return move_rew
    
    def _reward_target(self):
        target_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        target_rew[self.win_condition] = 10
        return target_rew
    
    def _reward_timeout(self):
        timeout_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        timeout_rew[self.time_condition] = -1
        return timeout_rew
    
    def _reward_already(self):
        already_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        already_rew[~self.already_move] = -1
        return already_rew
            