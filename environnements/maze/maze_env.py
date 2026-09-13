import gymnasium as gym
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from math import *
import pygame
import time
import gc

CELL_SIZE = 2
MAZE_SIZE = 501
BATCH_MEMORY_CEIL = 30000

class MazeEnv(gym.Env) :
    
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, writer_data, id, show_viewer=False, in_eval=False, in_debug=False, is_manager=False, manager_cfg=None, device="cuda"):
        super().__init__()
        
        self.in_init = True
        self.device = device
        self.dt = 0.01
        self.show_viewer = show_viewer
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.runner = None
        self.is_manager = is_manager
        self.writer_data = writer_data
        self.write_already = False
        self.id = id
        
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"] if not is_manager else manager_cfg["reward_cfg"]["reward_scales"]
        
        self.density = reward_cfg["density"][reward_cfg["current_density"]]
        self.difficulty = reward_cfg["difficulty"]
        self.range_steps = reward_cfg["range_steps"]
        self.timeout = reward_cfg["timeout"]
        self.it_limit_list = reward_cfg["iterations_to_next_diffculty"]
        self.it_limit = 0
        for i in range(self.difficulty+1):
            self.it_limit += self.it_limit_list[i]
        
        # optimisation
        self.all_batch_idx = torch.arange(self.num_envs, device=self.device) # remplace les ":" car le GPU les gères moins bien que des tenseurs d'index
        self.batch_idx = torch.arange(self.num_envs, device=self.device)
        
        # mazes
        self.csv_data = pd.read_csv(env_cfg["maze_data"])
        self.csv_size = self.csv_data.shape[0] - 1
        self.img_path = env_cfg["maze_folder"]
        self.mazes = torch.ones((num_envs, MAZE_SIZE, MAZE_SIZE), device=device, dtype=torch.uint8)
        self.env_idx_maze = torch.zeros((num_envs,), device=device, dtype=torch.long)
        self.sizes = torch.zeros((num_envs, 2), device=device, dtype=torch.uint16)
        self.heat_kernel = torch.tensor([[[
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]]], dtype=torch.float32, device=self.device)
        self.len_path = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)

        # players
        self.pos = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        self.init_pos = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        self.last_pos = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        self.target_worker = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        self.target_manager = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        
        self.rel_pos_to_target_worker = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        self.last_rel_pos_to_target_worker = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
        
        self.actions = torch.zeros((num_envs, 1), device=device, dtype=torch.long)
        self.memory = torch.zeros((num_envs, env_cfg["memory_size"], 2), device=device, dtype=torch.int)
        self.vision = torch.zeros((num_envs, 3, 3), device=device, dtype=torch.uint8)
        self.vision_range = env_cfg["vision_range"]
        
        # time
        self.episode_length_buf = torch.zeros((self.num_envs,1), device=self.device, dtype=torch.int)
        self.max_episode_length = self.timeout[self.difficulty]
        
        # buffer
        self.win_condition = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.subwin_condition = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.time_condition = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.active_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        
        
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_rewardW_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float16)
            
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        
        if self.is_manager :
            
            self.manager_write_already = False

            self.manager_cfg = manager_cfg
            self.reward_manager_cfg = manager_cfg["reward_cfg"]
            self.reward_scales_manager = manager_cfg["reward_cfg"]["reward_scales"]
            self.rew_buf_manager = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            self.num_obs_manager = manager_cfg["obs_cfg"]["num_obs"]
            
            self.manager_obs = torch.zeros((self.num_envs, self.num_obs_manager), device=self.device, dtype=torch.float32)
            
            self.rel_pos_to_target_manager = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
            self.last_rel_pos_to_target_manager = torch.zeros((num_envs, 2), device=device, dtype=torch.int)
            self.memory_manager = torch.zeros((num_envs, env_cfg["memory_size"], 2), device=device, dtype=torch.int)
            
            self.paused_env = torch.full_like(self.win_condition, False, device=self.device, dtype=torch.bool)
            
            self.reward_functions_manager, self.episode_sums_manager = dict(), dict()
            for name in self.reward_scales_manager.keys():
                self.reward_scales_manager[name] *= self.dt
                self.reward_functions_manager[name] = getattr(self, "_rewardM_" + name)
                self.episode_sums_manager[name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float16)
        
            self.extras_manager = dict()  # extra information for logging
            self.extras_manager["observations"] = dict()
        
        self.load_mazes_in_cache()
        # self.load_mazes_in_idx(range(num_envs))
        
        if self.show_viewer : 
            self.setup_viewer()
            self.is_update = False
        
        # stats
        self.stats_max = 5
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.bool)
        self.stats = torch.zeros((self.num_envs, self.stats_max,), device=self.device, dtype=torch.int)
        self.stats_len = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        self.stats_regret = torch.zeros((self.num_envs, self.stats_max,), device=self.device, dtype=torch.int)
        
        self.in_init = False
        
        # print(self.mazes[0, self.pos[0][0], self.pos[0][1]])
        # print(sqrt((self.pos[0][0].item() - self.target_worker[0][0].item())**2 + (self.pos[0][1].item() - self.target_worker[0][1].item())**2))
    
    # DFS 
    def find_target_at_distance(self,idx,distance):
        self.len_path[idx] = distance
        already = set()
        to_view = [(self.pos[idx,0].item(), self.pos[idx,1].item(), 0)] # set of triplet (x,y,distance)
        quat = [(-1,0),(1,0),(0,-1),(0,1)]
        
        while to_view :
            x,y,d = to_view.pop()
            already.add((x,y))
            for (vx, vy) in quat :
                nx, ny = x + vx, y + vy
                t = (nx,ny)
                
                if self.mazes[self.env_idx_maze[idx], ny, nx] == 0 and ny >= 0 and ny < MAZE_SIZE and nx >= 0 and nx < MAZE_SIZE and t not in already:
                    if (d + 1) >= distance :
                        return (nx,ny)
                    
                    to_view.append((nx,ny,d+1))
                    
        return None
    
    @torch.compile
    def target_by_heatmap_generator(self, env_idx, min_step, max_steps):
        num_envs = len(env_idx)
        
        dist_map = torch.zeros((num_envs, 1, MAZE_SIZE, MAZE_SIZE), device=self.device, dtype=torch.float32)
        visited = torch.zeros((num_envs, 1, MAZE_SIZE, MAZE_SIZE), device=self.device, dtype=torch.float32)
        
        walkable = 1.0 - self.mazes[self.env_idx_maze[env_idx]].unsqueeze(1).float()
        
        batch_idx = torch.arange(num_envs, device=self.device)
        visited[batch_idx, 0, self.init_pos[env_idx, 1], self.init_pos[env_idx, 0]] = 1.0
        
        for i in range(1, max_steps + 1):
            expanded = F.conv2d(visited, self.heat_kernel, padding=1)
            expanded = (expanded > 0).float() 
            
            new_cells = expanded * walkable * (1.0 - visited)
            
            dist_map += new_cells * i
            
            visited += new_cells

        target_dist = torch.randint(min_step, max_steps + 1, (num_envs,), device=self.device).view(-1, 1, 1, 1)
        self.len_path[env_idx] = target_dist.view(-1).int()
        
        mask = (dist_map == target_dist.float())
        
        has_target = mask.view(num_envs, -1).any(dim=1).view(-1, 1, 1, 1)
        final_mask = torch.where(has_target, mask, dist_map > 0)

        random_scores = torch.rand(dist_map.shape, device=self.device) * final_mask
        flat_indices = random_scores.view(num_envs, -1).argmax(dim=1)

        return torch.stack([flat_indices % MAZE_SIZE, flat_indices // MAZE_SIZE], dim=1).int()
    
    def load_mazes_in_cache(self):
        lst_random_data_idx = np.random.randint(1,self.csv_size,self.num_envs)
        
        for idx in range(self.num_envs):
            random_data_idx = lst_random_data_idx[idx]
            
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
            
            inner_h = height - 1
            inner_w = width - 1
            if self.density == 0 :
                self.mazes[idx, 1:inner_h, 1:inner_w] = 0
                
            elif self.density != 1 :
                interior_zone = self.mazes[idx, 1:inner_h, 1:inner_w]
                wall_mask = (interior_zone == 1)
                
                random_vals = torch.rand(interior_zone.shape, device=self.device)
                to_break = wall_mask & (random_vals > self.density)
                
                interior_zone[to_break] = 0
            
        self.env_idx_maze[self.all_batch_idx] = torch.randint(0,self.num_envs,(self.num_envs,), device=self.device)
        
        for idx in range(self.num_envs):
            
            coords = torch.nonzero(self.mazes[self.env_idx_maze[idx]] == 0, as_tuple=False)
            coords = coords[:, [1, 0]] # swap (y,x) to (x,y)
            self.init_pos[idx] = coords[torch.randint(0,coords.shape[0],(1,)).item()]
            self.pos[idx] = self.init_pos[idx]
            self.last_pos[idx] = self.init_pos[idx]
            
            min_range = self.range_steps[self.difficulty][0]
            max_range = self.range_steps[self.difficulty][1]
            target = self.find_target_at_distance(idx, np.random.randint(min_range,max_range))
            if not target : 
                self.target_manager[idx] = coords[torch.randint(0,coords.shape[0],(1,)).item()]
            else :
                x,y = target
                self.target_manager[idx, 0], self.target_manager[idx, 1] = x,y
                
            self.mazes[idx, self.target_manager[idx,1], self.target_manager[idx,0]] = 2
            
            if self.is_manager :
                self.memory_manager[idx,:,:] = 0
                self.memory_manager[idx,0,:] = self.init_pos[idx]
            
            if not self.is_manager :
                self.target_worker = self.target_manager

            
            self.memory[idx,:,:] = 0
            self.memory[idx,0,:] = self.init_pos[idx]
            
            if self.show_viewer and idx == 0 :
                self.is_update = True

    
    
    def print_gpu_memory(self, label=""):
        # On récupère les stats en octets
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        
        # On convertit en MiB pour que ce soit lisible
        free_inside_reserved = (reserved - allocated) / 1024**2
        free_total = (total - allocated) / 1024**2
        
        print(f"--- GPU Memory {label} ---")
        print(f"Allocated: {allocated / 1024**2:.2f} MiB")
        print(f"Reserved:  {reserved / 1024**2:.2f} MiB")
        print(f"Free (total): {free_total:.2f} MiB")
        print(f"Internal Buffer (Reserved - Allocated): {free_inside_reserved:.2f} MiB")
        print("-" * 30)
    
    # def load_mazes_in_idx(self, all_env_idx):
    #     nb_parts = ceil(len(all_env_idx) / BATCH_MEMORY_CEIL)
    #     list_parts = [BATCH_MEMORY_CEIL for _ in range(nb_parts-1)] + ([len(all_env_idx) % BATCH_MEMORY_CEIL] if len(all_env_idx) % BATCH_MEMORY_CEIL != 0 else [BATCH_MEMORY_CEIL])
    #     parts = torch.split(all_env_idx, list_parts, dim=0)
        
    #     for env_idx in parts :
            
    #         # start = time.time()

    #         # Reset previous target
    #         if not self.in_init : self.mazes[self.env_idx_maze[env_idx], self.target_manager[env_idx,1], self.target_manager[env_idx,0]] = 0

    #         # Choose maze
    #         self.env_idx_maze[env_idx] = torch.randint(0,self.num_envs,(len(env_idx),), device=self.device)
    #         # self.print_gpu_memory("Avant opération critique")

    #         walkable_mask = (self.mazes[self.env_idx_maze[env_idx]] == 0).to(torch.bool)
    #         # self.print_gpu_memory("Avant opération critique")
            
    #         random_scores = torch.rand(walkable_mask.shape, device=self.device, dtype=torch.float32) * walkable_mask

    #         flat_indices = random_scores.view(len(env_idx), -1).argmax(dim=1)

    #         new_coords = torch.stack([flat_indices % MAZE_SIZE, flat_indices // MAZE_SIZE], dim=1).int()

    #         self.init_pos[env_idx] = new_coords
    #         self.pos[env_idx] = new_coords
    #         self.last_pos[env_idx] = new_coords
            
    #         # start = time.time()
    #         del walkable_mask
    #         del random_scores
    #         # gc.collect()
    #         # torch.cuda.empty_cache()
    #         # print("heat :",time.time()-start)
            
    #         min_range = self.range_steps[self.difficulty][0]
    #         max_range = self.range_steps[self.difficulty][1]
    #         self.target_manager[env_idx] = self.target_by_heatmap_generator(env_idx, min_range, max_range)
                
    #         self.mazes[self.env_idx_maze[env_idx], self.target_manager[env_idx,1], self.target_manager[env_idx,0]] = 2
            
    #         self.memory[env_idx,:,:] = 0
    #         self.memory[env_idx,0,:] = self.init_pos[env_idx]
            
    #         if not self.is_manager :
    #             self.target_worker = self.target_manager
    #         else :
    #             self.memory_manager[env_idx,:,:] = 0
    #             self.memory_manager[env_idx,0,:] = self.init_pos[env_idx]
            
    #     if self.show_viewer :
    #         self.is_update = True
                
    #         # print("Time : ",time.time() - start, " s")


    def load_mazes_in_idx(self, env_idx):
        # start = time.time()

        # Reset previous target
        if not self.in_init : self.mazes[self.env_idx_maze[env_idx], self.target_manager[env_idx,1], self.target_manager[env_idx,0]] = 0

        # Choose maze
        self.env_idx_maze[env_idx] = torch.randint(0,self.num_envs,(len(env_idx),), device=self.device)
        # self.print_gpu_memory("Avant opération critique")

        walkable_mask = (self.mazes[self.env_idx_maze[env_idx]] == 0).to(torch.bool)
        # self.print_gpu_memory("Avant opération critique")
        
        random_scores = torch.rand(walkable_mask.shape, device=self.device, dtype=torch.float32) * walkable_mask

        flat_indices = random_scores.view(len(env_idx), -1).argmax(dim=1)

        new_coords = torch.stack([flat_indices % MAZE_SIZE, flat_indices // MAZE_SIZE], dim=1).int()

        self.init_pos[env_idx] = new_coords
        self.pos[env_idx] = new_coords
        self.last_pos[env_idx] = new_coords
        
        min_range = self.range_steps[self.difficulty][0]
        max_range = self.range_steps[self.difficulty][1]
        self.target_manager[env_idx] = self.target_by_heatmap_generator(env_idx, min_range, max_range)
            
        self.mazes[self.env_idx_maze[env_idx], self.target_manager[env_idx,1], self.target_manager[env_idx,0]] = 2
        
        self.memory[env_idx,:,:] = 0
        self.memory[env_idx,0,:] = self.init_pos[env_idx]
        
        if not self.is_manager :
            self.target_worker = self.target_manager
        else :
            self.memory_manager[env_idx,:,:] = 0
            self.memory_manager[env_idx,0,:] = self.init_pos[env_idx]
            
        if self.show_viewer :
            self.is_update = True
                
            # print("Time : ",time.time() - start, " s")
            
    
    def setup_viewer(self):
        pygame.init()
        self.display = pygame.display.set_mode((CELL_SIZE*MAZE_SIZE + 200, CELL_SIZE*MAZE_SIZE))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Maze Viewer")
            
    def update_viewer(self):
        
        # Draw Maze
        # if self.is_update :
        self.is_update = False
        
        self.display.fill((0,0,0))
        
        maze = self.mazes[self.env_idx_maze[0]].cpu()
        maze[self.target_worker[0,1], self.target_worker[0,0]] = 0
        
        img = (1 - maze) * 255  # vide = blanc, mur = noir   
        rgb = np.stack([img, img, img], axis=-1).astype(np.uint8) # RGB (H,W) -> (H,W,3)  
        rgb = np.transpose(rgb, (1, 0, 2))# pygame veut (W,H,3)

        surf = pygame.surfarray.make_surface(rgb)
        
        scaled = pygame.transform.scale(surf, (MAZE_SIZE * CELL_SIZE, MAZE_SIZE * CELL_SIZE))
        self.display.blit(scaled, (0, 0))
        
        # Draw view
        view = self.vision[0].cpu()
        img = (1-view) * 255
        # print(img)
        
        rgb = np.stack([img, img, img], axis=-1).astype(np.uint8) # RGB (H,W) -> (H,W,3)  
        rgb = np.transpose(rgb, (1, 0, 2))# pygame veut (W,H,3)

        surf = pygame.surfarray.make_surface(rgb)
        
        scaled = pygame.transform.scale(surf, (3 * CELL_SIZE * 10, 3 * CELL_SIZE * 10))
        self.display.blit(scaled, (CELL_SIZE*MAZE_SIZE + 10, 50))
        pygame.draw.rect(
                    self.display,
                    (255,0,0),
                    (CELL_SIZE*MAZE_SIZE + 10 + CELL_SIZE * 10, 50 + CELL_SIZE * 10, CELL_SIZE * 10, CELL_SIZE * 10)
        )
        
        # Draw player    
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
        x,y = self.target_worker[0]
        x,y = x.item(), y.item()
        pygame.draw.rect(
                    self.display,
                    (0, 140, 142),
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )
        
        if self.is_manager :
            # Draw target
            x,y = self.target_manager[0]
            x,y = x.item(), y.item()
            pygame.draw.rect(
                        self.display,
                        (255, 30, 197),
                        (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    )

        pygame.display.flip()
    
            
    def compute_vision_patch(self):
        for y in range(-self.vision_range,self.vision_range+1):
            for x in range(-self.vision_range,self.vision_range+1):
                rx, ry = x+1, y+1
                current_x, current_y = self.pos[:,0] + x, self.pos[:,1] + y
                envs_idx = (current_x < MAZE_SIZE) & (current_x >= 0) & (current_y < MAZE_SIZE) & (current_y >= 0)
                self.vision[envs_idx,ry,rx] = self.mazes[self.env_idx_maze[envs_idx], current_y[envs_idx], current_x[envs_idx]]
                self.vision[~envs_idx,ry,rx] = 1
                
    def set_subgoal(self, subgoal, env_idx):
        self.batch_idx = torch.arange(self.num_envs, device=self.device)
        self.paused_env[self.batch_idx] = False
        new_target = self.pos[env_idx] + (subgoal * 5).int()
        
        self.target_worker[env_idx] = torch.clamp(new_target, 0, MAZE_SIZE - 1).int()
        
        self.memory_manager[env_idx,1:,:] = self.memory_manager[env_idx,:-1,:].clone()
        self.memory_manager[env_idx,0,:] = self.pos[env_idx]
        
        self.last_rel_pos_to_target_manager = self.rel_pos_to_target_manager.clone()
        self.manager_write_already = False

            
    def step(self, actions):     
        # Set difficulty
        if (self.runner.current_learning_iteration > self.it_limit) and (self.difficulty < (len(self.range_steps) - 1)):
            self.difficulty += 1 
            self.max_episode_length = self.timeout[self.difficulty]
            self.it_limit = 0
            for i in range(self.difficulty+1):
                self.it_limit += self.it_limit_list[i]

        # Move player
        self.last_pos[self.batch_idx] = self.pos[self.batch_idx]
        
        # Reset extras
        # self.extras["time_outs"][self.all_batch_idx] = False
        # if self.is_manager : self.extras_manager["time_outs"] = self.extras["time_outs"]
        
        self.original_actions = actions
        self.actions[self.batch_idx] = torch.argmax(actions[self.batch_idx], dim=1).unsqueeze(-1)
        self.active_mask[self.all_batch_idx] = True
        self.active_mask[self.batch_idx] = False
        # print(self.actions[0])
        
        # 1 case : up
        up = ((self.actions[self.all_batch_idx,0] == 0)
              & (~self.active_mask) 
              & ((self.pos[self.all_batch_idx, 1] + 1).long() < MAZE_SIZE) 
              & (self.mazes[self.env_idx_maze , (self.pos[self.all_batch_idx, 1] + 1).long(), self.pos[self.all_batch_idx, 0].long()] != 1))
        
        idx = up.nonzero(as_tuple=True)[0]
        self.pos[idx, 1] += 1
        
        # -------------------------
        
        # 2 case : down
        down = ((self.actions[self.all_batch_idx,0] == 1) 
              & (~self.active_mask)
              & ((self.pos[self.all_batch_idx, 1] - 1).long() >= 0) 
              & (self.mazes[self.env_idx_maze, (self.pos[self.all_batch_idx, 1] - 1).long(), self.pos[self.all_batch_idx, 0].long()] != 1))
        
        idx = down.nonzero(as_tuple=True)[0]
        self.pos[idx, 1] -= 1
        
        # -------------------------
        
        # 3 case : right
        right = ((self.actions[self.all_batch_idx,0] == 2) 
              & (~self.active_mask)
              & ((self.pos[self.all_batch_idx, 0] + 1).long() < MAZE_SIZE) 
              & (self.mazes[self.env_idx_maze, self.pos[self.all_batch_idx, 1].long(), (self.pos[self.all_batch_idx, 0] + 1).long()] != 1))
        
        idx = right.nonzero(as_tuple=True)[0]
        self.pos[idx, 0] += 1
        
        # -------------------------
        
        # 4 case : left
        left = ((self.actions[self.all_batch_idx,0] == 3) 
              & (~self.active_mask)
              & ((self.pos[self.all_batch_idx, 0] - 1).long() >= 0) 
              & (self.mazes[self.env_idx_maze, self.pos[self.all_batch_idx, 1].long(), (self.pos[self.all_batch_idx, 0] - 1).long()] != 1))
        
        idx = left.nonzero(as_tuple=True)[0]
        self.pos[idx, 0] -= 1
        
        # --------- Update buffer -----------
        
        self.rel_pos_to_target_worker = self.target_worker - self.pos
        self.last_rel_pos_to_target_worker = self.target_worker - self.last_pos
        
        if self.is_manager :
            self.rel_pos_to_target_manager = self.target_manager - self.pos
            
        self.episode_length_buf[self.batch_idx,0] += 1
        
        # ------------------------------------

        # print("worker :", self.target_worker[0], "manager :", self.target_manager[0])
        self.subwin_condition = (self.pos == self.target_worker).all(dim=1)
        self.win_condition = (self.pos == self.target_manager).all(dim=1)
        self.time_condition = (self.episode_length_buf[self.all_batch_idx,0] > self.max_episode_length)
        
        self.reset_buf = self.time_condition | self.win_condition
        self.trigger_buf = self.reset_buf | self.subwin_condition
        
        # self.extras["time_outs"][self.time_condition] = True
        # if self.is_manager : self.extras_manager["time_outs"] = self.extras["time_outs"]
        
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # -- write data for report --
        if not self.write_already and self.runner.current_learning_iteration % 10 == 0:
            
            if self.is_manager :
                if not self.manager_write_already :
                    self.write_already = True
                    self.manager_write_already = True
                    w, f, r = self.stats_game()
                    min_range = self.range_steps[self.difficulty][0]
                    max_range = self.range_steps[self.difficulty][1]
                    self.writer_data.writerow([self.density, self.id, self.runner.current_learning_iteration, f"{min_range}-{max_range}", w, r])
            else :
                self.write_already = True
                w, f, r = self.stats_game()
                min_range = self.range_steps[self.difficulty][0]
                max_range = self.range_steps[self.difficulty][1]
                self.writer_data.writerow([self.density, self.id, self.runner.current_learning_iteration, f"{min_range}-{max_range}", w, r])
        
        elif self.runner.current_learning_iteration % 10 != 0 :
            self.write_already = False

        # Pause all "finish" envs
        if self.is_manager : 
            self.paused_env[self.trigger_buf.nonzero(as_tuple=False).flatten()] = True
            self.batch_idx = (~self.paused_env).nonzero().flatten()
        
        # ----------------------------------
        
        # compute vision
        self.compute_vision_patch()
        vision_flat = self.vision.reshape(self.num_envs,-1)

        # roll up the memory
        self.memory[self.batch_idx,1:,:] = self.memory[self.batch_idx,:-1,:].clone()
        self.memory[self.batch_idx,0,:] = self.pos[self.batch_idx]
        memory_flat = self.memory.reshape(self.num_envs,-1)
        
        if self.is_manager : memory_manager_flat = self.memory_manager.reshape(self.num_envs,-1)
        
        # ---------- Update viewer ------------
        
        if self.show_viewer : self.update_viewer()
        
        # -----------------------------------
        
        # compute reward Worker
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        # compute reward Manager
        if self.is_manager :
            self.rew_buf_manager[:] = 0.0
            for name, reward_func in self.reward_functions_manager.items():
                rew = reward_func() * self.reward_scales_manager[name]
                self.rew_buf_manager += rew
                self.episode_sums_manager[name] += rew

        # -----------------------------------

        
        # observations normalized (-1,1)
        #  --- Worker --
        self.obs_buf = torch.cat(
            [
                (self.pos.float() / (MAZE_SIZE-1)) * 2 - 1, # 2
                (self.rel_pos_to_target_worker.float() / (MAZE_SIZE-1)) * 2 - 1, # 2
                vision_flat.float()/2, # 9 X/2
                (memory_flat.float() / (MAZE_SIZE-1)) * 2 - 1, # 20
                self.actions.float()/3 * 2 - 1,  #1
                self.episode_length_buf.float()/self.max_episode_length, # 1
            ],
            axis=-1,
        )
        
        self.extras["observations"]["critic"] = self.obs_buf
        
        #  -- Manager --
        if self.is_manager :
            self.manager_obs = torch.cat(
                [
                    (self.pos.float() / (MAZE_SIZE-1))* 2 - 1, # 2
                    (self.rel_pos_to_target_manager.float() / (MAZE_SIZE-1))* 2 - 1, # 2
                    (self.rel_pos_to_target_worker.float() / (MAZE_SIZE-1))* 2 - 1, #2
                    (memory_manager_flat.float() / (MAZE_SIZE-1))* 2 - 1, # 20
                    (memory_flat.float() / (MAZE_SIZE-1))* 2 - 1, # 20
                    self.episode_length_buf.float()/self.max_episode_length, # 1
                ],
                axis=-1,
            )
            
            self.extras_manager["observations"]["critic"] = self.manager_obs
            
            return self.obs_buf, self.rew_buf, self.reset_buf, self.extras, self.manager_obs, self.rew_buf_manager, self.trigger_buf, self.extras_manager 
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras    
        
    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf 
        return self.obs_buf, self.extras
    
    def get_observations_manager(self):
    
        self.rel_pos_to_target_manager = self.target_manager - self.pos
        
        memory_manager_flat = self.memory_manager.reshape(self.num_envs,-1)
        memory_flat = self.memory.reshape(self.num_envs,-1)
        
        self.manager_obs = torch.cat(
            [
                (self.pos.float() / (MAZE_SIZE-1))* 2 - 1, # 2
                (self.rel_pos_to_target_manager.float() / (MAZE_SIZE-1))* 2 - 1, # 2
                (self.rel_pos_to_target_worker.float() / (MAZE_SIZE-1))* 2 - 1, #2
                (memory_manager_flat.float() / (MAZE_SIZE-1))* 2 - 1, # 20
                (memory_flat.float() / (MAZE_SIZE-1))* 2 - 1, # 20
                self.episode_length_buf.float()/self.max_episode_length, # 1
            ],
            axis=-1,
        )
        
        self.extras_manager["observations"]["critic"] = self.manager_obs 
        return self.manager_obs, self.extras_manager

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.max_episode_length
            )
            self.episode_sums[key][envs_idx] = 0.0
            
        # fill extras manager
        if self.is_manager :
            self.extras_manager["episode"] = {}
            for key in self.episode_sums_manager.keys():
                self.extras_manager["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums_manager[key][envs_idx]).item() / self.max_episode_length
                )
                self.episode_sums_manager[key][envs_idx] = 0.0
                
        # stats

        # == - win rate & regret - ==

        resets = self.reset_buf.bool()
        over = resets & (self.stats_len >= self.stats_max)
        under = resets & (self.stats_len < self.stats_max)

        if over.any():
            self.stats[over, :-1] = self.stats[over, 1:].clone()
            self.stats_regret[over, :-1] = self.stats_regret[over, 1:].clone()

        if under.any():
            self.stats_len[under] += 1

        insertion_idx = (self.stats_len - 1).clamp(max=(self.stats_max-1))

        win_mask = resets & self.win_condition
        time_mask = resets & self.time_condition

        if win_mask.any():
            self.stats[win_mask, insertion_idx[win_mask]] = 1
            self.stats_regret[win_mask, insertion_idx[win_mask]] = self.episode_length_buf[win_mask].squeeze() - self.len_path[win_mask]

        if time_mask.any():
            self.stats[time_mask, insertion_idx[time_mask]] = 0

        # ===========================

        # - reset time -
        self.episode_length_buf[envs_idx,0] = 0

        # start = time.time()
        self.load_mazes_in_idx(envs_idx)
        # print(time.time() - start)
        

    def stats_game(self):
        t = (self.stats_len.sum().item()) 
        w = 0
        f = 1

        if t != 0 :
            w = (self.stats[self.all_batch_idx, :].sum().item()) / t
            f = 1 - w

        valid_regret = self.stats_regret[self.stats_regret != 0].float()
        r = torch.mean(valid_regret).item() if valid_regret.numel() > 0 else 0.0
        
        # print(self.manager_obs[0])
        
        return (w,f,r)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf

    # ------------ reward functions Worker----------------

    def _rewardW_distance(self): 
        distance_rew = torch.sum(torch.square(self.last_rel_pos_to_target_worker), dim=1) - torch.sum(torch.square(self.rel_pos_to_target_worker), dim=1) 
        distance_rew = torch.clamp(distance_rew,min=-1) 
        return distance_rew
    
    def _rewardW_movement(self):
        move_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        exists = (self.memory == self.pos.unsqueeze(1)).all(dim=2).any(dim=1)
        move_rew[exists] = -1
        return move_rew
    
    def _rewardW_target(self):
        target_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        target_rew[self.subwin_condition] = (10 * (1 + (self.difficulty/(len(self.range_steps)-1))*1.50))
        return target_rew
    
    def _rewardW_timeout(self):
        timeout_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        timeout_rew[self.time_condition] = -1
        return timeout_rew
    
    # def _rewardW_already(self):
    #     already_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
    #     already_rew[~self.active_mask] = -1
    #     return already_rew
    
    
    # ------------ reward functions Manager----------------

    def _rewardM_distance(self):
        distance_rew = torch.sum(torch.square(self.last_rel_pos_to_target_manager), dim=1) - torch.sum(torch.square(self.rel_pos_to_target_manager), dim=1) 
        distance_rew = torch.clamp(distance_rew,min=-1) 
        return distance_rew
    
    def _rewardM_movement(self):
        move_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        exists = (self.memory_manager == self.pos.unsqueeze(1)).all(dim=2).any(dim=1)
        move_rew[exists] = -1
        return move_rew
    
    def _rewardM_target(self):
        target_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        target_rew[self.win_condition] = (10 * (1 + (self.difficulty/(len(self.range_steps)-1))*2))
        return target_rew
    
    # def _rewardM_subtarget(self):
    #     target_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
    #     target_rew[self.win_condition] = 8
    #     return target_rew
    
    def _rewardM_timeout(self):
        timeout_rew = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        timeout_rew[self.time_condition] = -1
        return timeout_rew
    
    
            