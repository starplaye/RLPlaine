import gymnasium as gym
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pygame

# --- Constantes ---
CELL_SIZE = 2
MAZE_SIZE = 501
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Variables Globales ---
mazes = torch.ones((1, MAZE_SIZE, MAZE_SIZE), device=device, dtype=torch.uint8)
sizes = torch.zeros((1, 2), device=device, dtype=torch.uint16)
init_pos = torch.zeros((1, 2), device=device, dtype=torch.int)
display = None # Sera initialisé par setup_viewer

heat_kernel = torch.tensor([[[
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]]], dtype=torch.float32, device=device)

def target_by_heatmap_generator(env_idx, max_steps):
    global display
    
    # Initialisation des maps (B=1, C=1, H, W)
    dist_map = torch.zeros((1, 1, MAZE_SIZE, MAZE_SIZE), device=device, dtype=torch.float32)
    visited = torch.zeros((1, 1, MAZE_SIZE, MAZE_SIZE), device=device)
    
    # Position initiale : init_pos stocke [x, y]
    start_x, start_y = init_pos[env_idx, 0].item(), init_pos[env_idx, 1].item()
    visited[0, 0, start_y, start_x] = 1.0
    
    # Masque des murs (inversé : 1 si passage possible, 0 si mur)
    # mazes[0] == 0 signifie chemin libre
    walkable_mask = (mazes[env_idx] == 0).float().unsqueeze(0).unsqueeze(0)

    for i in range(1, max_steps + 1):
        expanded = F.max_pool2d(visited, kernel_size=3, stride=1, padding=1)
        
        expanded = (expanded > 0).float() 
        
        # Nouvelles cellules : atteignables AND pas encore visitées AND pas un mur
        new_cells = expanded * (1 - visited) * walkable_mask
        
        dist_map[new_cells > 0] = float(i)
        visited = torch.clamp(visited + new_cells, 0, 1)

    # --- Affichage Pygame ---
    display.fill((0, 0, 0))
    
    # Normalisation pour le rendu (0-255)
    # On détache et on passe en CPU pour Pygame
    heatmap_cpu = dist_map[0, 0].cpu().numpy()
    maze = mazes[0].cpu().numpy()
    
    if max_steps > 0:
        img = (1 - (heatmap_cpu / max_steps)) * 255
    else:
        img = heatmap_cpu

    img[img[:,:] == 255] = 0
        
    # Création de l'image RGB
    rgb = np.stack([img, img, img], axis=-1).astype(np.uint8)
    # Transposition pour Pygame (Height, Width) -> (Width, Height)
    rgb = np.transpose(rgb, (1, 0, 2))
    rgb[:,:,1:] = 0
    rgb[start_x, start_y, :] = 0
    rgb[start_x, start_y, 1] = 255

    surf = pygame.surfarray.make_surface(rgb)
    scaled = pygame.transform.scale(surf, (MAZE_SIZE * CELL_SIZE, MAZE_SIZE * CELL_SIZE))
    display.blit(scaled, (0, 0))
    pygame.display.flip()

def load_mazes_in_cache():
    # Note : Assurez-vous que le chemin est correct
    try:
        csv_data = pd.read_csv("environnements/maze/mazer_model/train_data.csv")
        csv_size = len(csv_data)
        random_idx = np.random.randint(0, csv_size)
        
        image_id = csv_data.iloc[random_idx, 0].astype(str)
        width = csv_data.iloc[random_idx, 1].astype(int)
        height = csv_data.iloc[random_idx, 2].astype(int)
        
        sizes[0, 0], sizes[0, 1] = width, height
        
        path = f"environnements/maze/mazer_model/train/{image_id}.png"
        img = Image.open(path).convert('L')
        img_np = np.array(img, dtype=np.uint8)
        img_torch = torch.from_numpy(img_np).to(device)
        
        # On réinitialise à 1 (mur partout) puis on plaque l'image
        mazes[0] = 1
        h, w = img_torch.shape
        mazes[0, :h, :w] = (img_torch != 0).byte() 

        # Trouver une position libre (0) pour le départ
        coords = torch.nonzero(mazes[0] == 0)
        if len(coords) > 0:
            chosen = coords[torch.randint(0, len(coords), (1,)).item()]
            init_pos[0] = torch.tensor([chosen[1], chosen[0]]) # [x, y]
        else:
            print("Erreur: Aucun passage trouvé dans le labyrinthe")
            
    except FileNotFoundError:
        print("Fichiers introuvables. Vérifiez vos chemins.")

def setup_viewer():
    global display
    pygame.init()
    display = pygame.display.set_mode((CELL_SIZE * MAZE_SIZE, CELL_SIZE * MAZE_SIZE))
    pygame.display.set_caption("Maze Heatmap Viewer")

# --- Lancement ---
setup_viewer()

running = True
while running:
    load_mazes_in_cache()
    target_by_heatmap_generator(0, 60)
    pygame.time.delay(1000000)

pygame.quit()