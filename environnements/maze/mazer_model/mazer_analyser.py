from mazelib import *
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator
from PIL import Image
from PIL import Image
import pandas as pd
import numpy as np
import os

PATH = "environnements/maze/mazer_model/"
TRAIN = "environnements/maze/mazer_model/train/"

def main():
    mur = 0
    chemin = 0
    
    csv_path = os.path.join(PATH, "train_data.csv")
    
    df = pd.read_csv(csv_path)
    size = len(df)
    
    mean_size_maze = 0
    
    for row in df.itertuples(index=False):
        image_id = str(row[0])
        w = int(row[1])
        h = int(row[2])
        img_path = os.path.join(TRAIN, f"{image_id}.png")
        
        try:
            img = Image.open(img_path).convert('L')
            img_array = np.array(img, dtype=np.uint8)
            
            unique, counts = np.unique(img_array, return_counts=True)
            d = dict(zip(unique, counts))
            
            mur += (d.get(255, 0))/(w*h)
            chemin += (d.get(0, 0))/(w*h)
            mean_size_maze += (w*h)
            
        except FileNotFoundError:
            print(f"Erreur : L'image {img_path} est introuvable.")
            continue
    
    percent_mur = (mur / size) * 100
    percent_chemin = (chemin / size) * 100
    mean_size_maze = (mean_size_maze / size)
    
    print(f"{percent_mur:.2f}% de mur // {percent_chemin:.2f}% de chemins // {int(mean_size_maze)} nombre de pixels moyens dans les labyrinthes")
    print(f"Soit ~{int(mean_size_maze*(chemin / size))} pixels de chemin possibles")

if __name__ == "__main__":
    main()