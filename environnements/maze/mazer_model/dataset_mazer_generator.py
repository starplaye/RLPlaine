from mazelib import *
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator
from PIL import Image
from random import randrange
import csv

MIN_SIZE = 30
MAX_SIZE = 250
NB_IMAGES = 10000
PATH = "environnements/maze/mazer_model/"

def main():
    
    with open(PATH + "train_data.csv", 'w', newline='') as csvfile:
                    
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["ID", "W", "H"])
    
        for index in range(NB_IMAGES) :
            m = Maze()
            m.generator = BacktrackingGenerator(randrange(MIN_SIZE,MAX_SIZE), randrange(MIN_SIZE,MAX_SIZE))
            m.generate()
            
            h, w = len(m.grid),len(m.grid[0])
            img = Image.new('RGB', (w, h))
            for y in range(h):
                for x in range(w):
                    # print(img.size,w,h,x,y)
                    img.putpixel((x,y),(m.grid[y][x]*255, m.grid[y][x]*255, m.grid[y][x]*255))
            
            name = f"{PATH}train/{index}.png"
            img.save(name, 'PNG')
            writer.writerow([index, w ,h])
            
            print(f"=>\t{index}/{NB_IMAGES}\t<=")

if __name__ == "__main__":
    main()