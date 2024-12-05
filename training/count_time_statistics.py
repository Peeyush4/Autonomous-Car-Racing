import os
import glob
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
files = glob.glob(f'{dir_path}/*.out')
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.split(' ')[-1].strip()[:-1].replace("s/it", "") for line in lines]
        new_lines = []
        for line in lines:
            try: new_lines.append(float(line))
            except: pass
        print(f"File: {file}, Max: {np.max(new_lines)}, Min: {np.min(new_lines)}, Mean: {np.mean(new_lines)}, Count: {len(new_lines)}")        
        