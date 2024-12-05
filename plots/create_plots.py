import matplotlib.pyplot as plt 
import os
import numpy as np
import argparse
import re
import glob

argparser = argparse.ArgumentParser(description='Create plots for the training and validation loss and accuracy')
argparser.add_argument(
    '-f',
    '--file_name',
    help='Name of the file to create plots for')
args = argparser.parse_args()

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.file_name)

number_patten = re.compile(r'-?\d+\.?\d*')
with open(file_path, 'r') as f:
    lines = f.readlines()

epochs = []
last_score = []
moving_average_score = []

for line in lines:
    data = re.findall(number_patten, line)
    print(line, data)
    assert len(data) == 3
    epochs.append(int(data[0]))
    last_score.append(float(data[1]))
    moving_average_score.append(float(data[2]))

# plt.plot(epochs, last_score, label='Last Score')
plt.plot(epochs, moving_average_score, label='Moving Average Score')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.grid()
plt.legend()
file_header = args.file_name.split('.')[0]

plt.savefig(f'{file_header}.png')

plt.show()
