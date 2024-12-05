import matplotlib.pyplot as plt 
import os
import numpy as np
import argparse
import re
import glob

plt.style.use('dark_background')
dirname = os.path.dirname(__file__)
files = ['logs_param_20000.txt', 'logs_param_leakyReLU_20000.txt', 'logs_param_residual_blocks_20000.txt']    
labels = ['CNN', 'CNN with Leaky ReLU', 'Residual Blocks']    

for i, file in enumerate(files):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)

    number_patten = re.compile(r'-?\d+\.?\d*')
    with open(file_path, 'r') as f:
        lines = f.readlines()

    epochs = []
    last_score = []
    moving_average_score = []

    for line in lines:
        data = re.findall(number_patten, line)
        # print(line, data)
        assert len(data) == 3
        epochs.append(int(data[0]))
        last_score.append(float(data[1]))
        moving_average_score.append(float(data[2]))

    epochs, moving_average_score = np.array(epochs), np.array(moving_average_score)
    maximum = np.argmax(moving_average_score)
    print(f"Maximum maving average score for {labels[i]}: {moving_average_score[maximum]} at epoch {epochs[maximum]}")
    plt.plot(epochs[:maximum], moving_average_score[:maximum], label=labels[i])
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.grid()
plt.legend()

plt.savefig(f'comparison.png', transparent=True)

plt.show()
