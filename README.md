# Autonomous-Car-Racing
A reinforcement learning project that implements a Proximal Policy Optimization (PPO) agent to master the `CarRacing-v2` environment from Gymnasium. The agent uses a custom actor-critic neural network for policy and value estimation and is trained using PyTorch.

---

## Features
- Implements the **PPO algorithm** for continuous control tasks.
- Customizable convolutional neural network for feature extraction.
- Supports training with:
  - Domain-randomized tracks.
  - Moving average evaluation for stable rewards.
- Modular environment wrapper for preprocessing and reward adjustments.
- Built-in logging and visualization options.

---

## **Project Structure**
```plaintext
.
├── training/              # Contains various scripts to train the agent
|   ├── training_DSP.py                     # Neural network containing Depthwise Separable Network
|   ├── training_resnet_no_weights.py       # Using ResNet without pretraining
|   ├── training_resnet.py                  # Using ResNet with ImageNet pretrained weights
|   ├── training_with_bottleneck.py         # Using Batchnorm after convolution
|   ├── training_with_leakyReLU.py          # Conventional neural network with leakyReLU
|   ├── training_with_residual_blocks.py    # Using Residual blocks 
|   ├── training.py                         # Conventional neural network
|   ├── *.sh                                # Bash files to train in UMD Zaratan HPC ()
|   └──*.out                                # Output files when trained in HPC
├── testing/               # Contains scripts to test and parameters
|   ├── param_DSP_20000/                     # Parameters for Neural network containing Depthwise Separable Network
|   ├── param_resnet_no_weights_20000/       # Parameters for Using ResNet without pretraining
|   ├── param_resnet_20000/                  # Parameters for Using ResNet with ImageNet pretrained weights
|   ├── param_with_bottleneck_20000/         # Parameters for Using Batchnorm after convolution
|   ├── param_with_leakyReLU_20000/          # Parameters for Conventional neural network with leakyReLU
|   ├── param_with_residual_blocks_20000/    # Parameters for Using Residual blocks 
|   ├── param_20000/                         # Parameters for Conventional neural network
|   ├── testing_with_leakyReLU.py                               # Training for Conventional neural network with LeakyRelU
|   ├── testing_with_residual_blocks.py                         # Training for Residual Blocks
|   └── testing.py                                              # Training for Conventional neural network
├── plots/                 # Utilities for visualization and performance tracking
|   ├── *.png                               # Plots for respective parameters
|   ├── *.py                                # Python files top analyze plots
|   └── *.txt                               # Logs for repective model
├── slurm_outputs/         # Outputs while training in Zaratan HPC
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

---

## **Setup**

### **1. Clone the Repository**
```bash
git clone https://github.com/Peeyush4/Autonomous-Car-Racing.git
cd Autonomous-Car-Racing
```

### **2. Install Dependencies**
Use a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install swig                # This is dependency to download environment
pip install gymnasium[box2d]    # This is to download car racing environment
```

---

## **Usage**

### Train the PPO agent
First, go to the `training/` folder.
Run the training script with customizable options:
```bash
python training.py 
```

#### **Key Arguments**:
- `--gamma`: Discount factor for rewards (default: `0.99`).
- `--action-repeat`: Number of frames to repeat an action (default: `8`).
- `--img-stack`: Number of stacked grayscale frames for input (default: `4`).
- `--seed`: Random seed for reproducibility (default: `0`).
- `--render`: Render the environment during training.

---

## **Zaratan cluster**

---

### Test the PPO agent
First, go to the `testing/` folder.
Run the testing script with customizable options:
```bash
python testing.py
```

#### **Key Arguments**:
- `--gamma`: Discount factor for rewards (default: `0.99`).
- `--action-repeat`: Number of frames to repeat an action (default: `8`).
- `--img-stack`: Number of stacked grayscale frames for input (default: `4`).
- `--seed`: Random seed for reproducibility (default: `0`).
- `--render`: Render the environment during training.
- `-n, --final_epoch_testing`: Test the agent with final epoch trained.

---

## **Neural Network Architecture**

### **Feature Extractor**
Convolutional layers for extracting spatial features from input frames.

### **Actor and Critic Heads**
- **Actor**: Outputs parameters (`alpha`, `beta`) of a Beta distribution for continuous action sampling.
- **Critic**: Outputs state value (`V`) for the current observation.

### **Loss Function**
Combines:
- **Clipped PPO loss** for policy updates.
- **Value loss** for state-value predictions.

---

## **Results**

### **Training Performance**
- Maximum maving average score for CNN: 624.99 at epoch 5840
- Maximum maving average score for CNN with Leaky ReLU: 678.61 at epoch 2230
- Maximum maving average score for Residual Blocks: 569.75 at epoch 2430

---

## **Future Work**
- Add recurrent layers (e.g., LSTM) for temporal context in decision-making.
- Parallel training
- Work more on different CNN architectures

---

## **Acknowledgments**
- [Gymnasium](https://gymnasium.farama.org/environments/box2d/car_racing/) for the CarRacing environment.
- [PyTorch](https://pytorch.org/) for deep learning tools.
