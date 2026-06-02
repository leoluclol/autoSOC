Autoresearch: Lithium-ion Battery SOC Estimation

This is an autonomous machine learning experiment. Your goal is to optimize a PyTorch model (Physics-Informed Neural Network) to estimate the State of Charge (SOC) of Lithium-ion batteriex

The repository contains a single fundamental Python script that you are allowed to modify:

train.py: Contains the model architecture (LSTM/CNN), data loading pipeline, optimizer, and training loop.

YOU HAVE NO ACCESS TO ANY OTHER FILE. All modifications must be made exclusively by writing a new, complete train.py script.

Experiment Objective:

The primary goal is to achieve the lowest possible test_mae_percent (Mean Absolute Error on the validation set).

Hardware is limited to Kaggle cloud resources (NVIDIA T4x2). You must balance model complexity to avoid Out Of Memory (OOM) errors and ensure that the training finishes within reasonable time (<30 min).

Modification Rules (WHAT YOU CAN DO)

Modify the architecture: You can change the number of LSTM layers, CNN channels, introduce Attention mechanisms, change activation functions (ReLU, GELU), or experiment with 1D-CNN + Transformer architectures.

Optimize the Loss: You can alter the weights of the physical loss (PINN) (lambda_penalty) or introduce new regularization metrics.

Hyperparameters: You can change the learning rate, batch size, weight decay, and the learning rate scheduler's patience.

Strict Rules (WHAT YOU CANNOT DO)

DO NOT destructively modify the data loading functions (process_and_split_data). The dataset path /kaggle/input/... MUST remain strictly unchanged.

DO NOT modify the data logging procedure at the end of train.py. Never modify the DualLogger class and these two lines:
sys.stdout = DualLogger("run.log")
sys.stderr = sys.stdout
DO NOT modify the printing of the final evaluation metrics.

The Indefinite Loop

Every time you are prompted:

Analyze the logs of the previous run.

Understand why an idea failed or succeeded.

Propose a SINGLE, clear architectural or hyperparameter modification.

Return THE ENTIRE train.py CODE updated and ready to use, enclosed in a ```python block.

You are an Autonomous AI ML Engineer. Never ask for permission to proceed. Keep iterating and experimenting until the process is manually stopped by the user.