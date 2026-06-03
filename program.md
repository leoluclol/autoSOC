Autoresearch: Lithium-ion Battery SOC Estimation

This is an autonomous machine learning experiment.
The repository contains a single fundamental Python script that you are allowed to modify:
train.py: Contains the model architecture (LSTM/CNN), data loading pipeline, optimizer, and training loop.
YOU HAVE NO ACCESS TO ANY OTHER FILE. All modifications must be made exclusively by writing a new, complete train.py script.

OBJECTIVE:

Your goal is to optimize a PyTorch neural network model to estimate the State of Charge (SOC) of LFP batteries, achieving the lowest possible test_mae_percent (Mean Absolute Error on the validation set), ideally all the way below 0.01.
You must design your features and architecture around these two chemical realities:
- The Flat Voltage Plateau: Between 20% and 80% SOC, the Open Circuit Voltage (OCV) curve of an LFP battery is almost perfectly flat. A change of just a few millivolts can span a 40% difference in SOC. Because of this, instantaneous voltage is a terrible predictor on its own in the mid-range. Your network must heavily rely on time-series history to "integrate" current over time.
- Severe Hysteresis: LFP batteries exhibit strong voltage hysteresis. The voltage at 50% SOC during charging is completely different from the voltage at 50% SOC during discharging, even after long relaxation periods. Your model must know the recent history of current direction to determine which "branch" of the hysteresis curve it is currently navigating.

Hardware is limited to Kaggle cloud resources (NVIDIA T4x2). You must balance model complexity to avoid Out Of Memory (OOM) errors and ensure that the training finishes within reasonable time (<30 min). Keep the model's parameters below 200K.


Modification Rules (WHAT YOU CAN DO):

Modify the architecture: You can change the whole network's architecture, introduce Attention mechanisms, change activation functions eccetera.
Optimize the Loss: You can alter the Loss however you prefer.
Hyperparameters: You can change the batch size, epoch number, weight decay, early stopping mechanism, and the learning rate scheduler's patience if you think this will make performance better.
You can introduce whatever new mechanism you think can benefit the network's performance.


Strict Rules (WHAT YOU CANNOT DO):

DO NOT destructively modify the data loading functions (process_and_split_data). The dataset path /kaggle/input/... MUST remain strictly unchanged.
DO NOT modify the data logging procedure at the end of train.py. Never modify the DualLogger class and these two lines:
sys.stdout = DualLogger("run.log")
sys.stderr = sys.stdout
DO NOT modify the printing of the final evaluation metrics.


EVERY TIME YOU ARE PROMPTED:

Analyze the logs of the previous run.
Understand why an idea failed or succeeded.
Propose a SINGLE, clear architectural or hyperparameter modification.
Return the ENTIRE train.py code updated and ready to use, enclosed in a ```python block.


You are an Autonomous AI ML Engineer. Never ask for permission to proceed. Keep iterating and experimenting until the process is manually stopped by the user.