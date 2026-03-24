import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# ----- 0. REPRODUCIBILITY -----
# Setting the random seed to ensure the experiment yields exactly the same results every time it is run.
torch.manual_seed(42)

# ----- 1. DATA PREPARATION -----
# Downloading and loading the MNIST dataset, then splitting it into batches for training and testing.
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ----- 2. MODEL DEFINITIONS -----

# Baseline Model: A simple logistic regression model with no hidden layers.
class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 10)
        )
    def forward(self, x):
        return self.net(x)

# Deep MLP: A deep neural network with 4 hidden layers and dropout for regularization.
class DeepMLP(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            activation,
            nn.Dropout(0.2), # Regularization to prevent overfitting

            nn.Linear(128, 128),
            activation,
            nn.Dropout(0.2),

            nn.Linear(128, 128), # Increased depth to analyze vanishing gradients
            activation,
            nn.Dropout(0.2),
            
            nn.Linear(128, 128),
            activation,
            
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# ----- 3. TRAINING AND EVALUATION LOOP -----
def train_and_evaluate(model, name, epochs=15):
    # Using SGD to clearly illustrate the 'Vanishing Gradient' flaw of Sigmoid.
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []

    # Progress bar setup
    pbar = tqdm(range(epochs), desc=f"Training {name: <30}", unit="epoch")

    for epoch in pbar:
        # Training Phase
        model.train() 
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation (Test) Phase to measure generalization
        model.eval() 
        correct = 0
        total = 0
        with torch.no_grad(): 
            for x, y in test_loader:
                pred = model(x)
                _, predicted = torch.max(pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        # Updating progress bar metrics
        pbar.set_postfix({'Train Loss': f"{avg_train_loss:.4f}", 'Test Acc': f"{accuracy:.2f}%"})

    return train_losses, test_accuracies

# ----- 4. RUNNING THE EXPERIMENTS -----
epochs_to_run = 15

# 1. Baseline Model
base_model = BaselineModel()
base_loss, base_acc = train_and_evaluate(base_model, "Baseline (Logistic Regression)", epochs=epochs_to_run)

# 2. Deep MLP Models (Varying the activation functions)
relu_model = DeepMLP(nn.ReLU())
relu_loss, relu_acc = train_and_evaluate(relu_model, "Deep MLP - ReLU", epochs=epochs_to_run)

sigmoid_model = DeepMLP(nn.Sigmoid())
sig_loss, sig_acc = train_and_evaluate(sigmoid_model, "Deep MLP - Sigmoid", epochs=epochs_to_run)

tanh_model = DeepMLP(nn.Tanh())
tanh_loss, tanh_acc = train_and_evaluate(tanh_model, "Deep MLP - Tanh", epochs=epochs_to_run)

leaky_model = DeepMLP(nn.LeakyReLU(0.01))
leaky_loss, leaky_acc = train_and_evaluate(leaky_model, "Deep MLP - LeakyReLU", epochs=epochs_to_run)

# ----- 5. FINAL SUMMARY TABLE -----
print("\n" + "="*60)
print(" 🎓 FINAL RESULTS SUMMARY (After 15 Epochs)")
print("="*60)

results = {
    "Model Architecture": [
        "Baseline (Logistic Regression)", 
        "Deep MLP - ReLU", 
        "Deep MLP - Sigmoid", 
        "Deep MLP - Tanh", 
        "Deep MLP - LeakyReLU"
    ],
    "Final Train Loss": [
        base_loss[-1], relu_loss[-1], sig_loss[-1], tanh_loss[-1], leaky_loss[-1]
    ],
    "Final Test Acc (%)": [
        base_acc[-1], relu_acc[-1], sig_acc[-1], tanh_acc[-1], leaky_acc[-1]
    ]
}

df_results = pd.DataFrame(results)
df_results["Final Train Loss"] = df_results["Final Train Loss"].round(4)
df_results["Final Test Acc (%)"] = df_results["Final Test Acc (%)"].round(2)

print(df_results.to_markdown(index=False))
print("="*60 + "\n")

# ----- 6. VISUALIZING THE RESULTS -----
plt.figure(figsize=(14, 6))

# Plot 1: Training Loss (Learning Dynamics)
plt.subplot(1, 2, 1)
plt.plot(base_loss, label="Baseline (Linear Layer)", linestyle="--", color='gray')
plt.plot(relu_loss, label="ReLU", linewidth=2)
plt.plot(sig_loss, label="Sigmoid", linewidth=2)
plt.plot(tanh_loss, label="Tanh", linewidth=2)
plt.plot(leaky_loss, label="LeakyReLU", linewidth=2)
plt.yscale("log")
plt.title("Training Loss - Vanishing Gradient Analysis")
plt.xlabel("Epoch")
plt.ylabel("Loss (Log Scale)")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

# Plot 2: Test Accuracy (Generalization Performance)
plt.subplot(1, 2, 2)
plt.plot(base_acc, label="Baseline (Linear Layer)", linestyle="--", color='gray')
plt.plot(relu_acc, label="ReLU", linewidth=2)
plt.plot(sig_acc, label="Sigmoid", linewidth=2)
plt.plot(tanh_acc, label="Tanh", linewidth=2)
plt.plot(leaky_acc, label="LeakyReLU", linewidth=2)
plt.title("Test Accuracy - Generalization")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True, alpha=0.5)

plt.tight_layout()

plt.savefig("activation_functions_comparison.png", dpi=300, bbox_inches='tight')
print("Plot successfully saved as 'activation_functions_comparison.png'!")
