# create_desktop_samples.py - Generate model and dataset sample files on Desktop
import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import torch
import torch.nn as nn

# 1. Locate the Windows Desktop path reliably
home = os.path.expanduser("~")
desktop_paths = [
    os.path.join(home, "Desktop"),
    os.path.join(home, "OneDrive", "Desktop"),
    os.path.join(home, "OneDrive - Personal", "Desktop")
]

desktop = None
for path in desktop_paths:
    if os.path.exists(path):
        desktop = path
        break

if desktop is None:
    # Fallback to home directory if Desktop is not found
    desktop = home
    print(f"Warning: Could not find Desktop folder. Saving files to: {desktop}")
else:
    print(f"Found Desktop directory at: {desktop}")

# 2. Define a sample PyTorch Model architecture
class CustomClassifier(nn.Module):
    def __init__(self, in_features=12, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

# Instantiate the model
model = CustomClassifier(in_features=12, out_features=1)

# Save the full model (.pth file) on the Desktop
model_path = os.path.join(desktop, "sample_model.pth")
torch.save(model, model_path)
print(f"Saved PyTorch model to: {model_path}")

# 3. Create a matching CSV classification dataset
# Generate synthetic dataset with 12 features and 500 samples
X, y = make_classification(
    n_samples=500,
    n_features=12,
    n_informative=9,
    n_redundant=3,
    n_classes=2,
    random_state=42
)

# Convert to Pandas DataFrame
columns = [f"feat_{i}" for i in range(12)]
df = pd.DataFrame(X, columns=columns)
df["target"] = y

# Save to CSV on the Desktop
csv_path = os.path.join(desktop, "sample_dataset.csv")
df.to_csv(csv_path, index=False)
print(f"Saved matching dataset CSV to: {csv_path}")

# 4. Create a sample Jupyter Notebook (.ipynb) defining a PyTorch module
notebook_content = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "\n",
                "class NotebookModel(nn.Module):\n",
                "    def __init__(self, in_features=12, out_features=1):\n",
                "        super().__init__()\n",
                "        self.layer1 = nn.Linear(in_features, 64)\n",
                "        self.relu = nn.ReLU()\n",
                "        self.dropout = nn.Dropout(0.3)\n",
                "        self.layer2 = nn.Linear(64, out_features)\n",
                "        self.sigmoid = nn.Sigmoid()\n",
                "        \n",
                "    def forward(self, x):\n",
                "        return self.sigmoid(self.layer2(self.dropout(self.relu(self.layer1(x)))))\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "# This cell contains training logs or plots that the AST filter will strip\n",
                "print('Notebook compiled! Ready for FitNet Simulator!')\n"
            ]
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

notebook_path = os.path.join(desktop, "sample_notebook.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)
print(f"Saved Jupyter Notebook model to: {notebook_path}")

print("\nAll sample files have been generated on your Desktop! You can drag-and-drop them into FitNet.")
