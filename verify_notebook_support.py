# verify_notebook_support.py - Verify Jupyter Notebook parsing, AST safety, and training
import json
import os
import torch
import torch.nn as nn
from model_trainer import (
    FitNetTrainer,
    parse_ipynb_file,
    find_nn_modules_in_code,
    extract_declarative_code,
    instantiate_model_from_code
)

def run_notebook_verification():
    notebook_filename = "test_notebook.ipynb"
    
    # 1. Create a dummy Jupyter Notebook JSON structure
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
                    "import os\n",
                    "HIDDEN_SIZE = 128\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": [
                    "class NotebookClassifier(nn.Module):\n",
                    "    def __init__(self, in_features=20, out_features=1):\n",
                    "        super().__init__()\n",
                    "        self.linear1 = nn.Linear(in_features, HIDDEN_SIZE)\n",
                    "        self.relu = nn.ReLU()\n",
                    "        self.linear2 = nn.Linear(HIDDEN_SIZE, out_features)\n",
                    "        self.sigmoid = nn.Sigmoid()\n",
                    "        \n",
                    "    def forward(self, x):\n",
                    "        return self.sigmoid(self.linear2(self.relu(self.linear1(x))))\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# This cell contains runtime side effects that should be stripped by AST\n",
                    "print('Running code cell with side-effects!')\n",
                    "with open('nonexistent_file.txt', 'r') as f:\n",
                    "    text = f.read()\n",
                    "print(text)\n"
                ]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(notebook_filename, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
        
    print(f"Created test notebook: '{notebook_filename}'")
    
    try:
        # 2. Extract and join code cells
        print("\nStep 1: Extracting code cells...")
        raw_code = parse_ipynb_file(notebook_filename)
        print("Raw extracted code length:", len(raw_code))
        
        # 3. Find nn.Module classes in raw code
        print("\nStep 2: Detecting neural network classes in notebook...")
        classes = find_nn_modules_in_code(raw_code)
        print("Detected classes:", [c['name'] for c in classes])
        assert len(classes) == 1, "Should detect exactly 1 class"
        assert classes[0]['name'] == 'NotebookClassifier', "Should detect 'NotebookClassifier'"
        print("Class detection verified successfully!")
        
        # 4. Filter AST declarative code
        print("\nStep 3: Filtering declarative code using AST...")
        declarative_code = extract_declarative_code(raw_code)
        print("--- Filtered Declarative Code ---")
        print(declarative_code)
        print("---------------------------------")
        
        # Check that runtime file-opening side-effect is stripped
        assert "nonexistent_file.txt" not in declarative_code, "AST filter failed to strip side-effects!"
        assert "NotebookClassifier" in declarative_code, "AST filter stripped class definition!"
        print("AST safety filter verified successfully! Side-effects stripped.")
        
        # 5. Dynamically compile and instantiate the class
        print("\nStep 4: Compiling and instantiating custom Notebook model class...")
        model = instantiate_model_from_code(
            declarative_code, 
            "NotebookClassifier", 
            in_features=20, 
            out_features=1
        )
        print("Instantiated model:", model)
        assert isinstance(model, nn.Module), "Should be a PyTorch nn.Module subclass"
        print("Model instantiation verified successfully!")
        
        # 6. Run the FitNetTrainer simulator steps with the custom notebook model
        print("\nStep 5: Testing trainer pipeline with custom Notebook model...")
        trainer = FitNetTrainer(n_samples=1000, base_model=model)
        dataset_info = trainer.generate_dataset()
        print("Dataset generated:", dataset_info)
        assert dataset_info['features'] == 20, "Dataset input features should be 20"
        
        print("Running simulator overfitting training...")
        overfit_model, overfit_history = trainer.train_overfitting_model(epochs=5, batch_size=64)
        print("Overfitting model trained successfully! Val Acc:", trainer.results['overfitting']['val_acc'])
        
        print("Running simulator combined regularization training...")
        combined_model, combined_history = trainer.train_combined_model(epochs=5, batch_size=64)
        print("Combined model trained successfully! Val Acc:", trainer.results['combined']['val_acc'])
        
        print("\nALL NOTEBOOK VERIFICATION TESTS PASSED SUCCESSFULLY!")
        
    finally:
        # Cleanup
        if os.path.exists(notebook_filename):
            os.remove(notebook_filename)
            print(f"Cleaned up test notebook file: '{notebook_filename}'")

if __name__ == "__main__":
    run_notebook_verification()
