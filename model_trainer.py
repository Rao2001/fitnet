# model_trainer.py - FitNet Training Logic (PyTorch Implementation)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import io
import base64

class PyTorchHistory:
    def __init__(self):
        self.history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }

class FitNetTrainer:
    def __init__(self, n_samples=50000):
        self.n_samples = n_samples
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.histories = {}
        self.results = {}
        
    def generate_dataset(self):
        """Generate dataset with 50,000 samples"""
        X, y = make_classification(
            n_samples=self.n_samples, 
            n_features=20, 
            n_informative=15,
            n_redundant=5, 
            n_classes=2, 
            random_state=42
        )
        
        # Split: 70% train, 30% validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        return {
            'train_samples': self.X_train.shape[0],
            'val_samples': self.X_val.shape[0],
            'features': self.X_train.shape[1]
        }
        
    def _train_model(self, model, epochs=100, batch_size=256, weight_decay=0.0, early_stopping=False, patience=10):
        """Generic PyTorch training loop that matches Keras behavior"""
        # Convert numpy arrays to torch Tensors
        X_train_t = torch.FloatTensor(self.X_train)
        y_train_t = torch.FloatTensor(self.y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(self.X_val)
        y_val_t = torch.FloatTensor(self.y_val).unsqueeze(1)
        
        # Create dataset and loader
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        
        history = PyTorchHistory()
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        stopped_epoch = epochs
        
        for epoch in range(1, epochs + 1):
            # Training Mode
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                preds = (outputs >= 0.5).float()
                correct_train += (preds == batch_y).sum().item()
                total_train += batch_X.size(0)
                
            epoch_train_loss = train_loss / total_train
            epoch_train_acc = correct_train / total_train
            
            # Evaluation Mode
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                outputs_val = model(X_val_t)
                loss_val = criterion(outputs_val, y_val_t)
                val_loss = loss_val.item()
                
                preds_val = (outputs_val >= 0.5).float()
                correct_val = (preds_val == y_val_t).sum().item()
                total_val = y_val_t.size(0)
                
            epoch_val_acc = correct_val / total_val
            
            # Store histories
            history.history['loss'].append(epoch_train_loss)
            history.history['accuracy'].append(epoch_train_acc)
            history.history['val_loss'].append(val_loss)
            history.history['val_accuracy'].append(epoch_val_acc)
            
            # Early Stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        stopped_epoch = epoch
                        break
                        
        # Restore best weights if early stopping was triggered
        if early_stopping and best_weights is not None:
            model.load_state_dict(best_weights)
            
        return history, stopped_epoch
    
    def train_overfitting_model(self, epochs=100, batch_size=256):
        """Train overfitting model (high capacity)"""
        model = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        history, _ = self._train_model(model, epochs=epochs, batch_size=batch_size)
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        self.histories['overfitting'] = history
        self.results['overfitting'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': train_acc - val_acc,
            'diagnosis': 'OVERFITTING' if (train_acc - val_acc) > 0.05 else 'Normal'
        }
        return model, history
    
    def train_underfitting_model(self, epochs=100, batch_size=256):
        """Train underfitting model (low capacity)"""
        model = nn.Sequential(
            nn.Linear(20, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        history, _ = self._train_model(model, epochs=epochs, batch_size=batch_size)
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        self.histories['underfitting'] = history
        self.results['underfitting'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': train_acc - val_acc,
            'diagnosis': 'UNDERFITTING' if train_acc < 0.7 else 'Normal'
        }
        return model, history
    
    def train_dropout_model(self, epochs=100, batch_size=256):
        """Train with Dropout regularization"""
        model = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        history, _ = self._train_model(model, epochs=epochs, batch_size=batch_size)
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        self.histories['dropout'] = history
        self.results['dropout'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': train_acc - val_acc,
            'diagnosis': 'Improved with Dropout'
        }
        return model, history
    
    def train_l2_model(self, epochs=100, batch_size=256):
        """Train with L2 regularization"""
        model = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        history, _ = self._train_model(model, epochs=epochs, batch_size=batch_size, weight_decay=0.01)
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        self.histories['l2'] = history
        self.results['l2'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': train_acc - val_acc,
            'diagnosis': 'Improved with L2'
        }
        return model, history
    
    def train_early_stop_model(self, epochs=100, batch_size=256):
        """Train with Early Stopping"""
        model = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        history, stopped_epoch = self._train_model(model, epochs=epochs, batch_size=batch_size, early_stopping=True)
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        self.histories['early_stop'] = history
        self.results['early_stop'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': train_acc - val_acc,
            'stopped_epoch': stopped_epoch,
            'diagnosis': 'Optimized with Early Stopping'
        }
        return model, history
    
    def train_combined_model(self, epochs=100, batch_size=256):
        """Train with all techniques combined"""
        model = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        history, _ = self._train_model(model, epochs=epochs, batch_size=batch_size, weight_decay=0.01, early_stopping=True)
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        self.histories['combined'] = history
        self.results['combined'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': train_acc - val_acc,
            'diagnosis': 'BEST GENERALIZATION'
        }
        return model, history
    
    def train_all_models(self):
        """Train all 6 models"""
        print("Training Overfitting Model...")
        self.train_overfitting_model()
        
        print("Training Underfitting Model...")
        self.train_underfitting_model()
        
        print("Training Dropout Model...")
        self.train_dropout_model()
        
        print("Training L2 Model...")
        self.train_l2_model()
        
        print("Training Early Stopping Model...")
        self.train_early_stop_model()
        
        print("Training Combined Model...")
        self.train_combined_model()
        
        return self.results
    
    def get_figure_base64(self, fig):
        """Convert matplotlib figure to base64 for embedding"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_base64