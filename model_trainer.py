# model_trainer.py - FitNet Training Logic (PyTorch Implementation)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Definition of CustomClassifier to allow successful unpickling of Desktop sample_model.pth
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
import io
import base64
import pandas as pd
import copy

class PyTorchHistory:
    def __init__(self):
        self.history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }

class FitNetTrainer:
    def __init__(self, n_samples=50000, base_model=None):
        self.n_samples = n_samples
        self.base_model = base_model
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.histories = {}
        self.results = {}

    def get_model_in_out_features(self, model):
        """Infers the input and output features of the model"""
        in_features = 20
        out_features = 1
        
        # Traverse model modules to find first linear/conv layer
        for module in model.modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                break
        
        # Traverse in reverse to find the output features
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear):
                out_features = module.out_features
                break
                
        return in_features, out_features

    def reinitialize_weights(self, model):
        """Reinitialize weights of a model so we train from scratch"""
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def inject_dropout(self, module, p=0.5):
        """Recursively clone a model and inject Dropout layers after linear/activation layers"""
        if isinstance(module, nn.Sequential):
            new_layers = []
            for child in module:
                if isinstance(child, (nn.Sequential, nn.ModuleList)):
                    new_layers.append(self.inject_dropout(child, p))
                else:
                    new_layers.append(copy.deepcopy(child))
                    if isinstance(child, (nn.ReLU, nn.Linear, nn.Tanh, nn.ELU)):
                        # Do not add dropout after the final sigmoid/softmax or final output projection
                        new_layers.append(nn.Dropout(p))
            # Remove trailing dropout if it's placed right before the final activation/output
            if len(new_layers) > 0 and isinstance(new_layers[-1], nn.Dropout):
                new_layers.pop()
            return nn.Sequential(*new_layers)
            
        # Recursive copy for custom modules
        module_copy = copy.deepcopy(module)
        for name, child in module_copy.named_children():
            setattr(module_copy, name, self.inject_dropout(child, p))
        return module_copy
        
    def generate_dataset(self):
        """Generate dataset matching model input/output dimensions"""
        n_features = 20
        n_classes = 2
        
        if self.base_model is not None:
            in_features, out_features = self.get_model_in_out_features(self.base_model)
            n_features = in_features
            n_classes = 2 if out_features == 1 else out_features
            
        X, y = make_classification(
            n_samples=self.n_samples, 
            n_features=n_features, 
            n_informative=max(2, int(n_features * 0.75)),
            n_redundant=max(0, n_features - max(2, int(n_features * 0.75))), 
            n_classes=n_classes, 
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
        """Generic PyTorch training loop that dynamically supports binary and multiclass classification"""
        # Determine features and classification type
        in_features, out_features = self.get_model_in_out_features(model)
        is_multiclass = out_features > 1

        # Convert numpy arrays to torch Tensors
        X_train_t = torch.FloatTensor(self.X_train)
        X_val_t = torch.FloatTensor(self.X_val)
        
        if is_multiclass:
            y_train_t = torch.LongTensor(self.y_train.astype(np.int64))
            y_val_t = torch.LongTensor(self.y_val.astype(np.int64))
            criterion = nn.CrossEntropyLoss()
        else:
            y_train_t = torch.FloatTensor(self.y_train).unsqueeze(1)
            y_val_t = torch.FloatTensor(self.y_val).unsqueeze(1)
            criterion = nn.BCELoss()
        
        # Create dataset and loader
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
        
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
                
                # Handle raw logits if binary and model doesn't end with Sigmoid
                if not is_multiclass and (torch.any(outputs < 0) or torch.any(outputs > 1)):
                    outputs_for_loss = torch.sigmoid(outputs)
                else:
                    outputs_for_loss = outputs
                
                loss = criterion(outputs_for_loss, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                
                if is_multiclass:
                    _, preds = torch.max(outputs, dim=1)
                    correct_train += (preds == batch_y).sum().item()
                else:
                    preds = (outputs_for_loss >= 0.5).float()
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
                
                if not is_multiclass and (torch.any(outputs_val < 0) or torch.any(outputs_val > 1)):
                    outputs_val_for_loss = torch.sigmoid(outputs_val)
                else:
                    outputs_val_for_loss = outputs_val
                
                loss_val = criterion(outputs_val_for_loss, y_val_t)
                val_loss = loss_val.item()
                
                if is_multiclass:
                    _, preds_val = torch.max(outputs_val, dim=1)
                    correct_val += (preds_val == y_val_t).sum().item()
                else:
                    preds_val = (outputs_val_for_loss >= 0.5).float()
                    correct_val += (preds_val == y_val_t).sum().item()
                    
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
        """Train overfitting model (high capacity or baseline custom model)"""
        if self.base_model is not None:
            model = copy.deepcopy(self.base_model)
            self.reinitialize_weights(model)
        else:
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
        """Train underfitting model (low capacity model matching custom model shape)"""
        if self.base_model is not None:
            in_features, out_features = self.get_model_in_out_features(self.base_model)
            model = nn.Sequential(
                nn.Linear(in_features, 4),
                nn.ReLU(),
                nn.Linear(4, out_features),
                nn.Sigmoid() if out_features == 1 else nn.Identity()
            )
        else:
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
        if self.base_model is not None:
            model = self.inject_dropout(self.base_model, p=0.5)
            self.reinitialize_weights(model)
        else:
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
        if self.base_model is not None:
            model = copy.deepcopy(self.base_model)
            self.reinitialize_weights(model)
        else:
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
        if self.base_model is not None:
            model = copy.deepcopy(self.base_model)
            self.reinitialize_weights(model)
        else:
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
        if self.base_model is not None:
            model = self.inject_dropout(self.base_model, p=0.5)
            self.reinitialize_weights(model)
        else:
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

    def evaluate_uploaded_model(self, model_file, df, label_col, train_acc):
        """Load and evaluate pre-trained PyTorch model on custom DataFrame"""
        # Load the model safely on CPU
        try:
            try:
                model = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
            except TypeError:
                model = torch.load(model_file, map_location=torch.device('cpu'))
        except Exception as e:
            raise ValueError(f"Failed to load the PyTorch model file: {str(e)}")
            
        if isinstance(model, dict):
            raise ValueError(
                "The uploaded file is a state dictionary containing only weights, not the complete model object.\n\n"
                "💡 Please save your model using `torch.save(model, 'model.pth')` instead of `torch.save(model.state_dict(), 'model.pth')` "
                "so that the model architecture and layers are included."
            )
            
        if not isinstance(model, nn.Module):
            raise ValueError("The uploaded file did not load as a PyTorch Neural Network (`torch.nn.Module`).")

        # Separate features and target
        try:
            X = df.drop(columns=[label_col]).values
            y = df[label_col].values
        except Exception as e:
            raise ValueError(f"Failed to separate features and target from CSV. Please check that the target column '{label_col}' exists.")

        # Convert to FloatTensor
        try:
            X_t = torch.FloatTensor(X)
        except Exception as e:
            raise ValueError("Failed to convert features to PyTorch tensors. Ensure all columns in your CSV (except the label column) are purely numeric.")

        model.eval()
        with torch.no_grad():
            try:
                outputs = model(X_t)
            except Exception as e:
                raise ValueError(
                    f"Failed to run data through model: {str(e)}\n\n"
                    f"💡 This usually means your model expected a different number of input features than the {X.shape[1]} provided in your CSV dataset."
                )

        # Check output shape to detect task type
        if outputs.ndim == 1 or outputs.shape[1] == 1:
            # Binary Classification
            if torch.any(outputs < 0) or torch.any(outputs > 1):
                probs = torch.sigmoid(outputs)
            else:
                probs = outputs
            preds = (probs >= 0.5).float()
            
            y_t = torch.FloatTensor(y).unsqueeze(1)
            correct = (preds == y_t).float().sum().item()
            accuracy = correct / len(y)
            
            criterion = nn.BCELoss()
            loss = criterion(probs, y_t).item()
            task_type = "Binary Classification"
        else:
            # Multiclass Classification
            _, preds = torch.max(outputs, dim=1)
            try:
                y_t_long = torch.LongTensor(y.astype(np.int64))
            except Exception as e:
                raise ValueError("Failed to convert labels to integer tensors. Multiclass target labels must be integers.")
                
            correct = (preds == y_t_long).float().sum().item()
            accuracy = correct / len(y)
            
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y_t_long).item()
            task_type = f"Multiclass Classification ({outputs.shape[1]} classes)"

        # Gap diagnosis
        gap = train_acc - accuracy
        
        if gap > 0.05:
            diagnosis_state = "OVERFITTING"
            diagnosis_msg = "⚠️ OVERFITTING DETECTED (High Variance)"
            recommendations = [
                "Your model has a performance gap of **{:.1%}** between training and validation accuracy.".format(gap),
                "💡 **Add Dropout layers** (e.g., `nn.Dropout(0.5)`) between your linear layers to reduce co-adaptation of weights.",
                "💡 **Inject L2 Regularization (weight decay)** into your optimizer to penalize overly complex weights.",
                "💡 **Utilize Early Stopping** during training to stop before the network begins memorizing noise.",
                "💡 **Acquire more training data** or perform data augmentation to help the model generalize better."
            ]
        elif train_acc < 0.70:
            diagnosis_state = "UNDERFITTING"
            diagnosis_msg = "⚠️ UNDERFITTING DETECTED (High Bias)"
            recommendations = [
                "Your training accuracy is low (**{:.1%}**), meaning the model is too simple to learn the dataset's patterns.".format(train_acc),
                "💡 **Enhance model capacity** by adding more linear layers or increasing hidden dimension neurons.",
                "💡 **Train for more epochs** or slightly increase your optimization learning rate.",
                "💡 **Reduce regularization constraints** (such as decreasing Dropout rate or L2 weight decay coefficients) which might be too strict."
            ]
        else:
            diagnosis_state = "GOOD"
            diagnosis_msg = "✅ GOOD GENERALIZATION FIT"
            recommendations = [
                "Your model's generalization gap is very small (**{:.1%}**), showing highly stable performance on unseen data!".format(gap),
                "💡 Your model successfully learned general patterns rather than memorizing noise.",
                "💡 This model can be safely compiled and deployed to production workflows.",
                "💡 You can explore minor hyperparameter optimization (e.g. learning rates, batch sizes) to squeeze out more performance."
            ]

        return {
            'val_acc': accuracy,
            'val_loss': loss,
            'gap': gap,
            'task_type': task_type,
            'diagnosis_state': diagnosis_state,
            'diagnosis_msg': diagnosis_msg,
            'recommendations': recommendations
        }

# =====================================================================
# --- Jupyter Notebook (.ipynb) Support Helpers ---
# =====================================================================

def parse_ipynb_file(file_obj):
    """
    Parses a Jupyter Notebook (.ipynb) file object and extracts all code from code cells.
    Returns a combined string of all code cells.
    """
    import json
    try:
        # Check if it's a file-like object or a path/string
        if hasattr(file_obj, 'read'):
            # Check if read output is bytes (needs decode) or str
            content_data = file_obj.read()
            if isinstance(content_data, bytes):
                content_data = content_data.decode('utf-8')
            content = json.loads(content_data)
        else:
            with open(file_obj, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
        code_cells = []
        for cell in content.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    code_cells.append("".join(source))
                elif isinstance(source, str):
                    code_cells.append(source)
        return "\n\n# --- Code Cell ---\n\n".join(code_cells)
    except Exception as e:
        raise ValueError(f"Failed to parse Jupyter Notebook JSON format: {str(e)}")

def find_nn_modules_in_code(code_str):
    """
    Analyzes Python code using AST and returns a list of dictionaries with info
    on all classes inheriting from nn.Module or torch.nn.Module.
    """
    import ast
    try:
        tree = ast.parse(code_str)
    except Exception as e:
        raise ValueError(f"Syntax error in notebook code cells: {str(e)}")
        
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_nn_module = False
            for base in node.bases:
                # Check for direct base name (e.g., Module or nn.Module)
                if isinstance(base, ast.Name) and base.id in ('Module', 'nn.Module'):
                    is_nn_module = True
                elif isinstance(base, ast.Attribute):
                    # Check for nested attribute base name (e.g., torch.nn.Module)
                    parts = []
                    curr = base
                    while isinstance(curr, ast.Attribute):
                        parts.append(curr.attr)
                        curr = curr.value
                    if isinstance(curr, ast.Name):
                        parts.append(curr.id)
                    parts.reverse()
                    fullname = ".".join(parts)
                    if fullname in ('torch.nn.Module', 'nn.Module'):
                        is_nn_module = True
            
            if is_nn_module:
                classes.append({
                    'name': node.name,
                    'node': node,
                    'source': ast.unparse(node)
                })
                
    return classes

def extract_declarative_code(code_str):
    """
    Parses Python code and extracts only declarative parts (imports, classes, functions,
    and constant assignments), stripping away runtime execution code.
    """
    import ast
    try:
        tree = ast.parse(code_str)
    except Exception as e:
        raise ValueError(f"Syntax error in notebook code: {str(e)}")
        
    new_body = []
    for node in tree.body:
        # Keep imports, classes, functions
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            new_body.append(node)
        elif isinstance(node, ast.Assign):
            # Keep simple constant assignments (e.g., HIDDEN_DIM = 256)
            if isinstance(node.value, ast.Constant):
                new_body.append(node)
                
    new_tree = ast.Module(body=new_body, type_ignores=[])
    return ast.unparse(new_tree)

def instantiate_model_from_code(code_str, class_name, in_features=20, out_features=1):
    """
    Compiles and executes declarative code in a clean namespace, and instantiates
    the specified class, dynamically mapping constructor arguments if needed.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import inspect
    
    # Preload essential modules in namespace
    namespace = {
        'torch': torch,
        'nn': nn,
        'optim': optim,
        '__builtins__': __builtins__
    }
    
    # First, run the declarative code to define all classes and functions
    try:
        compiled = compile(code_str, '<notebook_extracted_code>', 'exec')
        exec(compiled, namespace)
    except Exception as e:
        raise ValueError(f"Failed to execute compiled notebook code definitions: {str(e)}")
        
    if class_name not in namespace:
        raise ValueError(f"Target model class '{class_name}' not found after execution.")
        
    model_class = namespace[class_name]
    
    # Get constructor signature
    try:
        sig = inspect.signature(model_class.__init__)
    except Exception:
        # No custom __init__ or default constructor
        try:
            return model_class()
        except Exception as e:
            raise ValueError(f"Failed to instantiate model '{class_name}' with default constructor: {str(e)}")
            
    # Prepare arguments to pass
    kwargs = {}
    
    in_names = {'in_features', 'input_dim', 'in_dim', 'input_size', 'num_features', 'features'}
    out_names = {'out_features', 'output_dim', 'out_dim', 'output_size', 'num_classes', 'classes'}
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
            
        has_default = param.default is not param.empty
        
        # Check standard input feature parameters
        if param_name.lower() in in_names:
            kwargs[param_name] = in_features
        # Check standard output class/feature parameters
        elif param_name.lower() in out_names:
            kwargs[param_name] = out_features
        # If mandatory parameter, try to map standard name or raise
        elif not has_default:
            if 'dropout' in param_name.lower():
                kwargs[param_name] = 0.5
            else:
                raise ValueError(
                    f"Model class '{class_name}' constructor has a mandatory parameter '{param_name}' "
                    f"with no default value that could not be automatically mapped.\n\n"
                    f"💡 Please modify your notebook class constructor to provide default values, "
                    f"e.g., `def __init__(self, {param_name}=default_value):`"
                )
                
    # Instantiate the model!
    try:
        model = model_class(**kwargs)
    except Exception as e:
        raise ValueError(f"Failed to instantiate model '{class_name}' with arguments {kwargs}: {str(e)}")
        
    if not isinstance(model, nn.Module):
        raise ValueError(f"Instantiated object of class '{class_name}' is not a subclass of `torch.nn.Module`.")
        
    return model