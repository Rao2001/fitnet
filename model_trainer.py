# model_trainer.py - FitNet Training Logic (50,000 samples)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import io
import base64

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
    
    def train_overfitting_model(self):
        """Train overfitting model (high capacity)"""
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(20,)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(
            self.X_train, self.y_train, 
            validation_data=(self.X_val, self.y_val), 
            epochs=100, batch_size=256, verbose=0
        )
        
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
    
    def train_underfitting_model(self):
        """Train underfitting model (low capacity)"""
        model = Sequential()
        model.add(Dense(4, activation='relu', input_shape=(20,)))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(
            self.X_train, self.y_train, 
            validation_data=(self.X_val, self.y_val), 
            epochs=100, batch_size=256, verbose=0
        )
        
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
    
    def train_dropout_model(self):
        """Train with Dropout regularization"""
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(20,)))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(
            self.X_train, self.y_train, 
            validation_data=(self.X_val, self.y_val), 
            epochs=100, batch_size=256, verbose=0
        )
        
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
    
    def train_l2_model(self):
        """Train with L2 regularization"""
        model = Sequential()
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01), input_shape=(20,)))
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(
            self.X_train, self.y_train, 
            validation_data=(self.X_val, self.y_val), 
            epochs=100, batch_size=256, verbose=0
        )
        
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
    
    def train_early_stop_model(self):
        """Train with Early Stopping"""
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(20,)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            self.X_train, self.y_train, 
            validation_data=(self.X_val, self.y_val), 
            epochs=100, batch_size=256, callbacks=[early_stop], verbose=0
        )
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        self.histories['early_stop'] = history
        self.results['early_stop'] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': train_acc - val_acc,
            'stopped_epoch': len(history.history['loss']),
            'diagnosis': 'Optimized with Early Stopping'
        }
        return model, history
    
    def train_combined_model(self):
        """Train with all techniques combined"""
        model = Sequential()
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01), input_shape=(20,)))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            self.X_train, self.y_train, 
            validation_data=(self.X_val, self.y_val), 
            epochs=100, batch_size=256, callbacks=[early_stop], verbose=0
        )
        
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