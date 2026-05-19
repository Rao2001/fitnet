# app.py - FitNet Web Application
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from model_trainer import (
    FitNetTrainer,
    parse_ipynb_file,
    find_nn_modules_in_code,
    extract_declarative_code,
    instantiate_model_from_code
)

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

import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="FitNet - Overfitting Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .diagnosis-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .overfitting {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .underfitting {
        background-color: #ffffcc;
        border-left: 5px solid #ffaa00;
    }
    .good {
        background-color: #ccffcc;
        border-left: 5px solid #00aa00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">🔍 FitNet - Overfitting & Underfitting Detective System</h1>', unsafe_allow_html=True)
st.markdown("### *Detect and Fix Overfitting/Underfitting in Neural Networks*")
st.markdown("---")

# Initialize variables to prevent NameError in conditional paths
train_button = False
train_overfit = False
train_underfit = False
train_dropout = False
train_l2 = False
train_early = False
train_combined = False
dataset_size = 50000

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("⚙️ Mode Selection")
    app_mode = st.radio(
        "Choose Mode:",
        ["🔍 FitNet Simulator", "📤 Upload & Diagnose Model"],
        index=0
    )
    
    st.markdown("---")
    
    if app_mode == "🔍 FitNet Simulator":
        st.title("⚙️ Configuration")
        
        # Dataset size selector
        dataset_size = st.select_slider(
            "📊 Dataset Size",
            options=[1000, 5000, 10000, 25000, 50000],
            value=50000,
            help="Larger dataset = More reliable results but slower training"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Models to Train")
        
        train_overfit = st.checkbox("Overfitting Model", value=True)
        train_underfit = st.checkbox("Underfitting Model", value=True)
        train_dropout = st.checkbox("Dropout Model", value=True)
        train_l2 = st.checkbox("L2 Regularization Model", value=True)
        train_early = st.checkbox("Early Stopping Model", value=True)
        train_combined = st.checkbox("Combined Model", value=True)
        
        st.markdown("---")
        st.markdown("### 📈 Training Parameters")
        epochs = st.slider("Epochs", 20, 200, 100)
        batch_size = st.select_slider("Batch Size", [64, 128, 256, 512], 256)
        
        st.markdown("---")
        st.markdown("### 📥 Custom Model Architecture (Optional)")
        uploaded_sim_model = st.file_uploader(
            "Upload Untrained PyTorch Model (`.pth`, `.pt` or `.ipynb`)", 
            type=['pth', 'pt', 'ipynb'],
            key="sim_model_uploader",
            help="Upload a PyTorch model (.pth/.pt) or Jupyter Notebook (.ipynb) to run simulator steps on your custom network!"
        )
        
        base_model = None
        if uploaded_sim_model is not None:
            file_name = uploaded_sim_model.name
            if file_name.endswith('.ipynb'):
                try:
                    # Parse notebook and extract all code
                    notebook_code = parse_ipynb_file(uploaded_sim_model)
                    # Find all classes inheriting from nn.Module
                    detected_classes = find_nn_modules_in_code(notebook_code)
                    
                    if not detected_classes:
                        st.error("❌ No PyTorch model classes (subclasses of `nn.Module`) found in the Jupyter Notebook.")
                    else:
                        st.success(f"✅ Parsed Notebook: Detected {len(detected_classes)} model class(es)")
                        
                        # Select class if multiple
                        class_names = [c['name'] for c in detected_classes]
                        if len(class_names) > 1:
                            selected_class = st.selectbox(
                                "🎯 Select Model Class to Train:",
                                options=class_names,
                                key="sim_selected_class"
                            )
                        else:
                            selected_class = class_names[0]
                            st.info(f"🏷️ Selected Model Class: `{selected_class}`")
                            
                        # Extract and compile declarative parts of notebook code
                        declarative_code = extract_declarative_code(notebook_code)
                        
                        # Try to instantiate the model dynamically
                        base_model = instantiate_model_from_code(
                            declarative_code, 
                            selected_class,
                            in_features=20,
                            out_features=1
                        )
                        st.success(f"⚡ Successfully instantiated `{selected_class}` model!")
                        
                        # Save in session state for the training execution
                        st.session_state.ipynb_declarative_code = declarative_code
                        st.session_state.ipynb_class_name = selected_class
                except Exception as e:
                    st.error(f"❌ Failed to extract model from Notebook: {str(e)}")
            else:
                # Standard .pt / .pth load
                try:
                    import torch
                    try:
                        base_model = torch.load(uploaded_sim_model, map_location=torch.device('cpu'), weights_only=False)
                    except TypeError:
                        base_model = torch.load(uploaded_sim_model, map_location=torch.device('cpu'))
                    if isinstance(base_model, dict):
                        st.error("The uploaded file is a state dict, not a full model object. Please save using torch.save(model, 'model.pth').")
                        base_model = None
                    elif not isinstance(base_model, torch.nn.Module):
                        st.error("The uploaded file is not a PyTorch model.")
                        base_model = None
                    else:
                        st.success("✅ Custom model loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load custom model: {str(e)}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "**FitNet** helps you understand overfitting and underfitting "
            "in neural networks. Train 6 different models and see how "
            "regularization techniques improve generalization."
        )
        
        # Train button
        st.markdown("---")
        train_button = st.button("🚀 START TRAINING", use_container_width=True)
    else:
        st.info(
            "**FitNet Custom Diagnostic**\n\n"
            "This mode lets you upload your own PyTorch `.pth`/`.pt` trained model "
            "and a validation `.csv` dataset. The engine will evaluate it live to "
            "diagnose whether it is overfitting, underfitting, or a good generalization fit!"
        )

# Main content area
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
    st.session_state.trained = False
    st.session_state.results = None
    st.session_state.diag_run = False
    st.session_state.diag_res = None

# Function to create plotly chart for training history
def create_training_plot(history, title):
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    fig.add_trace(
        go.Scatter(x=epochs, y=history.history['accuracy'], 
                   mode='lines', name='Train Accuracy',
                   line=dict(color='blue', width=2)),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history.history['val_accuracy'], 
                   mode='lines', name='Validation Accuracy',
                   line=dict(color='red', width=2)),
        secondary_y=False
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Epochs",
        yaxis_title="Accuracy",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

# Function to display results
def display_results(results, model_name, emoji, color):
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Train Accuracy", f"{results['train_acc']:.1%}")
        with col2:
            st.metric("🎯 Validation Accuracy", f"{results['val_acc']:.1%}")
        with col3:
            gap = results['gap']
            st.metric("📏 Gap", f"{gap:.1%}", 
                     delta="Large" if gap > 0.05 else "Small",
                     delta_color="inverse" if gap > 0.05 else "normal")
        
        # Diagnosis box
        diagnosis = results.get('diagnosis', '')
        if 'OVERFITTING' in diagnosis:
            st.markdown(f'<div class="diagnosis-box overfitting">⚠️ {emoji} {diagnosis}</div>', unsafe_allow_html=True)
        elif 'UNDERFITTING' in diagnosis:
            st.markdown(f'<div class="diagnosis-box underfitting">⚠️ {emoji} {diagnosis}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="diagnosis-box good">✅ {emoji} {diagnosis}</div>', unsafe_allow_html=True)

if app_mode == "🔍 FitNet Simulator":
    if train_button:
        with st.spinner("⏳ Generating dataset and training models... Please wait"):
            # Load base model if uploaded
            base_model = None
            if uploaded_sim_model is not None:
                if uploaded_sim_model.name.endswith('.ipynb'):
                    try:
                        declarative_code = st.session_state.get('ipynb_declarative_code')
                        selected_class = st.session_state.get('ipynb_class_name')
                        if declarative_code and selected_class:
                            base_model = instantiate_model_from_code(
                                declarative_code,
                                selected_class,
                                in_features=20,
                                out_features=1
                            )
                    except:
                        base_model = None
                else:
                    try:
                        import torch
                        try:
                            base_model = torch.load(uploaded_sim_model, map_location=torch.device('cpu'), weights_only=False)
                        except TypeError:
                            base_model = torch.load(uploaded_sim_model, map_location=torch.device('cpu'))
                        if isinstance(base_model, dict) or not isinstance(base_model, torch.nn.Module):
                            base_model = None
                    except:
                        base_model = None
                    
            # Initialize trainer
            trainer = FitNetTrainer(n_samples=dataset_size, base_model=base_model)
            
            # Generate dataset
            dataset_info = trainer.generate_dataset()
            
            # Show dataset info
            st.success(f"✅ Dataset generated: {dataset_info['train_samples']:,} training, {dataset_info['val_samples']:,} validation samples")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train selected models
            results = {}
            histories = {}
            
            models_to_train = []
            if train_overfit:
                models_to_train.append(('overfitting', 'Overfitting Model', '🔴'))
            if train_underfit:
                models_to_train.append(('underfitting', 'Underfitting Model', '🟡'))
            if train_dropout:
                models_to_train.append(('dropout', 'Dropout Model', '🟢'))
            if train_l2:
                models_to_train.append(('l2', 'L2 Regularization', '🟢'))
            if train_early:
                models_to_train.append(('early_stop', 'Early Stopping', '🟢'))
            if train_combined:
                models_to_train.append(('combined', 'Combined Model', '🏆'))
            
            for idx, (model_key, model_name, emoji) in enumerate(models_to_train):
                status_text.text(f"Training {model_name}... ({idx+1}/{len(models_to_train)})")
                
                if model_key == 'overfitting':
                    model, history = trainer.train_overfitting_model(epochs=epochs, batch_size=batch_size)
                elif model_key == 'underfitting':
                    model, history = trainer.train_underfitting_model(epochs=epochs, batch_size=batch_size)
                elif model_key == 'dropout':
                    model, history = trainer.train_dropout_model(epochs=epochs, batch_size=batch_size)
                elif model_key == 'l2':
                    model, history = trainer.train_l2_model(epochs=epochs, batch_size=batch_size)
                elif model_key == 'early_stop':
                    model, history = trainer.train_early_stop_model(epochs=epochs, batch_size=batch_size)
                elif model_key == 'combined':
                    model, history = trainer.train_combined_model(epochs=epochs, batch_size=batch_size)
                
                results[model_key] = trainer.results[model_key]
                histories[model_key] = history
                
                progress_bar.progress((idx + 1) / len(models_to_train))
            
            status_text.text("✅ Training complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            # Store in session state
            st.session_state.trainer = trainer
            st.session_state.results = results
            st.session_state.histories = histories
            st.session_state.trained = True

    # Display results if trained
    if st.session_state.trained:
        st.markdown("---")
        if st.session_state.trainer.base_model is not None:
            st.info("🔬 **Using Custom Model Architecture**: The dataset was dynamically adapted to match your model's feature size.")
        st.markdown("## 📊 Training Results")
        
        results = st.session_state.results
        histories = st.session_state.histories
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Results", "📉 Training Curves", "🎯 Comparison", "📋 Summary Table"])
        
        with tab1:
            # Display each model's results in expanders
            model_display = [
                ('overfitting', '🔴 Overfitting Model', 'Shows high train but low validation accuracy'),
                ('underfitting', '🟡 Underfitting Model', 'Shows low accuracy on both sets'),
                ('dropout', '🟢 Dropout Regularization (0.5)', 'Randomly drops 50% of neurons'),
                ('l2', '🟢 L2 Regularization (λ=0.01)', 'Penalizes large weights'),
                ('early_stop', '🟢 Early Stopping', 'Stops when validation stops improving'),
                ('combined', '🏆 Combined Approach', 'All techniques together')
            ]
            
            for model_key, title, description in model_display:
                if model_key in results:
                    with st.expander(f"{title} - {description}"):
                        display_results(results[model_key], title, "📊", "")
                        if model_key == 'early_stop' and 'stopped_epoch' in results[model_key]:
                            st.info(f"⏹️ Training stopped early at epoch {results[model_key]['stopped_epoch']} (prevented overfitting)")
        
        with tab2:
            st.markdown("### Training Progress Curves")
            
            # Create columns for plots
            cols = st.columns(2)
            plot_idx = 0
            
            for model_key, title in [('overfitting', 'Overfitting'), ('underfitting', 'Underfitting'),
                                      ('dropout', 'Dropout'), ('l2', 'L2 Regularization'),
                                      ('early_stop', 'Early Stopping'), ('combined', 'Combined')]:
                if model_key in histories:
                    fig = create_training_plot(histories[model_key], f"{title} Model")
                    with cols[plot_idx % 2]:
                        st.plotly_chart(fig, use_container_width=True)
                    plot_idx += 1
        
        with tab3:
            st.markdown("### Model Performance Comparison")
            
            # Create comparison dataframe
            comparison_data = []
            for model_key, results_data in results.items():
                comparison_data.append({
                    'Model': model_key.upper(),
                    'Train Acc': f"{results_data['train_acc']:.1%}",
                    'Val Acc': f"{results_data['val_acc']:.1%}",
                    'Gap': f"{results_data['gap']:.1%}",
                    'Diagnosis': results_data.get('diagnosis', '')
                })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Bar chart comparison using plotly
            fig = go.Figure()
            
            models = [m.upper() for m in results.keys()]
            train_accs = [results[m]['train_acc'] for m in results.keys()]
            val_accs = [results[m]['val_acc'] for m in results.keys()]
            
            fig.add_trace(go.Bar(name='Train Accuracy', x=models, y=train_accs, 
                                marker_color='blue', text=[f'{x:.1%}' for x in train_accs],
                                textposition='auto'))
            fig.add_trace(go.Bar(name='Validation Accuracy', x=models, y=val_accs,
                                marker_color='red', text=[f'{x:.1%}' for x in val_accs],
                                textposition='auto'))
            
            fig.update_layout(
                title="Train vs Validation Accuracy by Model",
                xaxis_title="Model",
                yaxis_title="Accuracy",
                barmode='group',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Gap analysis
            st.markdown("### 📏 Overfitting Gap Analysis")
            gap_data = []
            for model_key, results_data in results.items():
                gap_data.append({
                    'Model': model_key.upper(),
                    'Train-Val Gap': results_data['gap'],
                    'Status': '⚠️ Overfitting' if results_data['gap'] > 0.05 else '✅ Good'
                })
            
            gap_df = pd.DataFrame(gap_data)
            fig2 = go.Figure()
            colors = ['red' if g > 0.05 else 'green' for g in gap_df['Train-Val Gap']]
            fig2.add_trace(go.Bar(x=gap_df['Model'], y=gap_df['Train-Val Gap'],
                                 marker_color=colors,
                                 text=[f'{x:.1%}' for x in gap_df['Train-Val Gap']],
                                 textposition='auto'))
            fig2.update_layout(title="Overfitting Gap (Train - Validation)",
                              xaxis_title="Model",
                              yaxis_title="Gap",
                              height=400,
                              template='plotly_white')
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab4:
            st.markdown("## 📋 Complete Results Summary")
            
            summary_data = []
            for model_key, r in results.items():
                summary_data.append({
                    'Method': model_key.upper(),
                    'Train Accuracy': f"{r['train_acc']:.1%}",
                    'Validation Accuracy': f"{r['val_acc']:.1%}",
                    'Gap': f"{r['gap']:.1%}",
                    'Status': '⚠️ Problem' if r['gap'] > 0.05 or r['train_acc'] < 0.7 else '✅ Good',
                    'Diagnosis': r.get('diagnosis', '')
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Download button for results
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="fitnet_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Conclusion
            st.markdown("---")
            st.markdown("## 🎯 Key Insights")
            
            best_model = max(results.items(), key=lambda x: x[1]['val_acc'])
            smallest_gap = min(results.items(), key=lambda x: x[1]['gap'])
            
            st.info(f"""
            ### 📊 Key Findings:
            
            - **Best Performing Model**: {best_model[0].upper()} with {best_model[1]['val_acc']:.1%} validation accuracy
            - **Smallest Overfitting Gap**: {smallest_gap[0].upper()} (gap: {smallest_gap[1]['gap']:.1%})
            - **Dataset Size**: {dataset_size:,} samples (Train: {int(dataset_size*0.7):,}, Val: {int(dataset_size*0.3):,})
            
            ### 💡 Recommendations:
            
            1. **If you see OVERFITTING**: Use Dropout, L2 regularization, or Early Stopping
            2. **If you see UNDERFITTING**: Increase model capacity (more neurons/layers)
            3. **For best results**: Combine multiple regularization techniques
            """)
            
            st.success("🎉 **FitNet Analysis Complete!** Use these insights to build better neural networks.")

    elif not train_button:
        # Welcome screen
        st.markdown("""
        ### 👋 Welcome to FitNet!
        
        **FitNet** is an interactive tool that helps you understand and detect overfitting and underfitting in neural networks.
        
        ### 🎯 What you'll learn:
        
        - **Overfitting**: When a model memorizes training data but fails on new data
        - **Underfitting**: When a model is too simple to learn patterns
        - **Regularization**: Techniques like Dropout, L2, and Early Stopping
        - **Model Comparison**: Which techniques work best for different scenarios
        
        ### 🚀 Getting Started:
        
        1. **Adjust settings** in the sidebar (dataset size, epochs, etc.)
        2. **Select models** you want to train
        3. **Click "START TRAINING"** to begin
        4. **Explore results** in the tabs above
        
        ### 📊 Models Available:
        
        | Model | Description |
        |-------|-------------|
        | 🔴 Overfitting | 3 layers × 512 neurons (high capacity) |
        | 🟡 Underfitting | 1 layer × 4 neurons (low capacity) |
        | 🟢 Dropout | Adds dropout layers (rate 0.5) |
        | 🟢 L2 Reg | Adds L2 weight penalty |
        | 🟢 Early Stop | Stops when validation plateaus |
        | 🏆 Combined | All techniques together |
        
        ---
        
        **👉 Ready? Configure your settings in the sidebar and click START TRAINING!**
        """)
        
        # Sample visualization placeholder
        st.markdown("### 📈 What you'll see after training:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Training Curves** - Watch how accuracy improves over epochs")
        with col2:
            st.markdown("**Comparison Charts** - See which model performs best")
        
        st.info("💡 **Tip**: Start with 10,000 samples for faster training, then increase to 50,000 for more accurate results!")

else:
    # 📤 Upload & Diagnose Model
    st.markdown("---")
    st.markdown("## 📤 Upload & Diagnose Your Model")
    st.markdown("Analyze your own pre-trained PyTorch model for overfitting or underfitting on a custom validation dataset.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1. Upload Model")
        uploaded_model = st.file_uploader(
            "Upload PyTorch Model (`.pth` or `.pt`)", 
            type=['pth', 'pt'],
            help="Please ensure your model was saved using torch.save(model, 'model.pth') (not model.state_dict())"
        )
    with col2:
        st.markdown("### 2. Upload Evaluation Dataset")
        uploaded_dataset = st.file_uploader(
            "Upload Evaluation CSV Dataset (`.csv`)", 
            type=['csv'],
            help="Upload a CSV containing features and the target label column"
        )
        
    if uploaded_model is not None and uploaded_dataset is not None:
        try:
            # Read dataset
            df = pd.read_csv(uploaded_dataset)
            st.success(f"✅ CSV Loaded: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
            
            # Select target column
            columns = df.columns.tolist()
            label_col = st.selectbox(
                "🎯 Select the target label column:",
                options=columns,
                index=len(columns) - 1 # Default to last column
            )
            
            st.markdown("---")
            st.markdown("### 📈 Model Training Performance")
            train_acc = st.slider(
                "📊 Enter your Training Accuracy:",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.01,
                format="%.0f%%",
                help="We compare this against your validation accuracy to diagnose overfitting/underfitting"
            )
            
            st.markdown("---")
            run_diag = st.button("🔍 RUN DETECTIVE DIAGNOSIS", use_container_width=True)
            
            if run_diag:
                with st.spinner("🔄 Loading model and running forward passes on CSV dataset..."):
                    # We instantiate a trainer
                    from model_trainer import FitNetTrainer
                    eval_trainer = FitNetTrainer(n_samples=100) 
                    
                    try:
                        diag_res = eval_trainer.evaluate_uploaded_model(
                            model_file=uploaded_model,
                            df=df,
                            label_col=label_col,
                            train_acc=train_acc
                        )
                        
                        # Store in session state
                        st.session_state.diag_res = diag_res
                        st.session_state.diag_train_acc = train_acc
                        st.session_state.diag_label_col = label_col
                        st.session_state.diag_run = True
                    except Exception as err:
                        st.error(f"❌ Diagnostic Error: {str(err)}")
                        st.session_state.diag_run = False
                        
        except Exception as e:
            st.error(f"❌ Failed to parse the CSV file: {str(e)}")
            
    # Show custom diagnosis results if run
    if st.session_state.get('diag_run', False) and 'diag_res' in st.session_state:
        res = st.session_state.diag_res
        t_acc = st.session_state.diag_train_acc
        v_acc = res['val_acc']
        gap = res['gap']
        state = res['diagnosis_state']
        msg = res['diagnosis_msg']
        recs = res['recommendations']
        task = res['task_type']
        
        st.markdown("---")
        st.markdown("## 📊 Custom Model Diagnostic Report")
        
        # Grid of metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("📊 Model Task Type", task)
        with m_col2:
            st.metric("📈 Training Accuracy (You Entered)", f"{t_acc:.1%}")
        with m_col3:
            st.metric("🎯 Validation Accuracy (Calculated)", f"{v_acc:.1%}")
        with m_col4:
            st.metric("📏 Generalization Gap", f"{gap:.1%}",
                      delta="Large" if gap > 0.05 else "Small",
                      delta_color="inverse" if gap > 0.05 else "normal")
                      
        # Diagnostic banner
        if state == "OVERFITTING":
            st.markdown(f'<div class="diagnosis-box overfitting">{msg}</div>', unsafe_allow_html=True)
        elif state == "UNDERFITTING":
            st.markdown(f'<div class="diagnosis-box underfitting">{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="diagnosis-box good">{msg}</div>', unsafe_allow_html=True)
            
        # Recommendations
        st.markdown("### 💡 Detective Recommendations:")
        for r in recs:
            st.markdown(r)
            
        # Plotly Bar Chart Comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Accuracy',
            x=['Training Accuracy', 'Validation Accuracy'],
            y=[t_acc, v_acc],
            marker_color=['#1f77b4', '#d62728'],
            text=[f'{t_acc:.1%}', f'{v_acc:.1%}'],
            textposition='auto'
        ))
        fig.update_layout(
            title="Training vs Validation Accuracy Comparison",
            yaxis=dict(title="Accuracy", range=[0, 1.05]),
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("🎉 **Diagnosis Complete!** Follow the detective's suggestions above to improve model generalization.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "🔍 FitNet - Overfitting & Underfitting Detective System | Built with Streamlit & PyTorch"
    "</p>",
    unsafe_allow_html=True
)