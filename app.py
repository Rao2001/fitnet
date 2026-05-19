# app.py - FitNet Web Application
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model_trainer import FitNetTrainer
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

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
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
    st.markdown("### ℹ️ About")
    st.info(
        "**FitNet** helps you understand overfitting and underfitting "
        "in neural networks. Train 6 different models and see how "
        "regularization techniques improve generalization."
    )
    
    # Train button
    st.markdown("---")
    train_button = st.button("🚀 START TRAINING", use_container_width=True)

# Main content area
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
    st.session_state.trained = False
    st.session_state.results = None

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

if train_button:
    with st.spinner("🔄 Generating dataset and training models... Please wait (2-3 minutes)"):
        # Initialize trainer
        trainer = FitNetTrainer(n_samples=dataset_size)
        
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
                model, history = trainer.train_overfitting_model()
            elif model_key == 'underfitting':
                model, history = trainer.train_underfitting_model()
            elif model_key == 'dropout':
                model, history = trainer.train_dropout_model()
            elif model_key == 'l2':
                model, history = trainer.train_l2_model()
            elif model_key == 'early_stop':
                model, history = trainer.train_early_stop_model()
            elif model_key == 'combined':
                model, history = trainer.train_combined_model()
            
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

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "🔍 FitNet - Overfitting & Underfitting Detective System | Built with Streamlit & TensorFlow"
    "</p>",
    unsafe_allow_html=True
)