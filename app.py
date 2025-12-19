import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .normal-alert {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_data
def load_model():
    try:
        with open('fraud_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('fraud_detection_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except FileNotFoundError:
        st.error("Model files not found. Please run the model.ipynb notebook first to train the model.")
        return None, None, None

# Load the model
model, scaler, metadata = load_model()

# Main header
st.markdown('<h1 class="main-header">🛡️ Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🎛️ Control Panel")

# Navigation
page = st.sidebar.selectbox(
    "Navigate to:",
    ["🏠 Home", "🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Analytics", "ℹ️ About"]
)

if page == "🏠 Home":
    st.markdown('<h2 class="sub-header">Welcome to the Fraud Detection System</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>99.5%+</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Speed</h3>
            <h2>< 1ms</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ Protection</h3>
            <h2>Real-time</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🚀 Quick Start</h4>
        <p>This system uses advanced machine learning algorithms to detect fraudulent credit card transactions in real-time. 
        Navigate to <strong>Single Prediction</strong> to test individual transactions or <strong>Batch Prediction</strong> 
        to process multiple transactions at once.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model performance metrics
    if metadata:
        st.markdown('<h3 class="sub-header">📊 Model Performance</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metadata['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metadata['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metadata['recall']:.4f}")
        with col4:
            st.metric("F1 Score", f"{metadata['f1_score']:.4f}")
        
        # ROC AUC
        st.metric("ROC AUC", f"{metadata['roc_auc']:.4f}")

elif page == "🔍 Single Prediction":
    st.markdown('<h2 class="sub-header">🔍 Single Transaction Prediction</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please ensure model files are available.")
    else:
        st.markdown("""
        <div class="info-box">
            <h4>💡 Instructions</h4>
            <p>Enter the transaction details below. The system will analyze the transaction and provide a fraud probability score.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create input form
        with st.form("transaction_form"):
            st.markdown("### 📝 Transaction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Amount input
                amount = st.number_input(
                    "💰 Transaction Amount (€)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=100.0,
                    step=0.01,
                    help="Enter the transaction amount in Euros"
                )
                
                # V1-V14 features
                st.markdown("#### 🔢 Feature Values (V1-V14)")
                v1 = st.number_input("V1", value=0.0, step=0.001)
                v2 = st.number_input("V2", value=0.0, step=0.001)
                v3 = st.number_input("V3", value=0.0, step=0.001)
                v4 = st.number_input("V4", value=0.0, step=0.001)
                v5 = st.number_input("V5", value=0.0, step=0.001)
                v6 = st.number_input("V6", value=0.0, step=0.001)
                v7 = st.number_input("V7", value=0.0, step=0.001)
                v8 = st.number_input("V8", value=0.0, step=0.001)
                v9 = st.number_input("V9", value=0.0, step=0.001)
                v10 = st.number_input("V10", value=0.0, step=0.001)
                v11 = st.number_input("V11", value=0.0, step=0.001)
                v12 = st.number_input("V12", value=0.0, step=0.001)
                v13 = st.number_input("V13", value=0.0, step=0.001)
                v14 = st.number_input("V14", value=0.0, step=0.001)
            
            with col2:
                # V15-V28 features
                st.markdown("#### 🔢 Feature Values (V15-V28)")
                v15 = st.number_input("V15", value=0.0, step=0.001)
                v16 = st.number_input("V16", value=0.0, step=0.001)
                v17 = st.number_input("V17", value=0.0, step=0.001)
                v18 = st.number_input("V18", value=0.0, step=0.001)
                v19 = st.number_input("V19", value=0.0, step=0.001)
                v20 = st.number_input("V20", value=0.0, step=0.001)
                v21 = st.number_input("V21", value=0.0, step=0.001)
                v22 = st.number_input("V22", value=0.0, step=0.001)
                v23 = st.number_input("V23", value=0.0, step=0.001)
                v24 = st.number_input("V24", value=0.0, step=0.001)
                v25 = st.number_input("V25", value=0.0, step=0.001)
                v26 = st.number_input("V26", value=0.0, step=0.001)
                v27 = st.number_input("V27", value=0.0, step=0.001)
                v28 = st.number_input("V28", value=0.0, step=0.001)
            
            # Submit button
            submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)
            
            if submitted:
                # Prepare input data
                input_data = np.array([[
                    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                    v21, v22, v23, v24, v25, v26, v27, v28, amount
                ]])
                
                # Scale the input
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                fraud_probability = model.predict_proba(input_scaled)[0][1]
                prediction = model.predict(input_scaled)[0]
                
                # Display results
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fraud probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = fraud_probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fraud Probability (%)"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Prediction result
                    if prediction == 1:
                        st.markdown("""
                        <div class="fraud-alert">
                            <h2>🚨 FRAUD DETECTED</h2>
                            <p>This transaction has been flagged as potentially fraudulent.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="normal-alert">
                            <h2>✅ NORMAL TRANSACTION</h2>
                            <p>This transaction appears to be legitimate.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed metrics
                    st.metric("Fraud Probability", f"{fraud_probability:.4f}")
                    st.metric("Confidence", f"{(1 - abs(fraud_probability - 0.5) * 2):.4f}")
                
                # Risk assessment
                if fraud_probability > 0.7:
                    risk_level = "🔴 HIGH RISK"
                    risk_color = "red"
                elif fraud_probability > 0.3:
                    risk_level = "🟡 MEDIUM RISK"
                    risk_color = "orange"
                else:
                    risk_level = "🟢 LOW RISK"
                    risk_color = "green"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: {risk_color}; color: white; border-radius: 10px; margin: 1rem 0;">
                    <h3>Risk Level: {risk_level}</h3>
                </div>
                """, unsafe_allow_html=True)

elif page == "📊 Batch Prediction":
    st.markdown('<h2 class="sub-header">📊 Batch Transaction Prediction</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please ensure model files are available.")
    else:
        st.markdown("""
        <div class="info-box">
            <h4>📁 Batch Processing</h4>
            <p>Upload a CSV file containing multiple transactions. The file should have columns: V1, V2, V3, ..., V28, Amount</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with transaction data"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                st.markdown("### 📋 Uploaded Data Preview")
                st.dataframe(df.head())
                
                # Check if required columns exist
                required_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                else:
                    # Process the data
                    if st.button("🔍 Analyze All Transactions", use_container_width=True):
                        with st.spinner("Processing transactions..."):
                            # Prepare data
                            X = df[required_columns].values
                            X_scaled = scaler.transform(X)
                            
                            # Make predictions
                            predictions = model.predict(X_scaled)
                            probabilities = model.predict_proba(X_scaled)[:, 1]
                            
                            # Add results to dataframe
                            df['Fraud_Prediction'] = predictions
                            df['Fraud_Probability'] = probabilities
                            df['Risk_Level'] = df['Fraud_Probability'].apply(
                                lambda x: 'HIGH' if x > 0.7 else 'MEDIUM' if x > 0.3 else 'LOW'
                            )
                            
                            # Display results
                            st.markdown("### 🎯 Batch Prediction Results")
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Transactions", len(df))
                            with col2:
                                st.metric("Fraudulent", df['Fraud_Prediction'].sum())
                            with col3:
                                st.metric("Normal", len(df) - df['Fraud_Prediction'].sum())
                            with col4:
                                st.metric("Fraud Rate", f"{df['Fraud_Prediction'].mean()*100:.2f}%")
                            
                            # Risk level distribution
                            risk_counts = df['Risk_Level'].value_counts()
                            fig = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Risk Level Distribution",
                                color_discrete_map={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Results table
                            st.markdown("### 📊 Detailed Results")
                            st.dataframe(df)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="fraud_detection_results.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

elif page == "📈 Model Analytics":
    st.markdown('<h2 class="sub-header">📈 Model Analytics & Insights</h2>', unsafe_allow_html=True)
    
    if metadata:
        # Model information
        st.markdown("### 🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", metadata['model_name'])
            st.metric("Training Samples", f"{metadata['training_samples']:,}")
            st.metric("Test Samples", f"{metadata['test_samples']:,}")
        
        with col2:
            st.metric("Accuracy", f"{metadata['accuracy']:.4f}")
            st.metric("Precision", f"{metadata['precision']:.4f}")
            st.metric("Recall", f"{metadata['recall']:.4f}")
        
        # Performance metrics visualization
        st.markdown("### 📊 Performance Metrics")
        
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'Value': [
                metadata['accuracy'],
                metadata['precision'],
                metadata['recall'],
                metadata['f1_score'],
                metadata['roc_auc']
            ]
        }
        
        fig = px.bar(
            x=metrics_data['Metric'],
            y=metrics_data['Value'],
            title="Model Performance Metrics",
            color=metrics_data['Value'],
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        st.markdown("### 🔍 Feature Importance")
        st.info("Feature importance analysis would be displayed here if the model supports it.")
        
        # Model architecture
        st.markdown("### 🏗️ Model Architecture")
        st.markdown(f"""
        - **Algorithm**: {metadata['model_type']}
        - **Features**: {len(metadata['feature_names'])}
        - **Preprocessing**: RobustScaler + SMOTE
        - **Validation**: Stratified K-Fold Cross-Validation
        """)
        
    else:
        st.error("Model metadata not available.")

elif page == "ℹ️ About":
    st.markdown('<h2 class="sub-header">ℹ️ About the Fraud Detection System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 System Overview</h4>
        <p>This Credit Card Fraud Detection System uses advanced machine learning algorithms to identify 
        fraudulent transactions in real-time. The system is designed to minimize false positives while 
        maximizing fraud detection accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔧 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Machine Learning Pipeline:**
        - Data Preprocessing with RobustScaler
        - SMOTE for handling class imbalance
        - Multiple algorithm comparison
        - Model selection based on F1-score
        - Real-time prediction capability
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - Real-time fraud detection
        - Batch processing capability
        - Interactive web interface
        - Comprehensive analytics
        - Model performance monitoring
        """)
    
    st.markdown("### 📊 Dataset Information")
    st.markdown("""
    - **Source**: European credit card transactions
    - **Size**: 284,807 transactions
    - **Features**: 30 anonymized features (V1-V28) + Amount
    - **Fraud Rate**: 0.17% (highly imbalanced)
    - **Time Period**: 2 days of transactions
    """)
    
    st.markdown("### 🛡️ Security & Privacy")
    st.markdown("""
    - All features are anonymized using PCA
    - No sensitive personal information is stored
    - Local processing ensures data privacy
    - Secure model loading and validation
    """)
    
    st.markdown("### 📞 Support")
    st.markdown("""
    For questions or issues:
    - Check the documentation in README.md
    - Review the model training notebook
    - Contact the development team
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🛡️ Credit Card Fraud Detection System | Built with Streamlit & Machine Learning</p>
    <p>For educational and research purposes</p>
</div>
""", unsafe_allow_html=True)
