import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, 
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    classification_report
)

class MachineLearningApp:
    def __init__(self):

        st.set_page_config(
            page_title="ML Model Selection App", 
            page_icon=":robot_face:", 
            layout="wide"
        )
        

        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize all session state variables"""
        initial_states = {
            'data': None,
            'X': None,
            'y': None,
            'model': None,
            'scaler': None,
            'label_encoder': None,
            'problem_type': None,
            'test_size': 0.2,
            'selected_features': [],
            'target_column': None,
            'selected_model': None,
            'model_results': None
        }
        
        for key, value in initial_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def sidebar_data_upload(self):
        """Sidebar for data upload"""
        with st.sidebar:
            st.header("📊 Data Upload")
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file", 
                type=['csv', 'xlsx', 'xls']
            )
            
            return uploaded_file

    def sidebar_feature_selection(self, df):
        """Sidebar for feature and target selection"""
        with st.sidebar:
            st.header("🔍 Feature Selection")
            
            if df is None:
                st.warning("Please upload a dataset first.")
                return None, None, None
            

            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            

            selected_features = st.multiselect(
                "Select Features", 
                options=list(df.columns),
                default=numeric_cols
            )
            

            target_column = st.selectbox(
                "Select Target Column", 
                options=list(df.columns)
            )
            

            test_size = st.slider(
                "Test Set Percentage", 
                min_value=0.1, 
                max_value=0.5, 
                value=0.2, 
                step=0.05,
                help="Percentage of data to use for testing"
            )
            
            return selected_features, target_column, test_size

    def sidebar_model_selection(self, problem_type):
        """Sidebar for model selection"""
        with st.sidebar:
            st.header("🤖 Model Selection")
            
            if problem_type == 'classification':
                models = {
                    'Logistic Regression': LogisticRegression(),
                    'Decision Tree': DecisionTreeClassifier(),
                    'Random Forest': RandomForestClassifier(),
                    'SVM': SVC(),
                    'Naive Bayes (Gaussian)': GaussianNB(),
                    'Naive Bayes (Multinomial)': MultinomialNB(),
                    'K-Nearest Neighbors': KNeighborsClassifier(),
                    'Neural Network': MLPClassifier(max_iter=1000)
                }
            else:
                models = {
                    'Linear Regression': LinearRegression(),
                    'Decision Tree': DecisionTreeRegressor(),
                    'Random Forest': RandomForestRegressor(),
                    'SVR': SVR(),
                    'K-Nearest Neighbors': KNeighborsRegressor(),
                    'Neural Network': MLPRegressor(max_iter=1000)
                }
            
            selected_model = st.selectbox(
                "Choose a Model", 
                options=list(models.keys())
            )
            
            return models, selected_model

    def sidebar_prediction_input(self, selected_features):
        """Sidebar for prediction input"""
        with st.sidebar:
            st.header("🔮 Prediction Input")
            
            if st.session_state.model is None:
                st.warning("Please train a model first.")
                return None
            

            prediction_inputs = {}
            for feature in selected_features:
                prediction_inputs[feature] = st.number_input(
                    f"Enter {feature}", 
                    value=0.0,
                    step=0.1
                )
            

            if st.button("Predict"):
                return prediction_inputs
            
            return None

    def load_and_display_data(self, uploaded_file):
        """Load data and display dataset information"""
        if uploaded_file is not None:
            try:

                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                

                st.session_state.data = df
                

                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Dataset Preview")
                    st.dataframe(df.head())
                
                with col2:
                    st.subheader("📊 Dataset Information")
                    st.write(f"Total Rows: {df.shape[0]}")
                    st.write(f"Total Columns: {df.shape[1]}")
                    

                    col_types = df.dtypes.value_counts()
                    st.write("Column Types:")
                    for dtype, count in col_types.items():
                        st.text(f"{dtype}: {count} columns")
                
                return df
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return None

    def train_and_evaluate_model(self, X, y, test_size, models, selected_model_name):
        """Train and evaluate the selected model"""

        results_container = st.container()
        
        with results_container:

            X_scaled = StandardScaler().fit_transform(X)
            

            problem_type = 'classification' if y.dtype == 'object' else 'regression'
            

            label_encoder = None
            if problem_type == 'classification':
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
            else:
                y_encoded = y
            

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=test_size, random_state=42
            )
            

            model = models[selected_model_name]
            

            model.fit(X_train, y_train)
            

            y_pred = model.predict(X_test)
            

            st.header("🔬 Model Training Results")
            col1, col2 = st.columns(2)
            

            with col1:
                st.subheader("📊 Model Performance")
                if problem_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    
                    st.subheader("Classification Report")
                    report = classification_report(
                        y_test, y_pred, 
                        target_names=label_encoder.classes_ if label_encoder else None,
                        output_dict=True
                    )
                    

                    for key, value in report.items():
                        if isinstance(value, dict):
                            st.text(f"{key}:")
                            for metric, score in value.items():
                                st.text(f"  {metric}: {score:.2f}")
                
                else:

                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                    st.metric("Mean Absolute Error", f"{mae:.4f}")
                    st.metric("R² Score", f"{r2:.4f}")
            
            with col2:
                st.subheader("📈 Model Details")
                st.write(f"Selected Model: {selected_model_name}")
                st.write(f"Problem Type: {problem_type}")
                st.write(f"Test Set Size: {test_size:.0%}")
                st.write(f"Features Used: {', '.join(X.columns)}")
                st.write(f"Target Column: {y.name}")
            

            st.session_state.model = model
            st.session_state.scaler = StandardScaler().fit(X)
            st.session_state.label_encoder = label_encoder
            st.session_state.problem_type = problem_type
            st.session_state.X = X

    def make_prediction(self, prediction_inputs):
        """Make prediction on unseen data"""
        if st.session_state.model is None:
            st.error("Please train a model first.")
            return
        

        input_df = pd.DataFrame([prediction_inputs])
        

        input_scaled = st.session_state.scaler.transform(input_df)
        

        prediction = st.session_state.model.predict(input_scaled)
        

        if st.session_state.label_encoder:
            prediction = st.session_state.label_encoder.inverse_transform(prediction)
        

        st.header("🎯 Prediction Result")
        st.subheader("Input Data")
        st.dataframe(input_df)
        
        st.subheader("Predicted Value")
        st.write(prediction[0])

    def run(self):
        """Main application flow"""

        uploaded_file = self.sidebar_data_upload()
        

        st.title("🚀 Predict on Custom Data using any ML Model")
        

        df = self.load_and_display_data(uploaded_file)
        
        if df is not None:

            selected_features, target_column, test_size = self.sidebar_feature_selection(df)
            
            if selected_features and target_column:

                X = df[selected_features]
                y = df[target_column]
                

                problem_type = 'classification' if y.dtype == 'object' else 'regression'
                

                models, selected_model = self.sidebar_model_selection(problem_type)
                

                with st.sidebar:
                    if st.button("Train Model", type="primary"):

                        for key in ['model', 'scaler', 'label_encoder', 'problem_type']:
                            st.session_state[key] = None
                        

                        self.train_and_evaluate_model(
                            X, y, test_size, models, selected_model
                        )
                

                prediction_inputs = self.sidebar_prediction_input(selected_features)
                

                if prediction_inputs:
                    self.make_prediction(prediction_inputs)


if __name__ == "__main__":
    app = MachineLearningApp()
    app.run()
