import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import os

# ==============================
# Load ML Models
# ==============================
@st.cache_resource
def load_conductor_model():
    return joblib.load("conductor_model.joblib")

@st.cache_resource
def load_cost_model():
    return joblib.load("cost_model.joblib")

conductor_model = load_conductor_model()
cost_model = load_cost_model()

# ==============================
# Dashboard Setup
# ==============================
st.set_page_config(
    page_title="AI-Assisted Transmission Line Design",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡",
)

# ==============================
# University Branding Header
# ==============================
header_col1, header_col2 = st.columns([0.15, 0.85])
with header_col1:
    st.image("Logo.png", width=90)
            
with header_col2:
    st.markdown("""
    <h2 style='color:#00bfff;'>End-to-End AI-Assisted Transmission Line Design System</h2>
    <p style='color:gray;'>Supervised by: <b>Dr. Supervisor</b> | Students: <b>Fares Mussa Basha </b></p>
    <hr style='border: 1px solid #333;'>
    """, unsafe_allow_html=True)

# ==============================
# Sidebar - Navigation Menu
# ==============================
menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔌 Conductor & Insulation Selection",
        "💰 Cost & Loss Estimation",
        "📊 Results & Reports",
        "📈 Model Performance",
        "🔄 Scenario Comparison",
        "🔧 Model Retraining",
    ]
)

# ==============================
# 1️⃣ Overview Page
# ==============================
if menu == "🏠 Overview":
    st.markdown("""
    ## ⚙️ Project Overview
    ### The **AI-Assisted Transmission Line Design System** automates material selection and cost-performance analysis using machine learning.

    ### **Core Modules:**
    - 🔌 *Conductor & Insulation Selection*: Uses ML to recommend optimal materials.
    - 💰 *Cost & Loss Estimation*: Predicts total cost, energy losses, and efficiency.

    **Objective:** To build an intelligent assistant that helps engineers design high-voltage transmission lines faster, cheaper, and more accurately.
    """)

# ==============================
# 2️⃣ Conductor & Insulation Selection Module
# ==============================
el_col1, el_col2 = st.columns(2)
if menu == "🔌 Conductor & Insulation Selection":
    with el_col1:
        st.subheader("Input Parameters")
        voltage = st.slider("System Voltage (kV)", 66, 800, 220)
        current = st.number_input("Expected Current (A)", 100, 5000, 800)
        temperature = st.slider("Ambient Temperature (°C)", -10, 60, 25)
        pollution = st.selectbox("Pollution Level", ["Low", "Medium", "High"])
        
        if st.button("Run Material Prediction"):
            input_df = pd.DataFrame({
                'voltage': [voltage],
                'current': [current],
                'temperature': [temperature],
                'pollution': [pollution]
            })
            
            predicted_conductor = conductor_model['conductor'].predict(input_df)[0]
            predicted_insulator = conductor_model['insulator'].predict(input_df)[0]
            
            pollution_factors = {'Low': 20, 'Medium': 25, 'High': 31}
            creepage_prediction = voltage * pollution_factors[pollution]
            
            st.session_state['conductor'] = predicted_conductor
            st.session_state['insulator'] = predicted_insulator
            st.session_state['creepage'] = creepage_prediction
            st.success("Prediction complete!")

    with el_col2:
        st.subheader("Predicted Outputs")
        if 'conductor' in st.session_state:
            st.metric("Recommended Conductor", st.session_state['conductor'])
            st.metric("Insulator Material", st.session_state['insulator'])
            st.metric("Predicted Creepage Distance (mm)", st.session_state['creepage'])
        else:
            st.info("Run the prediction to view results.")

# ==============================
# 3️⃣ Cost & Loss Estimation Module
# ==============================
if menu == "💰 Cost & Loss Estimation":
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input Parameters")
        route_length = st.number_input("Total Route Length (km)", 10, 1000, 120)
        tower_count = st.number_input("Tower Count", 10, 500, 50)
        conductor_type = st.selectbox("Conductor Type", ["AAAC", "ACSR", "ACCC"])
        region_factor = st.slider("Regional Cost Coefficient", 0.8, 2.0, 1.0)

        if st.button("Estimate Cost and Loss"):
            input_df = pd.DataFrame({
                'route_length': [route_length],
                'tower_count': [tower_count],
                'conductor_type': [conductor_type],
                'region_factor': [region_factor]
            })
            
            predictions = cost_model.predict(input_df)[0]
            predicted_total_cost = predictions[0]
            predicted_line_loss = predictions[1]
            
            st.session_state['total_cost'] = predicted_total_cost
            st.session_state['line_loss'] = predicted_line_loss
            st.success("Estimation complete!")

    with c2:
        st.subheader("Predicted Outputs")
        if 'total_cost' in st.session_state:
            st.metric("Estimated Total Cost ($)", f"{st.session_state['total_cost']:,.2f}")
            st.metric("Predicted Line Loss (MW/km)", round(st.session_state['line_loss'], 3))
        else:
            st.info("Run the estimation to view cost and losses.")

# ==============================
# 4️⃣ Results & Reports Page
# ==============================
if menu == "📊 Results & Reports":
    st.markdown("### 📄 Design Summary Report")
    if 'conductor' in st.session_state and 'total_cost' in st.session_state:
        report_df = pd.DataFrame({
            'Parameter': ['Conductor Type', 'Insulator Material', 'Creepage Distance (mm)', 'Total Cost ($)', 'Line Loss (MW/km)'],
            'Predicted Value': [
                st.session_state['conductor'],
                st.session_state['insulator'],
                st.session_state['creepage'],
                f"{st.session_state['total_cost']:,.2f}",
                round(st.session_state['line_loss'], 3)
            ]
        })
        st.table(report_df)
        
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                report_df.to_excel(writer, sheet_name='Design Report', index=False)
            excel_buffer.seek(0)
            
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_buffer,
                file_name="transmission_line_design_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()
            
            title = Paragraph("Transmission Line Design Report", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            data = [['Parameter', 'Predicted Value']]
            for _, row in report_df.iterrows():
                data.append([row['Parameter'], str(row['Predicted Value'])])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            
            doc.build(elements)
            pdf_buffer.seek(0)
            
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name="transmission_line_design_report.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("Please run both modules before generating a report.")

# ==============================
# 5️⃣ Model Performance Page
# ==============================
if menu == "📈 Model Performance":
    st.markdown("### 🔍 Model Validation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Conductor Selection Model")
        st.markdown("**Random Forest Classifier**")
        conductor_clf = conductor_model['conductor'].named_steps['classifier']
        
        preprocessor = conductor_model['conductor'].named_steps['preprocessor']
        cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(['pollution']))
        passthrough_features = ['voltage', 'current', 'temperature']
        feature_names = cat_feature_names + passthrough_features
        
        conductor_importance = conductor_clf.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': conductor_importance
        }).sort_values('Importance', ascending=False)
        
        fig1 = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title='Conductor Model Feature Importance', color='Importance',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("#### Insulator Selection Model")
        insulator_clf = conductor_model['insulator'].named_steps['classifier']
        insulator_importance = insulator_clf.feature_importances_
        
        importance_df2 = pd.DataFrame({
            'Feature': feature_names,
            'Importance': insulator_importance
        }).sort_values('Importance', ascending=False)
        
        fig2 = px.bar(importance_df2, x='Importance', y='Feature', orientation='h',
                     title='Insulator Model Feature Importance', color='Importance',
                     color_continuous_scale='Greens')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("#### Cost & Loss Estimation Model")
        st.markdown("**XGBoost Regressor**")
        
        cost_preprocessor = cost_model.named_steps['preprocessor']
        cat_cost_features = list(cost_preprocessor.named_transformers_['cat'].get_feature_names_out(['conductor_type']))
        passthrough_cost_features = ['route_length', 'tower_count', 'region_factor']
        cost_feature_names = cat_cost_features + passthrough_cost_features
        
        cost_regressor = cost_model.named_steps['regressor'].estimators_[0]
        cost_importance = cost_regressor.feature_importances_
        
        importance_df3 = pd.DataFrame({
            'Feature': cost_feature_names,
            'Importance': cost_importance
        }).sort_values('Importance', ascending=False)
        
        fig3 = px.bar(importance_df3, x='Importance', y='Feature', orientation='h',
                     title='Cost Model Feature Importance', color='Importance',
                     color_continuous_scale='Reds')
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("**Model Performance Metrics**")
        metrics_df = pd.DataFrame({
            'Model': ['Conductor RF', 'Insulator RF', 'Cost XGBoost'],
            'Accuracy/R²': ['90.0%', '93.0%', '0.9652'],
            'Estimators': [100, 100, 100]
        })
        st.dataframe(metrics_df, use_container_width=True)

# ==============================
# 6️⃣ Scenario Comparison Page
# ==============================
if menu == "🔄 Scenario Comparison":
    st.markdown("### 🔄 Compare Multiple Design Scenarios")
    st.markdown("Evaluate and compare different transmission line configurations side-by-side.")
    
    if 'scenarios' not in st.session_state:
        st.session_state['scenarios'] = []
    
    st.markdown("#### Add New Scenario")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Conductor & Insulation Parameters**")
        scenario_name = st.text_input("Scenario Name", f"Scenario {len(st.session_state['scenarios']) + 1}")
        voltage_comp = st.slider("Voltage (kV)", 66, 800, 220, key="comp_voltage")
        current_comp = st.number_input("Current (A)", 100, 5000, 800, key="comp_current")
        temperature_comp = st.slider("Temperature (°C)", -10, 60, 25, key="comp_temp")
        pollution_comp = st.selectbox("Pollution", ["Low", "Medium", "High"], key="comp_pollution")
    
    with col2:
        st.markdown("**Cost & Loss Parameters**")
        route_length_comp = st.number_input("Route Length (km)", 10, 1000, 120, key="comp_route")
        tower_count_comp = st.number_input("Tower Count", 10, 500, 50, key="comp_towers")
        conductor_type_comp = st.selectbox("Conductor Type", ["AAAC", "ACSR", "ACCC"], key="comp_conductor")
        region_factor_comp = st.slider("Region Factor", 0.8, 2.0, 1.0, key="comp_region")
    
    if st.button("➕ Add Scenario to Comparison"):
        input_conductor_df = pd.DataFrame({
            'voltage': [voltage_comp],
            'current': [current_comp],
            'temperature': [temperature_comp],
            'pollution': [pollution_comp]
        })
        
        predicted_conductor = conductor_model['conductor'].predict(input_conductor_df)[0]
        predicted_insulator = conductor_model['insulator'].predict(input_conductor_df)[0]
        pollution_factors = {'Low': 20, 'Medium': 25, 'High': 31}
        creepage_pred = voltage_comp * pollution_factors[pollution_comp]
        
        input_cost_df = pd.DataFrame({
            'route_length': [route_length_comp],
            'tower_count': [tower_count_comp],
            'conductor_type': [conductor_type_comp],
            'region_factor': [region_factor_comp]
        })
        
        predictions = cost_model.predict(input_cost_df)[0]
        
        scenario_data = {
            'Name': scenario_name,
            'Voltage (kV)': voltage_comp,
            'Current (A)': current_comp,
            'Temperature (°C)': temperature_comp,
            'Pollution': pollution_comp,
            'Conductor': predicted_conductor,
            'Insulator': predicted_insulator,
            'Creepage (mm)': creepage_pred,
            'Route Length (km)': route_length_comp,
            'Tower Count': tower_count_comp,
            'Region Factor': region_factor_comp,
            'Total Cost ($)': f"{predictions[0]:,.2f}",
            'Line Loss (MW/km)': round(predictions[1], 3)
        }
        
        st.session_state['scenarios'].append(scenario_data)
        st.success(f"✅ {scenario_name} added to comparison!")
    
    if len(st.session_state['scenarios']) > 0:
        st.markdown("#### 📊 Scenario Comparison Table")
        comparison_df = pd.DataFrame(st.session_state['scenarios'])
        st.dataframe(comparison_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Scenarios"):
                st.session_state['scenarios'] = []
                st.rerun()
        
        with col2:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                comparison_df.to_excel(writer, sheet_name='Scenario Comparison', index=False)
            excel_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Comparison (Excel)",
                data=excel_buffer,
                file_name="scenario_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("No scenarios added yet. Add your first scenario above to start comparing.")

# ==============================
# 7️⃣ Model Retraining Page
# ==============================
if menu == "🔧 Model Retraining":
    st.markdown("### 🔧 Model Retraining Interface")
    st.markdown("Update the ML models with new real-world data to improve prediction accuracy.")
    
    tab1, tab2 = st.tabs(["Conductor & Insulator Model", "Cost & Loss Model"])
    
    with tab1:
        st.markdown("#### Upload New Training Data for Conductor Selection")
        st.markdown("Upload a CSV file with columns: `voltage`, `current`, `temperature`, `pollution`, `conductor`, `insulator`")
        
        uploaded_conductor_file = st.file_uploader("Choose CSV file", type="csv", key="conductor_upload")
        
        if uploaded_conductor_file is not None:
            try:
                new_data = pd.read_csv(uploaded_conductor_file)
                st.success(f"✅ Loaded {len(new_data)} samples")
                st.dataframe(new_data.head(10))
                
                required_cols = ['voltage', 'current', 'temperature', 'pollution', 'conductor', 'insulator']
                if all(col in new_data.columns for col in required_cols):
                    if st.button("🔄 Retrain Conductor Model", key="retrain_conductor"):
                        with st.spinner("Retraining model..."):
                            from sklearn.compose import ColumnTransformer
                            from sklearn.preprocessing import OneHotEncoder
                            from sklearn.pipeline import Pipeline
                            
                            X = new_data[['voltage', 'current', 'temperature', 'pollution']]
                            y_conductor = new_data['conductor']
                            y_insulator = new_data['insulator']
                            
                            preprocessor = ColumnTransformer(
                                transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['pollution'])],
                                remainder='passthrough'
                            )
                            
                            conductor_pipeline = Pipeline([
                                ('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
                            ])
                            
                            insulator_pipeline = Pipeline([
                                ('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
                            ])
                            
                            conductor_pipeline.fit(X, y_conductor)
                            insulator_pipeline.fit(X, y_insulator)
                            
                            combined_model = {
                                'conductor': conductor_pipeline,
                                'insulator': insulator_pipeline
                            }
                            
                            joblib.dump(combined_model, 'conductor_model.joblib')
                            st.success("✅ Model retrained and saved successfully!")
                            st.info("Please restart the application to load the new model.")
                else:
                    st.error(f"Missing required columns. Expected: {required_cols}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with tab2:
        st.markdown("#### Upload New Training Data for Cost Estimation")
        st.markdown("Upload a CSV file with columns: `route_length`, `tower_count`, `conductor_type`, `region_factor`, `total_cost`, `line_loss`")
        
        uploaded_cost_file = st.file_uploader("Choose CSV file", type="csv", key="cost_upload")
        
        if uploaded_cost_file is not None:
            try:
                new_data = pd.read_csv(uploaded_cost_file)
                st.success(f"✅ Loaded {len(new_data)} samples")
                st.dataframe(new_data.head(10))
                
                required_cols = ['route_length', 'tower_count', 'conductor_type', 'region_factor', 'total_cost', 'line_loss']
                if all(col in new_data.columns for col in required_cols):
                    if st.button("🔄 Retrain Cost Model", key="retrain_cost"):
                        with st.spinner("Retraining model..."):
                            from sklearn.compose import ColumnTransformer
                            from sklearn.preprocessing import OneHotEncoder
                            from sklearn.pipeline import Pipeline
                            
                            X = new_data[['route_length', 'tower_count', 'conductor_type', 'region_factor']]
                            y = new_data[['total_cost', 'line_loss']]
                            
                            preprocessor = ColumnTransformer(
                                transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['conductor_type'])],
                                remainder='passthrough'
                            )
                            
                            pipeline = Pipeline([
                                ('preprocessor', preprocessor),
                                ('regressor', MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
                            ])
                            
                            pipeline.fit(X, y)
                            joblib.dump(pipeline, 'cost_model.joblib')
                            
                            st.success("✅ Model retrained and saved successfully!")
                            st.info("Please restart the application to load the new model.")
                else:
                    st.error(f"Missing required columns. Expected: {required_cols}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

# ==============================
# Dark Theme Styling
# ==============================
st.markdown(
    """
    <style>
        body { background-color: #0e1117; color: #fafafa; }
        .stButton>button { background-color: #00bfff; color: white; border-radius: 10px; }
        .stMetric { background-color: #1c1f26; border-radius: 10px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)
