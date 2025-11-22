# AI-Assisted Transmission Line Design System

## Overview

This is a **machine learning-powered transmission line design system** built with Streamlit. The application assists electrical engineers in selecting optimal conductors and insulators for high-voltage transmission lines, and estimating project costs and power losses.

The system uses two ML models:
1. **RandomForestClassifier** - Predicts conductor type (AAAC/ACSR/ACCC) and insulator type (Porcelain/Glass/Composite) based on voltage, current, temperature, and pollution levels
2. **XGBoost Regressor** - Estimates total project cost and line losses based on route length, tower count, conductor type, and regional factors

The application provides an interactive dashboard with multiple pages for data input, predictions, results visualization, and model performance metrics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit (Python web framework for data applications)
- **Layout**: Wide layout with expandable sidebar navigation
- **Pages**: Multi-page application using radio button navigation
  - Overview/Dashboard
  - Conductor & Insulation Selection
  - Cost & Loss Estimation  
  - Results & Reports
  - Model Performance

**Design Decision**: Streamlit was chosen for rapid prototyping and deployment of ML applications without requiring frontend framework expertise. The radio button navigation keeps all pages in a single file, simplifying state management.

### Backend Architecture
- **ML Training Pipeline**: Separate `train_models.py` script for offline model training
- **Model Serving**: Models loaded at application startup using `@st.cache_resource` decorator
- **State Management**: Streamlit's native `st.session_state` for cross-page data persistence

**Design Decision**: Separating training from serving allows models to be retrained independently without modifying the main application. Caching prevents redundant model loading on each user interaction.

### Machine Learning Models

#### Conductor Selection Model
- **Algorithm**: RandomForestClassifier (scikit-learn)
- **Input Features**: 
  - Voltage (66-800 kV)
  - Current (100-5000 A)
  - Temperature (-10 to 60°C)
  - Pollution level (Low/Medium/High - categorical)
- **Outputs**: Conductor type, Insulator type
- **Preprocessing**: OneHotEncoder for categorical pollution feature via ColumnTransformer
- **Training Data**: 100,000 synthetically generated samples with rule-based labels

**Design Decision**: RandomForest chosen for its interpretability and robust handling of mixed feature types. Synthetic data generation uses domain-specific rules (e.g., higher voltage requires better conductors).

#### Cost Estimation Model
- **Algorithm**: XGBRegressor wrapped in MultiOutputRegressor
- **Input Features**:
  - Route length (10-1000 km)
  - Tower count (10-500)
  - Conductor type (AAAC/ACSR/ACCC - categorical)
  - Region factor (0.8-2.0 multiplier)
- **Outputs**: Total cost, Line loss percentage
- **Preprocessing**: OneHotEncoder for conductor type
- **Training Data**: 100,000 synthetically generated samples

**Design Decision**: XGBoost selected for superior performance on regression tasks with tabular data. MultiOutputRegressor enables simultaneous prediction of cost and losses.

### Data Storage
- **Model Persistence**: Joblib serialization (`.joblib` files)
- **Application State**: In-memory via Streamlit session state
- **No Database**: Stateless application; no persistent user data storage

**Design Decision**: Since this is a calculation tool without user accounts or historical data requirements, database overhead is avoided. Models are pre-trained and loaded from disk.

### Engineering Calculations
- **Creepage Distance**: Calculated using pollution-based formula
  ```
  creepage = voltage × pollution_factor
  pollution_factors = {'Low': 20, 'Medium': 25, 'High': 31}
  ```

**Design Decision**: Real electrical engineering formula replaces placeholder logic to provide accurate insulator sizing.

## External Dependencies

### Python Libraries
- **streamlit** - Web application framework
- **pandas** - Data manipulation and model input formatting
- **numpy** - Numerical computations and synthetic data generation
- **plotly.express** - Interactive data visualizations
- **scikit-learn** - RandomForest classifier, preprocessing pipelines, ColumnTransformer, OneHotEncoder
- **xgboost** - Gradient boosting regressor for cost/loss prediction
- **joblib** - Model serialization and deserialization

### Assets
- **Logo.png** - University/institution branding image (90px width)

### Model Files (Generated)
- **conductor_model.joblib** - Trained RandomForest pipeline for material selection
- **cost_model.joblib** - Trained XGBoost pipeline for cost/loss estimation

**Note**: Models must be trained by running `train_models.py` before launching the Streamlit application.