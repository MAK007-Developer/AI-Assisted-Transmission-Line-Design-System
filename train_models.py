import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib

print("=" * 60)
print("AI-Assisted Transmission Line Design - Model Training")
print("=" * 60)

# ==============================
# 1. Generate Synthetic Dataset for Conductor & Insulation Selection
# ==============================
print("\n[1/5] Generating synthetic dataset for Conductor & Insulation Selection...")
np.random.seed(42)
n_samples = 100000

voltage = np.random.randint(66, 801, n_samples)
current = np.random.randint(100, 5001, n_samples)
temperature = np.random.randint(-10, 61, n_samples)
pollution = np.random.choice(['Low', 'Medium', 'High'], n_samples)

conductor_types = ['AAAC', 'ACSR', 'ACCC']
insulator_types = ['Porcelain', 'Glass', 'Composite']

conductor = []
insulator = []

for i in range(n_samples):
    v, c, t, p = voltage[i], current[i], temperature[i], pollution[i]
    
    if v > 500:
        cond = 'ACCC' if c > 3000 else 'ACSR'
    elif v > 300:
        cond = 'ACSR' if c > 2000 else 'AAAC'
    else:
        cond = 'AAAC' if c < 2500 else 'ACSR'
    
    if p == 'High':
        ins = 'Composite'
    elif p == 'Medium':
        ins = 'Composite' if v > 400 else 'Glass'
    else:
        ins = 'Porcelain' if v < 300 else 'Glass'
    
    if t > 40 and ins == 'Porcelain':
        ins = 'Composite'
    
    conductor.append(cond)
    insulator.append(ins)

conductor_data = pd.DataFrame({
    'voltage': voltage,
    'current': current,
    'temperature': temperature,
    'pollution': pollution,
    'conductor': conductor,
    'insulator': insulator
})

conductor_data.to_csv('Conductor & Insulation Selection Data.csv', index=False)
print(f"   ✓ Dataset saved: 'Conductor & Insulation Selection Data.csv' ({n_samples:,} samples)")

# ==============================
# 2. Generate Synthetic Dataset for Cost & Loss Estimation
# ==============================
print("\n[2/5] Generating synthetic dataset for Cost & Loss Estimation...")
np.random.seed(42)

route_length = np.random.randint(10, 1001, n_samples)
tower_count = np.random.randint(10, 501, n_samples)
conductor_type = np.random.choice(['AAAC', 'ACSR', 'ACCC'], n_samples)
region_factor = np.random.uniform(0.8, 2.0, n_samples)

total_cost = []
line_loss = []

conductor_cost_multiplier = {'AAAC': 450, 'ACSR': 500, 'ACCC': 650}
conductor_loss_factor = {'AAAC': 0.055, 'ACSR': 0.048, 'ACCC': 0.035}

for i in range(n_samples):
    rl, tc, ct, rf = route_length[i], tower_count[i], conductor_type[i], region_factor[i]
    
    base_cost = rl * tc * conductor_cost_multiplier[ct] * rf
    cost_variance = np.random.normal(0, base_cost * 0.15)
    total_cost.append(max(50000, base_cost + cost_variance))
    
    base_loss = rl * conductor_loss_factor[ct]
    loss_variance = np.random.normal(0, base_loss * 0.1)
    line_loss.append(max(0.01, base_loss + loss_variance))

cost_data = pd.DataFrame({
    'route_length': route_length,
    'tower_count': tower_count,
    'conductor_type': conductor_type,
    'region_factor': region_factor,
    'total_cost': total_cost,
    'line_loss': line_loss
})

cost_data.to_csv('Cost & Loss Estimation Data.csv', index=False)
print(f"   ✓ Dataset saved: 'Cost & Loss Estimation Data.csv' ({n_samples:,} samples)")

# ==============================
# 3. Train Conductor & Insulation Selection Model
# ==============================
print("\n[3/5] Training RandomForestClassifier for Conductor & Insulation Selection...")

X_conductor = conductor_data[['voltage', 'current', 'temperature', 'pollution']]
y_conductor = conductor_data['conductor']
y_insulator = conductor_data['insulator']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['pollution'])
    ],
    remainder='passthrough'
)

conductor_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

conductor_pipeline.fit(X_conductor, y_conductor)
conductor_accuracy = conductor_pipeline.score(X_conductor, y_conductor)
print(f"   ✓ Conductor model trained - Accuracy: {conductor_accuracy*100:.2f}%")

insulator_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

insulator_pipeline.fit(X_conductor, y_insulator)
insulator_accuracy = insulator_pipeline.score(X_conductor, y_insulator)
print(f"   ✓ Insulator model trained - Accuracy: {insulator_accuracy*100:.2f}%")

combined_model = {
    'conductor': conductor_pipeline,
    'insulator': insulator_pipeline
}

joblib.dump(combined_model, 'conductor_model.joblib')
print(f"   ✓ Model saved: 'conductor_model.joblib'")

# ==============================
# 4. Train Cost & Loss Estimation Model
# ==============================
print("\n[4/5] Training XGBRegressor for Cost & Loss Estimation...")

X_cost = cost_data[['route_length', 'tower_count', 'conductor_type', 'region_factor']]
y_cost = cost_data[['total_cost', 'line_loss']]

cost_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['conductor_type'])
    ],
    remainder='passthrough'
)

cost_pipeline = Pipeline([
    ('preprocessor', cost_preprocessor),
    ('regressor', MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
])

cost_pipeline.fit(X_cost, y_cost)
cost_score = cost_pipeline.score(X_cost, y_cost)
print(f"   ✓ Cost & Loss model trained - R² Score: {cost_score:.4f}")

joblib.dump(cost_pipeline, 'cost_model.joblib')
print(f"   ✓ Model saved: 'cost_model.joblib'")

# ==============================
# 5. Summary
# ==============================
print("\n[5/5] Training Complete!")
print("=" * 60)
print("Generated Files:")
print("  1. Conductor & Insulation Selection Data.csv")
print("  2. Cost & Loss Estimation Data.csv")
print("  3. conductor_model.joblib")
print("  4. cost_model.joblib")
print("=" * 60)
print("✓ All models trained and saved successfully!")
print("✓ Ready to run the Streamlit application")
print("=" * 60)
