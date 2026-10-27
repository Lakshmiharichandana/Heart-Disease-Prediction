# ============================================================
# QUANTUM-ENHANCED HEART DISEASE PREDICTION
# ============================================================
# Classical preprocessing:
#   1. Train/Test Split
#   2. Standard Scaling
#   3. Random Forest Feature Selection
#   4. Top 4 Feature Selection
#   5. Isolation Forest Outlier Detection
#
# Quantum Machine Learning:
#   6. 4-Qubit Angle Encoding using RY gates
#   7. CNOT Entanglement
#   8. Real Amplitudes Variational Ansatz
#   9. COBYLA Optimizer
#   10. Variational Quantum Classifier (VQC)
#
# Evaluation:
#   Accuracy
#   Precision
#   Recall
#   F1 Score
#   Confusion Matrix
#   Classification Report
# ============================================================


# ============================================================
# STEP 1 : IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# QISKIT IMPORTS
# ============================================================

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector

# New function instead of deprecated RealAmplitudes class
from qiskit.circuit.library import real_amplitudes

from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA


# ============================================================
# STEP 2 : LOAD DATASET
# ============================================================

print("\n======================================")
print("LOADING HEART DISEASE DATASET")
print("======================================\n")

data = pd.read_csv("heart.csv")

print(data.head())

print("\nDataset Shape :", data.shape)


# ============================================================
# STEP 3 : SEPARATE INPUT AND TARGET
# ============================================================

print("\n======================================")
print("DATA PREPARATION")
print("======================================\n")

X = data.drop("target", axis=1)

y = data["target"]

print("Input Features :", X.shape[1])
print("Total Samples  :", X.shape[0])


# ============================================================
# STEP 4 : TRAIN / TEST SPLIT
# IMPORTANT:
# SPLIT BEFORE SCALING, FEATURE SELECTION
# AND OUTLIER DETECTION
# ============================================================

print("\n======================================")
print("TRAIN TEST SPLIT")
print("======================================\n")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train_raw))
print("Testing Samples  :", len(X_test_raw))


# ============================================================
# STEP 5 : STANDARD SCALING
# FIT ONLY ON TRAINING DATA
# ============================================================

print("\n======================================")
print("STANDARD SCALING")
print("======================================\n")

scaler_all = StandardScaler()

X_train_scaled = scaler_all.fit_transform(
    X_train_raw
)

X_test_scaled = scaler_all.transform(
    X_test_raw
)

print("Standard Scaling Completed")


# ============================================================
# STEP 6 : RANDOM FOREST FEATURE SELECTION
# TRAINING DATA ONLY
# ============================================================

print("\n======================================")
print("FEATURE SELECTION")
print("======================================\n")

rf_selector = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_selector.fit(
    X_train_scaled,
    y_train.to_numpy()
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = rf_selector.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)


print("\nFeature Importance:")
print(feature_df)


# ============================================================
# STEP 7 : SELECT TOP 4 FEATURES
# ============================================================

top_features = feature_df[
    "Feature"
].head(4).values

print("\nTop Selected Features:")
print(top_features)


# ============================================================
# STEP 8 : KEEP ONLY TOP 4 FEATURES
# ============================================================

X_train_selected = X_train_raw[
    top_features
].copy()

X_test_selected = X_test_raw[
    top_features
].copy()


# ============================================================
# STEP 9 : SCALE SELECTED FEATURES
# FIT ONLY ON TRAINING DATA
# ============================================================

print("\n======================================")
print("SCALING SELECTED FEATURES")
print("======================================\n")

selected_scaler = StandardScaler()

X_train_selected = selected_scaler.fit_transform(
    X_train_selected
)

X_test_selected = selected_scaler.transform(
    X_test_selected
)

print("Selected Feature Scaling Completed")


# ============================================================
# STEP 10 : OUTLIER DETECTION
# TRAINING DATA ONLY
# ============================================================

print("\n======================================")
print("OUTLIER DETECTION")
print("======================================\n")

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

outlier_pred = iso.fit_predict(
    X_train_selected
)


# ============================================================
# REMOVE ONLY TRAINING OUTLIERS
# ============================================================

train_mask = outlier_pred != -1

X_train_clean = X_train_selected[
    train_mask
]

# IMPORTANT:
# Convert Pandas Series to NumPy array
# This fixes the Qiskit reshape error.

y_train_clean = y_train.iloc[
    train_mask
].to_numpy()


# ============================================================
# TEST DATA IS KEPT COMPLETELY UNTOUCHED
# ============================================================

X_test_clean = X_test_selected.copy()

# Convert test labels to NumPy
y_test_clean = y_test.copy().to_numpy()


print(
    "Original Training Samples :",
    len(X_train_selected)
)

print(
    "Training Samples After Outlier Removal :",
    len(X_train_clean)
)

print(
    "Training Outliers Removed :",
    len(X_train_selected) - len(X_train_clean)
)

print(
    "Test Samples Kept Untouched :",
    len(X_test_clean)
)


# ============================================================
# STEP 11 : QUANTUM FEATURE MAP
# ============================================================

print("\n======================================")
print("ANGLE ENCODING")
print("======================================\n")


# Four selected features
# Four qubits

n_qubits = 4


# ============================================================
# CREATE PARAMETER VECTOR
# ============================================================

input_params = ParameterVector(
    "x",
    n_qubits
)


# ============================================================
# CREATE QUANTUM CIRCUIT
# ============================================================

feature_map = QuantumCircuit(
    n_qubits
)


# ============================================================
# RY ANGLE ENCODING
# ============================================================

for i in range(n_qubits):

    feature_map.ry(
        input_params[i],
        i
    )


# ============================================================
# CNOT ENTANGLEMENT
# ============================================================

feature_map.cx(0, 1)

feature_map.cx(1, 2)

feature_map.cx(2, 3)


# ============================================================
# DISPLAY FEATURE MAP
# ============================================================

print("\nQuantum Feature Map:")
print(feature_map)


# ============================================================
# STEP 12 : VARIATIONAL QUANTUM CIRCUIT
# ============================================================

print("\n======================================")
print("VARIATIONAL QUANTUM CIRCUIT")
print("======================================\n")


# Same basic RealAmplitudes architecture
# reps = 2

ansatz = real_amplitudes(
    num_qubits=n_qubits,
    reps=2
)


print(ansatz)


# ============================================================
# STEP 13 : COBYLA OPTIMIZER
# ============================================================

print("\n======================================")
print("COBYLA OPTIMIZER")
print("======================================\n")

optimizer = COBYLA(
    maxiter=50
)

print("Optimizer Initialized")


# ============================================================
# STEP 14 : QUANTUM SAMPLER
# ============================================================

print("\n======================================")
print("STATEVECTOR SAMPLER")
print("======================================\n")

sampler = StatevectorSampler()

print("Quantum Sampler Ready")


# ============================================================
# STEP 15 : CREATE VQC
# ============================================================

print("\n======================================")
print("CREATING VQC MODEL")
print("======================================\n")

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)

print("VQC Model Created")


# ============================================================
# STEP 16 : TRAIN VQC
# ============================================================

print("\n======================================")
print("TRAINING VQC MODEL")
print("======================================\n")

print(
    "Training samples used :",
    len(X_train_clean)
)

vqc.fit(
    X_train_clean,
    y_train_clean
)

print("\nQuantum Training Completed")


# ============================================================
# STEP 17 : PREDICTION
# ============================================================

print("\n======================================")
print("QUANTUM PREDICTION")
print("======================================\n")

quantum_pred = vqc.predict(
    X_test_clean
)

# Make prediction a 1-D NumPy array
quantum_pred = np.asarray(
    quantum_pred
).reshape(-1)


# ============================================================
# STEP 18 : ACCURACY
# ============================================================

quantum_acc = accuracy_score(
    y_test_clean,
    quantum_pred
)

print(
    "\nQuantum VQC Accuracy :",
    quantum_acc
)


# ============================================================
# STEP 19 : PRECISION
# ============================================================

quantum_precision = precision_score(
    y_test_clean,
    quantum_pred,
    zero_division=0
)

print(
    "Quantum VQC Precision :",
    quantum_precision
)


# ============================================================
# STEP 20 : RECALL
# ============================================================

quantum_recall = recall_score(
    y_test_clean,
    quantum_pred,
    zero_division=0
)

print(
    "Quantum VQC Recall :",
    quantum_recall
)


# ============================================================
# STEP 21 : F1 SCORE
# ============================================================

quantum_f1 = f1_score(
    y_test_clean,
    quantum_pred,
    zero_division=0
)

print(
    "Quantum VQC F1 Score :",
    quantum_f1
)


# ============================================================
# STEP 22 : CONFUSION MATRIX
# ============================================================

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================\n")

cm = confusion_matrix(
    y_test_clean,
    quantum_pred
)

print(cm)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Quantum VQC Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()


# ============================================================
# STEP 23 : CLASSIFICATION REPORT
# ============================================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================\n")

print(
    classification_report(
        y_test_clean,
        quantum_pred,
        zero_division=0
    )
)


# ============================================================
# STEP 24 : ACCURACY GRAPH
# ============================================================

print("\n======================================")
print("ACCURACY GRAPH")
print("======================================\n")

plt.figure(
    figsize=(5, 5)
)

plt.bar(
    ["Quantum VQC"],
    [quantum_acc]
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Quantum VQC Accuracy"
)

plt.show()


# ============================================================
# STEP 25 : SAMPLE PREDICTION
# ============================================================

print("\n======================================")
print("SAMPLE PREDICTION")
print("======================================\n")


sample = X_test_clean[0]

sample_prediction = vqc.predict(
    [sample]
)

sample_prediction = np.asarray(
    sample_prediction
).reshape(-1)


print(
    "Prediction :",
    sample_prediction[0]
)


if sample_prediction[0] == 1:

    print(
        "Heart Disease Detected"
    )

else:

    print(
        "No Heart Disease Detected"
    )


# ============================================================
# STEP 26 : FINAL RESULTS
# ============================================================

print("\n======================================")
print("FINAL RESULTS")
print("======================================\n")

print(
    "Dataset Used : Heart Disease Dataset"
)

print(
    "Total Samples :",
    len(data)
)

print(
    "Original Training Samples :",
    len(X_train_raw)
)

print(
    "Original Testing Samples :",
    len(X_test_raw)
)

print(
    "Training Samples After Outlier Removal :",
    len(X_train_clean)
)

print(
    "Testing Samples Used :",
    len(X_test_clean)
)

print(
    "Selected Features :",
    list(top_features)
)

print(
    "Number of Qubits :",
    n_qubits
)

print(
    "Quantum Accuracy :",
    quantum_acc
)

print(
    "Quantum Precision :",
    quantum_precision
)

print(
    "Quantum Recall :",
    quantum_recall
)

print(
    "Quantum F1 Score :",
    quantum_f1
)

print("\nPROJECT COMPLETED SUCCESSFULLY")
