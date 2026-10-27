Hybrid Quantum Machine Learning for Heart Disease Prediction

Project Overview
This project explores a hybrid classical–quantum machine learning approach for heart disease prediction using a heart disease dataset.

Classical machine learning techniques are used for preprocessing, feature selection, and outlier detection. A Variational Quantum Classifier (VQC) is then trained using a parameterized quantum circuit.

Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Qiskit
- Qiskit Machine Learning

Workflow
1. Load the heart disease dataset.
2. Split the data into training and testing sets.
3. Apply standard scaling.
4. Select four features using Random Forest feature importance.
5. Detect and remove outliers from the training data.
6. Encode the selected features using a four-qubit quantum circuit with RY gates.
7. Apply CNOT entanglement.
8. Build a RealAmplitudes variational ansatz.
9. Train a Variational Quantum Classifier using the COBYLA optimizer.
10. Evaluate predictions using classification metrics.

Evaluation Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- Classification report

How to Run

1. Install the required Python packages listed in `requirements.txt`.
2. Ensure `heart.csv` is in the same directory as `pp.py`.
3. Run the Python script:

```bash
python pp.py
```

Dataset
The program reads the dataset from `heart.csv`. Add the original dataset source and licence information here before publishing.

## Limitations
This is an educational machine learning project. Its predictions are experimental and are not intended for medical diagnosis. The project does not establish that quantum machine learning performs better than classical models.
