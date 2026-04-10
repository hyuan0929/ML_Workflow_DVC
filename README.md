# ML Workflow with DVC (MNIST CNN Pipeline)

## 📌 Project Overview

This project demonstrates a simple **Machine Learning pipeline** using **DVC (Data Version Control)**.

The goal is to show how data preparation, model training, and prediction can be organized into a **reproducible and automated workflow**.

We use the **MNIST dataset** and a simple **Convolutional Neural Network (CNN)** to illustrate this process.

---

## ⚙️ Pipeline Structure

The pipeline consists of three stages:
prepare → train → predict


### 1. Prepare
- Downloads the MNIST dataset
- Converts data into PyTorch tensors
- Saves processed data into `data/processed`

### 2. Train
- Loads processed data
- Trains a CNN model
- Saves:
  - model (`model.pt`)
  - evaluation metrics (`metrics.json`)

### 3. Predict
- Loads trained model and test data
- Runs inference
- Saves predictions (`predictions.json`)

---

## 🔁 Role of DVC

DVC is used to:

- Track **data and model artifacts**
- Define pipeline stages in `dvc.yaml`
- Automatically rerun only necessary stages
- Ensure **reproducibility**

Example command:
dvc repro


DVC detects changes in:
- code
- data
- parameters

and reruns only affected stages.

---

## 🧪 Experiments

We conducted multiple experiments to analyze how different hyperparameters affect model performance.

### 🔹 Activation Functions

We tested:
- ReLU
- LeakyReLU
- GELU

| Experiment | Activation | Init | Epochs | LR | Batch Size | Accuracy |
|----------|------------|------|--------|----|------------|----------|
| 1 | ReLU | default | 2 | 0.001 | 64 | 0.9546 |
| 2 | LeakyReLU | default | 2 | 0.001 | 64 | 0.9624 |
| 3 | GELU | default | 2 | 0.001 | 64 | 0.9662 |

**Observation:**  
GELU achieved the highest accuracy, followed by LeakyReLU and ReLU. This suggests smoother activation functions can improve performance.

---

### 🔹 Optimizers

We tested:
- Adam
- SGD
- SGD with Momentum

| Experiment | Optimizer | Momentum | Accuracy |
|----------|----------|----------|----------|
| 1 | Adam | 0.9 | 0.9567 |
| 2 | SGD | 0.0 | 0.8440 |
| 3 | SGD with Momentum | 0.9 | 0.9133 |

**Observation:**  
Adam converged faster and achieved better accuracy. SGD without momentum performed worst, while momentum improved stability and performance.

---

### 🔹 Forward and Backward Pass Inspection

We instrumented the training process to observe:

- Forward outputs (logits)
- Loss values
- Gradient norms
- Parameter updates

Example observations:

- Batch loss: **2.2902**
- Conv gradient norm: **0.060388**
- FC gradient norm: **0.919531**

**Insight:**  
This confirms that:
- forward pass generates predictions
- loss measures error
- backward pass computes gradients
- optimizer updates parameters

---

## 🧠 Key Concepts Demonstrated

- Machine Learning pipelines
- Hyperparameter tuning
- Optimizer behavior
- Activation functions
- Forward and backward passes
- Reproducibility with DVC

---

## 🚀 ML Pipelines in CI/CD

This project reflects how ML systems are used in **real-world CI/CD environments**:

- Pipelines automate data → training → prediction
- Version control ensures reproducibility
- Experiments can be tracked and compared
- Changes trigger only necessary recomputation

This is essential for scalable and production-ready ML workflows.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- DVC
- YAML
- Git

---

## 👥 Team Members

- Haibo Yuan  
- Ce Chen  
- Zhuoran Zhang  

---

## ▶️ How to Run

From the project root:
pip install -r requirements.txt
dvc repro
dvc metrics show


---

## ✅ Conclusion

This project demonstrates how to build a structured and reproducible ML workflow using DVC.  

By organizing the pipeline into stages and tracking dependencies, we can efficiently experiment, debug, and scale machine learning systems.
