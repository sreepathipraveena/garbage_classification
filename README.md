# 🗑️ Garbage Classification using Deep Learning

This project was developed as part of the **AICTE–Shell–Edunet Internship** under the domain of **Green Skills and Artificial Intelligence**. It aims to classify various types of waste using a **deep learning model** powered by **transfer learning** with the **EfficientNetV2B2** architecture. The system is designed to aid in **automated waste segregation**, contributing to smarter and more sustainable environmental practices.

---

## 📌 Project Overview

Effective waste management starts with accurate segregation. This project uses **image classification** techniques to automatically identify and classify garbage into six key categories. The model is trained on a labeled dataset and deployed using a user-friendly **Gradio** interface.

---

## 📁 Dataset

- The dataset contains labeled images organized into six categories:
  - `cardboard`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `trash`

- Directory structure follows the supervised learning setup:
garbage_image_dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/

yaml
Copy
Edit

- Total Images: ~2500+  
- Dataset Split:
- 70% Training
- 20% Validation
- 10% Testing

📦 Dataset Source: [Trash Type Dataset - Kaggle](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)

---

## 🧠 Model Architecture

The model leverages **EfficientNetV2B2**, a state-of-the-art CNN architecture, using **transfer learning**:

```python
EfficientNetV2B2 (frozen layers)
→ Data Augmentation (flip, rotation, zoom)
→ GlobalAveragePooling2D
→ Dense(64, activation='relu')
→ Dropout(0.3)
→ Dense(6, activation='softmax')  # For 6 classes
Key Features:

Pretrained on ImageNet

Fine-tuned top layers

Uses dropout to prevent overfitting

Optimized using EarlyStopping and class weights

🛠️ Tech Stack
Category	Tools / Frameworks
💻 Programming	Python
📦 Framework	TensorFlow, Keras
🧠 Model	EfficientNetV2B2
📊 Visualization	Matplotlib, Seaborn
🧪 Evaluation	Scikit-learn (Confusion Matrix, Accuracy)
🌐 Deployment	Hugging Face Spaces + Gradio Interface

✅ Weekly Progress Summary
Week 1: Data & Setup
Loaded dataset using image_dataset_from_directory

Visualized class distribution

Implemented data augmentation techniques

Initialized EfficientNetV2B0 for benchmarking

Week 2: Model Development & Training
Switched to EfficientNetV2B2 for better performance

Built and trained the model (3+ epochs)

Applied EarlyStopping, ModelCheckpoint, and class weights

Achieved validation accuracy of ~89%

Week 3: Evaluation & Deployment
Evaluated model on test set (85%+ accuracy)

Generated classification report and confusion matrix

Deployed using Gradio on Hugging Face Spaces

📈 Model Evaluation
The model was tested on unseen images, and performance was measured using:

Validation Accuracy: ~89%

Test Accuracy: ~85%

Evaluation Metrics: Precision, Recall, F1-Score

📸 Sample Prediction Output
Below is a real prediction output from the deployed model. The image of compressed cardboard was uploaded to the interface. The model correctly predicted the class as cardboard with the highest confidence:

🔍 Predicted Output Example:


Predicted Class: cardboard
Confidence Scores:

cardboard: 38%

paper: 16%

trash: 14%

glass: 14%

metal: 10%

plastic: 9%

🚀 Deployment
The model was deployed using Gradio and hosted on Hugging Face Spaces.

🔗 Live Demo: Click here to try it out
(Replace the link above with your actual Hugging Face deployment URL)

Features:

Upload image or use webcam

Real-time prediction and confidence visualization

Supports all six garbage categories

🧪 Sample Predictions Table
Image	Predicted Class
🥤 Plastic Cup	plastic
📄 Crumpled Paper	paper
🥫 Tin Can	metal
📦 Flattened Boxes	cardboard

🤝 Acknowledgements
Internship Support: AICTE – Shell – Edunet Foundation

Dataset Provider: Kaggle Trash Dataset

Frameworks & Tools: TensorFlow, Gradio, Hugging Face:https://huggingface.co/spaces/praveena5jessy/garbage-classification

📬 Contact
Made with ❤ by Sreepathi Praveena
📫 Email: praveena555p@gmail.com

