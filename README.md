# garbage_classification
AI project for classifying garbage into several types using image dataset
# 🗑️ Garbage Classification using Deep Learning

This project is developed as part of the **AICTE–Shell–Edunet Internship** under the domain of **Green Skills and AI**. It aims to classify different types of garbage images using transfer learning with **EfficientNetV2B0**.

---

## 📁 Dataset

- The dataset used is a structured image dataset with the following classes:
  - `cardboard`
  - `glass`
  - `metal`
  - `paper`
  - `plastic`
  - `trash`

- The dataset is stored in the folder: `garbage_image_dataset/`
- Images are pre-categorized into respective folders (supervised learning setup).

---

## ✅ Week 1 Progress

### 🔹 Tasks Completed:
- Uploaded dataset to GitHub
- Loaded dataset using TensorFlow `image_dataset_from_directory`
- Printed class names
- Implemented **data augmentation**
- Used **EfficientNetV2B0** as base model
- Created a custom classification model
- Trained the model for 3 epochs with validation split

---

## 🧠 Model Architecture

- **EfficientNetV2B0** (pre-trained on ImageNet, frozen)
- Data augmentation (flip, rotation)
- Global Average Pooling
- Dense layer with ReLU
- Dropout layer
- Output layer with Softmax activation

```python
EfficientNetV2B0 (frozen)
→ GlobalAveragePooling2D
→ Dense(64, activation='relu')
→ Dropout(0.3)
→ Dense(6, activation='softmax')
# 🗑️ Garbage Classification using EfficientNetV2B2

A deep learning-based waste classification system using transfer learning with EfficientNetV2B2. This project classifies garbage images into 6 categories: cardboard, glass, metal, paper, plastic, and trash. Built and deployed as part of the AICTE–Shell–Edunet Internship.

---

## 📌 Project Overview

Waste segregation is essential for a cleaner and sustainable environment. This project uses state-of-the-art image classification with `EfficientNetV2B2` to automatically detect garbage types from images, aiding smart waste management systems.

---

## 🛠️ Tech Stack

- 🧠 **Model**: EfficientNetV2B2 (Transfer Learning)
- 🖼️ **Framework**: TensorFlow / Keras
- 📊 **Evaluation**: Confusion Matrix, Accuracy, Classification Report
- 📁 **Dataset**: Trash classification dataset from [Kaggle](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)
- 🌐 **Deployment**: Hugging Face Spaces with Gradio Interface
- 💻 **Tools**: Python, Jupyter Notebook, Matplotlib, Seaborn, Pandas

---

## 📂 Dataset

- 6 Classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- Total Images: ~2500+
- Split: 70% Train, 20% Validation, 10% Test
- Format: Directory-based image classification

-

- ✅ Understood the problem statement and domain.
- ✅ Installed required Python libraries (TensorFlow, OpenCV, etc.).
- ✅ Explored and preprocessed dataset using `image_dataset_from_directory`.
- ✅ Performed data augmentation (Random Flip, Rotation, Zoom, Contrast).

### 📅 Week 2: Model Development & Training

- ✅ Built CNN model using EfficientNetV2B2 with transfer learning.
- ✅ Froze initial layers and fine-tuned last few layers.
- ✅ Used `EarlyStopping`, `ModelCheckpoint`, and `class_weights` to boost accuracy.
- ✅ Trained and validated model (Best Val Accuracy: **~89%**).
- ✅ Saved the model in `.keras` format.

---

## 📈 Model Evaluation

- ✅ Achieved 85%+ accuracy on the validation set.
- ✅ Evaluated on unseen test data.
- ✅ Generated classification report and confusion matrix using Scikit-learn.
- ✅ Saved plots of accuracy and loss curves.

---

## 🚀 Deployment

- ✅ Deployed the final model on **Hugging Face Spaces** with Gradio.
- ✅ Interface allows image upload or webcam input for real-time prediction.

🔗 **Live Demo on Hugging Face**: [Click Here to Try It](https://huggingface.co/spaces/your-username/garbage-classifier)  
*(Replace link with your actual Hugging Face Space URL)*

---

## 📸 Sample Predictions

| Input Image | Predicted Class |
|-------------|-----------------|
| 🥤 Plastic Cup | `plastic` |
| 📄 Crumpled Paper | `paper` |
| 🥫 Tin Can | `metal` |

---

## 🤝 Acknowledgements

- 📚 Internship Support: AICTE–Shell–Edunet Foundation
- 💾 Dataset Source: [Trash Dataset - Kaggle](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)
- 🙏 TensorFlow, Gradio, Hugging Face Teams

---

## 📬 Contact

Made with  by **Sreepathi Praveena**  
  
📫 Email: praveena555p@gmail.com


