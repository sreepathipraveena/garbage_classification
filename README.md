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

