# 👔 HR Job Title Predictor

## 🚀 Overview
The **HR Job Title Predictor** is a machine learning-powered web application built with **Streamlit**. It predicts job titles based on skills, responsibilities, and job descriptions. This tool is designed to assist HR professionals in categorizing job roles efficiently.

## 📂 How It Works
1. **User Inputs**:
   - Required skills (e.g., Python, Leadership)
   - Job responsibilities
   - Full job description
2. **Preprocessing**:
   - Text normalization (lowercasing, cleaning, etc.)
   - TF-IDF vectorization for feature extraction
3. **Prediction**:
   - A **Random Forest model** trained on **300,000+ job listings** predicts the most likely job title.
   - Displays **alternative suggestions** with confidence scores.

## 🏗 Application Features
- 🎯 **Accurate job title predictions** based on text input
- 📊 **Confidence-based alternative job title suggestions**
- 🎨 **Enhanced UI styling for better user experience**
- 📂 **Expandable model details and usage guide**

## 📌 Installation & Setup
### 🔧 Dependencies
Ensure you have Python installed along with the required packages:
```bash
pip install streamlit pandas scikit-learn pickle-mixin
```

### ▶ Running the App
```bash
streamlit run main.py
```

## 📜 Model Components
- **`rf_model.pkl`** → Pretrained **Random Forest model**
- **`tfidf_vectorizer.pkl`** → **TF-IDF vectorizer** for text processing
- **`label_encoder.pkl`** → Converts predicted labels back to job titles

## 🧠 Machine Learning Approach
### 🔍 Preprocessing
- **TF-IDF Vectorization**: Converts text into numerical representations
- **Feature Selection**: Focuses on **skills, responsibilities, and job descriptions**

### 🤖 Model Used
- **Random Forest Classifier** trained on 300,000+ job listings
- **Predicts job titles with high accuracy**

## 🔮 Future Improvements
- 🏆 **Fine-tune model hyperparameters for better accuracy**
- 🌍 **Expand dataset with global job listings**
- 📡 **Deploy API for external integrations**

🚀 **Helping HR professionals streamline job classification!**

