# Wine-Quality-Prediction
# 🍷 Wine Quality Detector

#An end-to-end Machine Learning project that predicts whether a wine is of good
#quality or not based on its chemical properties.

## 🔹 Algorithm
Random Forest Classifier

## 🔹 Features
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

## 🔹 Target
Wine quality is classified as:
- 1 → Good Quality (quality >= 7)
- 0 → Low Quality (quality < 7)

## 🔹 Tech Stack
- Python
- Scikit-learn
- Streamlit

## 🔹 Deployment
The model is deployed as a live web app using Streamlit Cloud.

## 🔹 How to Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app_ui.py
