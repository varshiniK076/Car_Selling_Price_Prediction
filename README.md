# 🚗 Car Selling Price Prediction Using Regression

## **Project Overview**

This project is a **web-based application** that predicts the **selling price of used cars** using a **linear regression model**. The app allows users to input car features such as:

- Car Name / Brand  
- Year of Manufacture  
- Present Price  
- Kilometers Driven  
- Fuel Type (Petrol, Diesel, CNG)  
- Seller Type (Dealer / Individual)  
- Transmission (Manual / Automatic)  
- Owners (0/1)  

and outputs the **estimated selling price** in Indian Rupees(Lakhs).

This project demonstrates **data preprocessing, feature engineering, model training, and deployment** using Python, Pandas, Scikit-learn, and Flask.

---
## 📊 Dataset Description
The dataset contains historical car sale information with the following features:

| Feature        | Description                                  |
|----------------|----------------------------------------------|
| Car_Name       | Name of the car                              |
| Year           | Manufacturing year                           |
| Present_Price  | Current showroom price (in lakhs)            |
| Kms_Driven     | Total kilometers driven                      |
| Fuel_Type      | Petrol / Diesel / CNG                        |
| Seller_Type    | Individual / Dealer                          |
| Transmission   | Manual / Automatic                           |
| Owner          | Number of previous owners                    |
| Selling_Price  | Target variable                              |

---
## ⚙️ Project Structure

```text
CarPricePrediction/
│
├─ car_price.csv
├─ app.py
├─ main.py
├─ handling_missing_values.py
├─ variable_transformation.py
├─ categorical_to_numeric.py
├─ feature_scaling.py
├─ model_training.py
├─ log_code.py
│
├─ brand_price_map.pkl
├─ global_mean.pkl
├─ scaler.pkl
├─ linear_regression_model.pkl
│
├─ templates/
│   └─ index.html
│
├─ static/
│   └─ style.css
│
└─ README.md
```
---

## **🔄 ML - Pipeline**

Before training the model, the raw dataset goes through feature engineering,feature selection and scaling steps to ensure the data is clean, numeric, and suitable for modeling. The pipeline is structured as follows:

---

### **1. Handling Missing Values**

- The dataset is first checked for missing values in both training and testing sets.  
- Missing numeric or categorical values are imputed using a **Random Sample Imputation (RSI) technique**, which replaces missing values with randomly selected values from the existing distribution of that column.  
- This step ensures there are **no null values** before proceeding to transformations and encoding.

---

### **2. Variable Transformation (Numerical Features)**

- Numerical features such as `Year`, `Present_Price`, `Kms_Driven`, and `Owner` are examined for skewness and outliers.  
- **Yeo-Johnson Power Transformation** is applied to reduce skewness, normalize the distributions, and make the data more Gaussian-like, which improves regression performance.  
- Outliers are capped using the **1st and 99th percentiles** to reduce their impact on the model. 
- This step ensures the numeric features are **well-distributed and scaled consistently** for modeling.

---

### **3. Categorical to Numerical Conversion**

- **Brand Extraction & Target Encoding:**  
  - The `Brand` is extracted from the `Car_Name`.  
  - A **target encoding** is computed using the average selling price for each brand (`Brand_mean`).  
  - Missing brands are filled with the **global mean selling price**.  

- **Label Encoding:**  
  - Categorical features such as `Fuel_Type`, `Seller_Type`, and `Transmission` are converted to numeric codes:  
    - `Fuel_Type` → Petrol: 1, Diesel: 0, CNG: 2  
    - `Seller_Type` → Dealer: 1, Individual: 0  
    - `Transmission` → Manual: 1, Automatic: 0  

- After this step, all features are **numerical** and ready for scaling.

---

### **4. Feature Scaling**

- All numeric features are scaled using **StandardScaler**, which standardizes each feature to have **zero mean and unit variance**.  
- Scaling ensures that all features contribute equally to the regression model and prevents features with larger ranges from dominating the model learning.

---

### **5. Model Training**

- A **Linear Regression** model is trained on the preprocessed and scaled training data.  
- The model predicts the selling price of cars based on input features.  
- Model performance is evaluated using the following metrics:  
  - **R² Score**
  - **Root Mean Squared Error (RMSE)**  
  - **Mean Absolute Error (MAE)**

- After training, the model and preprocessing objects (`brand_price_map`, `global_mean`, `scaler`, and `linear_regression_model`) are saved as **.pkl files** for use in the web application.

---

## **🧠 Machine Learning Model**
- Algorithm: Linear Regression
- Inputs:
  - Car Name / Brand  
  - Year of Manufacture  
  - Present Price  
  - Kilometers Driven  
  - Fuel Type (Petrol, Diesel, CNG)  
  - Seller Type (Dealer / Individual)  
  - Transmission (Manual / Automatic)  
  - Owners (0/1) 
- Target Variable: Selling_Price
- Model trained on transformed & scaled features
- Saved for deployment

---
## **🌐 Web Application Using Flask**

## Backend
  - Flask framework used
  - Loads trained model, scaler & encoder
  - Ensures preprocessing consistency with training pipeline
## Frontend
  - Simple HTML form
  - User inputs car details
  - Displays predicted selling price



---
## **📦 Install required packages**
> ```
> pip install -r requirements.txt
> ```

## **🚀 Run the Project**
> ```
> python main.py
> python app.py
> ```
---
### **Open your browser and navigate to**
> ```
> http://127.0.0.1:5000/
> ```

---

## **👤 Author**
 ```
 Varadhana Varshini Kolipakula
 Machine Learning & Data Science Enthusiast
 ```

---
