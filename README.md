# 🌊 ENSO Explorer — Interactive Climate Analysis App

This Streamlit app analyzes **ENSO (El Niño / La Niña)** impacts on the tropical Pacific Ocean using daily TAO mooring observations. It provides data cleaning, visualization, anomaly detection, and predictive modeling tools.

---

## 🚀 Features

### **1. Overview**
- Dataset description  
- Temporal coverage  
- Missing/outlier detection  
- Explanation of how raw TAO files were merged and daily-averaged

### **2. Missingness Analysis**
- Missing value summary  
- Missingness heatmap  
- Random Forest–based MICE imputation  
- Visual comparison of original vs imputed values  

### **3. Temporal Coverage & ENSO Influence**
- SST colored by ENSO index  
- Multi-depth temperature time series (10–250 m)  
- Interactive vertical temperature profile animation  

### **4. Correlation Study**
- Correlation heatmap  
- Scatterplots with regression lines  
- ENSO-colored scatterplots  
- Pairwise scatter matrix  

### **5. ENSO Anomalies**
- Monthly climatology  
- SST anomaly time series  
- SST anomaly heatmap  
- STL decomposition (trend, seasonality, residual)  
- Feature engineering for prediction models  

### **6. SST Prediction Models**
Two models illustrate different predictive capabilities:

- **Autoregressive (AR) model**  
  - Uses past SST values only  
  - Performs poorly because ENSO depends on winds, subsurface heat content, and seasonality  
  - Low R² (slightly negative)

- **Random Forest Regression**  
  - Uses engineered atmospheric and oceanic features  
  - Captures nonlinear ENSO dynamics  
  - Very high predictive skill (R² ≈ 0.98)

### **7. Conclusion**
- ENSO strongly modulates SST, winds, and humidity  
- Imputation and feature engineering greatly improve model performance  
- The Random Forest far outperforms the AR model because it incorporates physical drivers of ENSO  

------------------------------------------------------------------------------------------------------

## 🙌 Acknowledgements
- TAO mooring data from **NOAA PMEL**
- ENSO index (Niño 3.4 ANOM) from **NOAA CPC**
