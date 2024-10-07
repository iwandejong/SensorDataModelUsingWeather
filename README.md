## Soil Data Imputation

### Problem Statement

Soil data is a critical component of many environmental and agricultural studies. However, soil data is often incomplete due to the high cost of collecting soil samples. This project aims to develop a machine learning model to impute missing soil data. The model will be trained on a dataset of soil samples with missing values. The model will predict the missing values based on the other features in the dataset.

## NPK Model

An example of a model where it's difficult to engineer features based on temperature variables is the NPK model. The reason for this is that the NPK model is based on the chemical composition of the soil, which is not directly related to temperature. However, the NPK model can still be used to impute missing values in the soil dataset.

### RMSE
```
RMSE for N:  13.148592481085865
RMSE for P:  9.386787549765902
RMSE for K:  3.487565349637972
```

* this means that the model is able to predict Nitrogen with an error of 13.15, Phosphorus with an error of 9.39, and Potassium with an error of 3.49, which is relatively high.

## Soil Moisture/Temperature Model

An example of a model where there's much more accurate data is the soil moisture/temperature model. This model is based on the relationship between soil temperature/moisture and humidity and temperature. This model can be used to impute missing values in the soil dataset.

### RMSE
```
RMSE Soil Moisture:  1.5507586655843835
RMSE Soil Temperature:  0.07878415991080076
```

* this means that the model is able to predict soil moisture with an error of 1.55 and soil temperature with an error of 0.08, which is relatively low.
