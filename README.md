# Regression Model Analysis with PyCaret
## Introduction
This project presents the process of building and evaluating a regression model using the PyCaret library

In `Pycarte` module we have the following ML techniques:
- Regression
- Classification
- Clustering
- Anomaly Detection
#### What is the best for us based on our dataset?
since we want to predict PPV which is continuous we will use the Regression algorithms



## Project Structure

The notebook consists of the following key steps:

### 1. Data Loading and Cleaning
#### loading:-
```python
df = pd.read_excel("TG_T_CashValues_Rel.xlsx")
```
#### cleaning:-
```python 
df.replace(0, np.nan, inplace=True)
df=df.dropna()
```
#### Labeling Categorical Data:-


```python
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

```

### 2. Model Setup and Training
   #### Model Setup:-
Setup function prepare dataset for training and testing, this function handles the missing values and encoding categorical variables

```python
regression_setup = setup(df, target='PPV', session_id=123,
                         categorical_features=['Gender'],
                         numeric_features=['Age', 'Dur'],
                         normalize=True)
```
#### Model Training:-
Compares multiple regression models on the dataset and selects the one with the best performance.
```python
best_model = compare_models()
best_model

```
### 3. Model Evaluation
Evaluates the best model using PyCaret’s built-in tools and cross-validation.
```python
evaluate_model(best_model)
```
Predicts values for the target variable on the testing set and calculates performance metrics RMSE and MAE.
```python
predictions = predict_model(best_model)
```
### 4. Final Model Training
training the best model on the entire dataset to make it read for deployment
```python
final_model = finalize_model(best_model)
```
---

## Model Evaluation
### Evaluating the model using RMSE and MAE

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred = predictions['prediction_label']
y_true = df['PPV'].loc[y_pred.index.tolist()]

rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
```

## Running Code
### Prerequisites:-
Ensure installing the following libraries:- 
- pandas        
- pycaret
- numpy
- scikit-learn
- openpyxl
```bash
pip install pandas numpy pycaret scikit-learn openpyxl
```
### Setup and Execution
1. Place the dataset file `TG_T_CashValues_Rel.xlsx` in the directory of the notebook.
2. Open the Jupyter Notebook and execute the cells in order.

---
## Dataset 
Source: `TG_T_CashValues_Rel.xlsx`.

Columns:
- `Gender`: Categorical variable.
- `Age`, `Dur`: Numeric Features.
- `PPV`: Target variable 

---
## Model Justification 
### Extra Trees Regressor
This model was the best selected model by `PyCaret`, It provided excellent accuracy and robustness.
#### Performance metrics:-
| MAE                      | MSE    | RMSE   | R²    | RMSLE  | MAPE   |
|--------------------------|--------|--------|-------|--------|--------|
| 0.0094                   | 0.0007 | 0.0259 | 1.0000| 0.0020 | 0.0007 |
|

---
#### Why Extra Trees Regressor?
1- it can handle complex datasets with multiple features.

2- it prevents overfitting by randomization and robust ensemble techniques.






## Outputs
```
RMSE: 0.02585692621975396
MAE: 0.009442293030224192
```
## Resources 
- [Pycarte Documentation](https://pycaret.gitbook.io/docs/get-started/modules)
- [Microsoft's AI Machine Learning Algorithm Cheat Sheet](https://learn.microsoft.com/en-us/azure/machine-learning/algorithm-cheat-sheet?view=azureml-api-1&WT.mc_id=docs-article-lazzeri#download-machine-learning-algorithm-cheat-sheet)
- [Scikit-learn Extra Trees Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
## Screenshots
![Cheat Sheet](https://learn.microsoft.com/en-us/azure/machine-learning/media/algorithm-cheat-sheet/machine-learning-algorithm-cheat-sheet.png?view=azureml-api-1#lightbox)

## Authors

- [Jad Atout](https://www.github.com/Jad-Atout)

