# Predictors-of-Breast-Cancer-Recurrence

- perhaps refer to the notebooks and the .py files within the documentation.
- a dash makes a list item with bullet points, interesting

## Data Collection

*Source and Description*: Data was sourced from the [UCI Machine Learning Data Archive](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer). Alongside the description in the data repository, further description is available in this [paper](https://www.causeweb.org/usproc/sites/default/files/usclap/2018-1/Predictors_for_Breast_Cancer_Recurrence.pdf)

*Objective*: Predicting the recurrence of breast cancer based on selected variables.

## Exploratory Data Analysis

*Data Loading*: The data was loaded directly into [Google Colab](https://github.com/Otobi1/Predictors-of-Breast-Cancer-Recurrence/blob/master/notebooks/Predictors_of_Cancer_Recurrence.ipynb) as follows

```bash
!wget http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data
```

After loading, the data was read as a csv file using the pandas dataframe. Column names had to be included
as part of the dataframe attributes since the data originally had no labels.

```bash
df = pd.read_csv('breast-cancer.data', sep=',', names=['RecClass', 'Age', 'Menopause',
                                                       'TumorSize', 'InvNodes', 'NodeCaps',
                                                       'DegMalig', 'Breast', 'Quadrant', 'Radiation']) 
```

*Variable Types*: The data contained 286 observations and 10 variables and all the variables were of the object type
except the "DegMalig". In addition to the exploration of the dataframe, the "data[variable].unique" revealed the specific
categories of possibilities within each variable. Substitute the "variable" with the actual label.

*Independent and Dependent Variable*: All the variables are dependent except the "RecClass" variable, which specifies whether there
was a "recurrence-event" or "no-recurrence-event".

## Preprocessing/Feature Enginnering/Dummy Variables

*Categorical to Integers*: For the categorical variables, dummies were created as follows. Note that "Variable" here is
a placeholder for the categorical variables. To prevent multicollinearity (dummy variable trap), the first of new columns
was dropped. Also, the original variable is dropped from the dataframe the new dataframe is concatenated with the original one.
This is done for all the categorical variables. (Next time, would perhaps use the LabelEncoder class of Sklearn)
The dummy variable creation process changed the dimensions of the dataframe into 286 observations and 34 varaibles, all of which
became the integer variable type except the "RecClass" which will be predicted.

```bash
Variable = pd.get_dummies(data['Variable'], drop_first=True)

data = data.drop('Variable', axis=1)

Variable = Variable.add_prefix('Variable ')

data = pd.concat([data, Variable], axis=1)
```

*Mapping, Splitting and Standardisation*: All numeric variables were mapped as the input variable X, and the outcome,
which is a categorical variable was mapped as y. After that the mapped dataset was split into training and test datasets.
A validation dataset was not quite necessary as it was a small sample size (arguable). In the splitting, the outcome y was
used as the stratification parameter to ensure a fairly balanced division. The sizes of the training and testing split was
further explored just because and the class distribution further explored as follows.

```bash
class_count = dict(collections.Counter(y))
train_class_count = dict(collections.Counter(y_train))
test_class_count = dict(collections.Counter(y_test))

print (f'classes: {class_count}')
print (f"no-rec:rec = {class_count['no-recurrence-events']/class_count['recurrence-events']:.2f}")
print (f'train classes: {train_class_count}')
print (f"train no-rec:rec = {train_class_count['no-recurrence-events']/train_class_count['recurrence-events']:.2f}")
print (f'test classes: {test_class_count}')
print (f"test no-rec:rec = {test_class_count['no-recurrence-events']/test_class_count['recurrence-events']:.2f}")
```

Lastly, the input X was standardised to ensure a mean of approximately 0 and standard deviation of approximately 1.

## Modelling and Evaluation

*Overview*: Several classification models from the Sklearn's library was used. Each model was evaluated by comparing the
Predicted and the Actual RecClass, the training score, cross validation score and the test score. The following describes
the peculiarities of each model and the evaluation outcomes.

- *Logistic Regression*: used a pipeline and parameters including the l1 penalty and the liblinear solver. The grid search
  for the best estimator used a cv of 5, accuracy scoring and 2 number of jobs. The model had a training score of 78%, a
  cross val score of 72% and a test score of 69%.

- *K-Nearest Neighbors Classifier*: similarly used a pipeline and params grid and yielded a training score of 71%, a cross val score of 69% and a test score of 72%.

- *Decision Tree Classifier*: had a similar structure with varying arguments. Had a training score of 90%, a cross val score of 72% and a test score of 68%.

- *Bagging Classifier*: had a training score of 99% (overfits on the training data), a cross val score of 72% and a test score of 72%.

- *Random Forest Classifier*: similar structure with different arguments. Had a training score of 99% (overfits on the training data), a cross val score of 75% and test score of 72%.

- *Extra Trees Classifier*: overfits on the training data with a score of 99%, a cross val score of 72% and test score of 69%.

- *AdaBoost Classifier*: had a training score of 77%, a cross val score of 70% and test score of 69%.

- *Gradient Boosting Classifier*: overfits on the training data with a score of 98%, a cross val score of 68% and test score of 69%.

- *Support Vector Classifier*: performs optimally on the training set at 92% accuracy, a cross val score of 75% and a test score of 76%.

- *Voting Classifier*: Almost overfits on the training data at 96% accuracy, a cross val score of 73% and test score of 72%.

### Bias - Variance Tradeoff

Bias describes the difference between the model's prediction and the actual value the model tries to predict.
Variance describes the variability of a model's prediction for a given data point. Models with high bias oversimplifies on the training data (generalises too much) and also performs poorly on the test data, while models that have high variance performs too well on the training data and fails to generalise enough leading to poor performance on the test data. Underfitting occurs when the model is unable to capture the underlying pattern of the data. Such models have high bias and low variance (typically seen when linear models are used on non-linear data). Overfitting on the other hand occurs when models capture nose together with the patterns of the data. These models have low bias and high variance (such as decision trees).

The bias-variance tradeoff aims to provide a good balance  between bias and variance without overfitting or underfitting the data. See [Understanding the Bias-Variance Tradeoff](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229) for more details.

### Confusion Matrix

Predictions were gotten from the model by fitting them on the test data split and this was fed in turn into the function below to generate the confusion matrix.

```bash
def pretty_confusion_matrix(y_true, y_pred):
  """ Creating a confusion in a nice chart."""

  # Handling the data
  cm = confusion_matrix(y_true, y_pred)
  labels = y_true.unique()
  labels.sort()

  # Plotting 
  sns.set(font_scale=1)
  plt.figure(figsize=(6, 5))

  chart = sns.heatmap(cm, annot=True, fmt='g', cmap='coolwarm', xticklabels=labels, yticklabels=labels)
  chart.set_yticklabels(chart.get_yticklabels(), rotation = 0)
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted Class')
  plt.ylabel('True Class')

pretty_confusion_matrix(y_test, svc_preds)
```

![Sample of the cool confusion matrix for the Support Vector Classifier](https://github.com/Otobi1/Predictors-of-Breast-Cancer-Recurrence/blob/master/screenshots/confusion%20matrix.PNG)

### Classification Report

Further evaluation was done using the classification report, which includes details on the precision, recall, f1-score and support on each of the RecClass outcomes.

```bash
print (f'confusion matrix:\n {confusion_matrix(y_test, svc_preds)}')
print (f'classification report:\n {classification_report(y_test, svc_preds)}')
print (f'accuracy score:\n {accuracy_score(y_test, svc_preds)}')
```

Classification Report for the Support Vector Classifier:

| Parameter             | precision | recall | f1-score | support |
| :-------------------- | :-------- | :----- | :------- | :------ |
|no-reccurence-events   |   0.78    |   0.92 |   0.85   |   51    |
|   recurrence-events   |   0.67    |   0.38 |   0.48   |   21    |
|            accuracy   |           |        |   0.76   |   72    |
|           macro avg   |   0.72    |   0.65 |   0.67   |   72    |
|        weighted avg   |   0.75    |   0.76 |   0.74   |   72    |

accuracy score: 0.7638888888888

### Exploring the Coefficients

The keys (variable labels) of the model inputs X were mapped into a keys dataframe. After that, the coefficients of the logistic regression was explored and placed in a dataframe as follows and further explored in a bar plot to identify which variables are key to the prediction.

```bash
# Putting the coefficients of the logistic regression in a dataframe

# Check the unique classes of the output (y feature)
classes = gs_lr.best_estimator_.steps[0][1].classes_

# Map the coefficients of the best estimator to the coefs dataframe
coefs = gs_lr.best_estimator_.steps[0][1].coef_

# Input features mapped into the Feature dataframe
Feature = X.columns

# Mapping the coefs and the features into the coefs dataframe
coefs_df = pd.DataFrame(coefs[0], Feature, columns=['coef']).sort_values(by='coef', ascending=False)

# Resetting the coefs_df index (the features were originally the index)
coefs_df = coefs_df.reset_index()

# Renaming the index to features
coefs_df.rename(columns={'index':'features'}, inplace=True)

# Checking the coefs_df dataframe
coefs_df
```

## Production Model

For the [production model](https://github.com/Otobi1/Predictors-of-Breast-Cancer-Recurrence/blob/master/notebooks/Predictors_of_Cancer_Recurrence_Production_Model.ipynb), the least impact variables were removed from the original preprocessed dataset, leaving a total of 19 variables which were modelled in the same way as the initial modelling process. The random forest model was selected based on the outcomes of the new modelling and pickled for production.

```bash
with open('prod_model.pkl', 'wb') as f_out:
  pickle.dump(gs_forest, f_out)
  f_out.close()
```

## Deployment

[Flask API](https://github.com/Otobi1/Predictors-of-Breast-Cancer-Recurrence/blob/master/app.py)

[Breast Cancer Recurrence Prediction App](https://breastcancer-pred.herokuapp.com/)
