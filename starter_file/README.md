*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Breast Cancer Detection

*TODO:* In this project I configured an automl model to identify whether the patient has malignant or benign tumour and then deployed the model to create an endpoint.Apart
from an automl model I even developed a hyperparameter model using logistic regression that will also classify the same feature as automl model.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: The dataset contains information about two different type of breast cancer:Malignant and Benign,it contains different features that can be used to identify if a 
patient suffers from Malignant or Benign type of tumour.I found this dataset from kaggle. Link for the data set :<i>https://www.kaggle.com/uciml/breast-cancer-wisconsin-data</i>

### Task
*TODO*: Using certain features like radius_mean,texture_mean,perimeter_mean,smoothness,etc. the task is to identify the type of breast cancer the patient is suffering from.

### Access
*TODO*: I am accessing the dataset in the workspace using the get_by_name function from the Dataset library,where the dataset is retrieved from the dataset store using it's name.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
After accessing the dataset in the workspace,I configured the automl model by specifying the task i.e 'classification',the metric for evaluation:'accuracy',the training data
and the target column(which is dignosis) and the number of cross-validation the model should make.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
From the automl run, the best model for the given problem was voting ensemble with an accuracy of 95.42%.I used first 10 columns of the dataset for training and the model paramters were:
<ul>experiment_timeout_minutes=30,
    task= 'classification',
    primary_metric= 'accuracy',
    training_data= X,
    label_column_name= 'diagnosis',
    n_cross_validations= 5</ul>
 I could have improved the model by changing the no. of cross validations and tweaking other parameters and evening by including more features in the training data.


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
### RunDetails
![image](https://user-images.githubusercontent.com/53776611/106358172-650f1780-6330-11eb-8647-b018ed7d9050.png)

### Best Model
![image](https://user-images.githubusercontent.com/53776611/106358196-88d25d80-6330-11eb-8c47-dae2e5d2a274.png)
![image](https://user-images.githubusercontent.com/53776611/106359431-c63ae900-6338-11eb-8cfd-4054f5ce84fd.png)





## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
For hyperparameter I chose logistic regression model as it is highly efficient and widely used for binary classification.The model is simple and provides good accuracy with less
computational need.It even makes no assumptions about distributions of classes in feature space.
### Types of parameters and their ranges used for the hyperparameter search
<li>RandomParameterSampling Class: Defines random sampling over a hyperparameter search space.</li>
<p> Range for "--C" parameter under this class was between 0.05-1</p>
<p> Range for "--max-iter" parameter was between 10-200</p>
<li>BanditPolicy Class: Defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation.</li>
<p>Parameter used in this class:</p>
<p>slack_factor:The ratio used to calculate the allowed distance from the best performing experiment run.</p>
<p>slack_amount: The absolute distance allowed from the best performing run.</p>
<p>evaluation_interval: The frequency for applying the policy.</p>
<p>delay_evaluation:The number of intervals for which to delay the first policy evaluation. If specified, the policy applies every multiple of evaluation_interval that is greater than or equal to delay_evaluation.</p>


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
<p>The logistic regression gave an accuracy of 90% and the parameters used for model are:</p>
<p>primary_metric_name='Accuracy',primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,max_total_runs=10, max_concurrent_runs=4</p>
<p>The model performance can be improved by either changing the ranges of the parameters or by adding more parameters and also using other classfications models such as
  RandomForest or SVM instead of logistic regression.</p>

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
### RunDetails
![image](https://user-images.githubusercontent.com/53776611/106358568-0e570d00-6333-11eb-9af3-d69e45eb900d.png)
### Best Model
![image](https://user-images.githubusercontent.com/53776611/106358596-38103400-6333-11eb-985a-d2c3c6ee372f.png)


## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
I deployed the automl model and the endpoint of the model is an url.When a json request (which contains all the needed input features and their values) is made to the url,
the model response with an output being M or B. M means the patient suffers from malignant tumour and B means the patient suffers from benign tumour.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
### Screen recording link: <i>https://youtu.be/VCPRUfOvY1w</i>

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
