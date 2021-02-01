*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Breast Cancer Detection

*TODO:* In this project I configured an automl model to identify whether the patient has malignant or benign tumour and then deployed the model to create an endpoint.Apart
from an automl model I even developed a hyperparameter model using logistic regression that will also classify the same feature as automl model.

Architecture of the model

![image](https://user-images.githubusercontent.com/53776611/106390205-40d53880-640d-11eb-842f-f6e6a7843abe.png)



## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: The dataset contains information about two different type of breast cancer:Malignant and Benign,it contains different features that can be used to identify if a 
patient suffers from Malignant or Benign type of tumour.I found this dataset from kaggle. Link for the data set :<i>https://www.kaggle.com/uciml/breast-cancer-wisconsin-data</i>

### Task
<p> *TODO*: Using certain features like:</p>
<li>radius_mean:mean of distances from center to points on the perimeter</li>
<li>texture_mean:standard deviation of gray-scale values</li>
<li>perimeter_mean:mean size of the core tumor</li>
<li>smoothness_mean:mean of local variation in radius lengths</li>
<li>compactness:mean of perimeter^2 / area - 1.0</li>
<li>concavity_mean: mean of severity of concave portions of the contour</li>
<li>concave points :mean for number of concave portions of the contour</li>
<p>These features are used to identify the type of breast cancer the patient is suffering from.</p>

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
The following screenshots show the run details of automl model,with each step being carried out during the run process.Apart from this the screenshots also show the accuracy of each model under automl run.
![image](https://user-images.githubusercontent.com/53776611/106358172-650f1780-6330-11eb-8647-b018ed7d9050.png)
![image](https://user-images.githubusercontent.com/53776611/106388271-c227cd80-6403-11eb-86f7-86623f90b537.png)
![image](https://user-images.githubusercontent.com/53776611/106388303-dcfa4200-6403-11eb-9cd2-b5364deb0030.png)
![image](https://user-images.githubusercontent.com/53776611/106388316-eb485e00-6403-11eb-88bb-bb90c70ed802.png)
![image](https://user-images.githubusercontent.com/53776611/106388333-0024f180-6404-11eb-848a-529b875172cd.png)
![image](https://user-images.githubusercontent.com/53776611/106388915-965a1700-6406-11eb-8045-9f78bfadfa2c.png)


### Best Model
These are the screenshots of the best model with it's id and parameters.

![image](https://user-images.githubusercontent.com/53776611/106479686-99b9d500-64d0-11eb-97a2-dcec790126fc.png)
![image](https://user-images.githubusercontent.com/53776611/106479775-b2c28600-64d0-11eb-9314-800d3193d5cb.png)

![image](https://user-images.githubusercontent.com/53776611/106388934-ab36aa80-6406-11eb-99dc-808140d908e3.png)
![image](https://user-images.githubusercontent.com/53776611/106388960-d7eac200-6406-11eb-8add-efdc5def59aa.png)
![image](https://user-images.githubusercontent.com/53776611/106388978-ef29af80-6406-11eb-900e-60055496d2f5.png)

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
The following screenshots show the run details of hyperparameter tuning model,with each step being carried out during the run process.
![image](https://user-images.githubusercontent.com/53776611/106389564-e7b7d580-6409-11eb-950e-8ff00ff590e8.png)
![image](https://user-images.githubusercontent.com/53776611/106389588-0d44df00-640a-11eb-95b7-ed2bc5be1d11.png)

![image](https://user-images.githubusercontent.com/53776611/106358568-0e570d00-6333-11eb-9af3-d69e45eb900d.png)
### Best Model
These are the screenshots of the best model with it's id and parameters.
![image](https://user-images.githubusercontent.com/53776611/106389606-28175380-640a-11eb-8e02-997b115497af.png)

![image](https://user-images.githubusercontent.com/53776611/106358596-38103400-6333-11eb-985a-d2c3c6ee372f.png)
![image](https://user-images.githubusercontent.com/53776611/106389624-382f3300-640a-11eb-8493-c8b26697d4c9.png)



## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
I deployed the automl model and the endpoint of the model is an url.When a json request (which contains all the needed input features and their values) is made to the url,
the model response with an output being 1 or 0. 1 means the patient suffers from malignant tumour and 0 means the patient suffers from benign tumour.

## Consume the deployed model
To deploy the model I have initially registered the model,configured the model and then deployed it.The endpoint of the model is an URL.
![image](https://user-images.githubusercontent.com/53776611/106389773-37e36780-640b-11eb-986f-9f837989324f.png)

![image](https://user-images.githubusercontent.com/53776611/106388216-87259a00-6403-11eb-9b19-80e18434b52d.png)
![image](https://user-images.githubusercontent.com/53776611/106388242-a6242c00-6403-11eb-9415-633bd13977b6.png)

![image](https://user-images.githubusercontent.com/53776611/106388152-2dbd6b00-6403-11eb-94a8-c95f0e4d7db6.png)
![image](https://user-images.githubusercontent.com/53776611/106388232-973d7980-6403-11eb-9844-b3a6890db231.png)


## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
### Screen recording link: <i>https://youtu.be/VCPRUfOvY1w</i>

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
