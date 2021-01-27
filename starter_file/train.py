from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
from azureml.core import Experiment, Workspace


run = Run.get_context()
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '510b94ba-e453-4417-988b-fbdc37b55ca7'
resource_group = 'aml-quickstarts-136286'
workspace_name = 'quick-starts-ws-136286'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='cancer')
df=dataset.to_pandas_dataframe()


def clean_data(data):
    x_df=df.iloc[ :,2:10]
    y_df=df.iloc[:,1]


    return x_df,y_df
    

x, y = clean_data(ds)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs',exist_ok=True)
    joblib.dump(value=model,filename='outputs/model.pkl')

if __name__ == '__main__':
    main()