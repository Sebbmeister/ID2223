# ID2223
This repository contains the files for the first lab assignment in the course ID2223 Scalable Machine Learning and Deep Learning. The repository contains files for predicting a type of Iris flower (provided to us) and files for predicting whether a passenger survived the sinking of the Titanic (written by us based on the Iris files).

## Titanic files
There are six files related to the Titanic survival predictions.

### titanic-feature-pipeline.py
Based on iris-feature-pipeline.py, this file downloads the Titanic passenger dataset (https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv) and uploads the features to a feature group in hopsworks. Since not all features will affect the result in a meaningful way, several are removed (name, id, ticket, embarked). Some passengers are missing values for certain features, so missing values are also replaced with -1. Some data was also converted to be more manageable; the cabin numbers were for instance converted to deck numbers which seemed more useful and specific ages were converted to larger age groups (children, adults, etc.).

### titanic-feature-pipeline-daily.py
Based on iris-feature-pipeline-daily.py, this file generates a new synthetic Titanic passenger. The new passenger gets random values for the features (within some interval) and is randomly assigned as a survivor or deceased. 

### titanic-training-pipeline.py
Based on iris-training-pipeline.py, this file uses a 80/20 data split and trains a KNeighborsClassifier to train the data, after which the results are uploaded.

### titanic-batch-inference-pipeline.py
Based on iris-batch-inference-pipeline, this file is mainly used to predict the survival of the most recently generated synthetic passengers, after which it creates a confusion matrix (titanic_confusion_matrix.png) and stores the latest predictions (latest_titanic_prediction.png and latest_titanic_actual.png). Image files of a happy and sad face are used to symbolize a passenger surviving or dying, respectively.

### app.py (in huggingface-spaces-titanic)
Code for the gradio application for predicting the survival of a passenger based on user-inputted values for the features.

### app.py (in huggingface-spaces-titanic-monitor)
Code for the gradio application that shows the most recent synthetic passenger prediction and outcome, and a confusion matrix with historical prediction performance.

## Deployed apps
This repository is missing the hopsworks api key so the files (if downloaded) cannot be run directly. The apps created through the files are deployed as spaces at huggingface.co and can be accessed through the links below:

* Iris prediction: https://huggingface.co/spaces/SebLih/iris
* Iris monitoring: https://huggingface.co/spaces/SebLih/iris-monitoring
* Titanic prediction: https://huggingface.co/spaces/SebLih/titanic
* Titanic monitoring: https://huggingface.co/spaces/SebLih/titanic-monitor
