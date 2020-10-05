## End to End Machine Learning

The idea behind this project is to develop a Machine Learning model which can be deployed in the production. As part of this I have built a binary classification model on titanic dataset.Premilary Anlaysis on data can be found in the Jupyter notebook which resides in research folder.
These are the following items which I have implemented in the code.
+ Create Virtual Environment using python.
+ Pylint to follow best coding practices.
+ Defining unit test cases using unittest library.
+ Run unit test cases using pytest.
+ Train Machine Learning model
+ Make predictions on the model

## Project Instructions:

### This project can be run locally by following the steps:

+ Create Virtual Environmnet
```
python3 -m venv venv
```
+ Activate Virtual Environment

```
source venv/bin/activate
```
+ Install necessary libraries using pip

```
pip install -r requirements.txt 
```
+ Run pylint to follow style recommended by PEP 8, the Python style guide
```
python3 -m pylint app
```
+ Run unit test cases to make sure our code works as expected
```
python3 -m pytest app/test
```
+ Finally train the model and get predictions
```
python3  app/src/train.py
```

### Running the project using docker  

+ Clone the repository 

+ Build Docker Image
```
docker build . -t ml_project
```
+ Run Docker Image
```
docker run  ml_project
```

