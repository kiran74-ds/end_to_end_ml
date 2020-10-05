# End to End Macchine Learning

The idea behind this project is to develop a Machine Learning model which can be deployed in the production.

Project Instructions:

This project can be run locally by following the steps:

```
python3 -m venv venv
```

```
source venv/bin/activate
```

```
pip install -r requirements.txt 
```

```
python3 -m pylint app
```

```
python3 -m pytest app/test
```

```
python3  app/src/train.py
```

Run the project using docker  

```
docker build . -t ml_project
```

```
docker run  ml_project
```
