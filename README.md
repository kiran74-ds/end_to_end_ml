# end_to_end_ml

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

```
docker build . -t ml_project
```

```
docker run  ml_project
```
