FROM python:3.6-slim

WORKDIR /code
COPY . /code

RUN pip install -r requirements.txt
RUN python -m pylint app/
RUN python -m pytest app/test/

CMD [ "python", "app/src/train.py" ]

