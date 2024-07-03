FROM python:3.9.7-slim

RUN apt update -y && apt install awscli -y
RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY . /app
# COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

# CMD ["python3","app.py"]
# COPY [ "predict.py", "lin_reg.bin", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app:app" ]