# End-to-end-Machine-Learning-Project-with-MLflow


## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py



# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/atishgautam07/e2e-mlops-gcp.git
```
### STEP 01- Create an environment after opening the repository

```bash
pipenv install -r requirements.txt -python=3.9
```

```bash
pipenv shell
```


### STEP 02- install the requirements

```bash
# Finally run the following command
python test.py
```

Now,
```bash
open up you local host and port
```

### Docker build and run
```bash
docker build -t ride-duration-pred-service:v1 .
```

```bash
docker run -it --rm -p 9696:9696 ride-duration-pred-service:v1
```
<!-- ## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model -->
