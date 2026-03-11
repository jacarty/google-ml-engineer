FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    joblib==1.3.2 \
    google-cloud-storage==2.14.0 \
    gcsfs>=2023.12.0

COPY train.py /app/train.py

ENTRYPOINT ["python", "train.py"]
