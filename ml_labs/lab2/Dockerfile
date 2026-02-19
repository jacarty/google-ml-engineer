FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    joblib==1.3.2 \
    gcsfs \
    pyarrow \
    google-cloud-storage \
    flask==3.0.0 \
    gunicorn==21.2.0

COPY lab2_custom_train.py /app/train.py
COPY serve.py /app/serve.py

ENTRYPOINT ["python", "train.py"]
