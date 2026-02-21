FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    joblib \
    gcsfs \
    cloudml-hypertune

# Copy training script
COPY lab3_train.py .

ENTRYPOINT ["python", "lab3_train.py"]
