#!/bin/bash

OUTPUT="resource_report.txt"

echo "=== GCP Resource Report ===" > "$OUTPUT"
echo "Generated: $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

echo "--- Compute Instances ---" >> "$OUTPUT"
gcloud compute instances list >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "--- Custom Jobs ---" >> "$OUTPUT"
gcloud ai custom-jobs list --region=us-central1 >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "--- HP Tuning Jobs ---" >> "$OUTPUT"
gcloud ai hp-tuning-jobs list --region=us-central1 >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "--- Endpoints ---" >> "$OUTPUT"
gcloud ai endpoints list --region=us-central1 >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "--- Models ---" >> "$OUTPUT"
gcloud ai models list --region=us-central1 >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "--- Monitoring Jobs ---" >> "$OUTPUT"
gcloud ai model-monitoring-jobs list --region=us-central1 >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "--- Storage ---" >> "$OUTPUT"
gsutil ls gs://carty-470812-ml-census-data/ >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "--- Index Endpoints ---" >> "$OUTPUT"
gcloud ai index-endpoints list --region=us-central1 >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "--- Tensorboards ---" >> "$OUTPUT"
gcloud ai tensorboards list --region=us-central1 >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "--- Persistent Resources ---" >> "$OUTPUT"
gcloud ai persistent-resources list --region=us-central1 >> "$OUTPUT" 2>&1

echo "--- Artifact Registry Repositories ---" >> "$OUTPUT"
gcloud artifacts repositories list --location=us-central1 >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "Report saved to $OUTPUT"