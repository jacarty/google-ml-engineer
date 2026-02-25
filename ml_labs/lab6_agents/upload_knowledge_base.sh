#!/bin/bash
# =============================================================
# Lab 6: Agent Builder — Knowledge Base Upload Script
# =============================================================
# This script:
#   1. Downloads GCP documentation pages as clean text files
#   2. Uploads everything in a local folder to GCS
#
# Prerequisites:
#   - gcloud CLI authenticated (gcloud auth login)
#   - pip install beautifulsoup4 requests
#
# Usage:
#   chmod +x upload_knowledge_base.sh
#   ./upload_knowledge_base.sh
# =============================================================

set -e

# --- Configuration ---
BUCKET="gs://carty-470812-ml-census-data"
GCS_PREFIX="agent-builder/knowledge-base"
LOCAL_DIR="."
DOWNLOADS_DIR="$LOCAL_DIR/gcp-docs"

echo "============================================"
echo "  Lab 6: Knowledge Base Preparation"
echo "============================================"
echo ""
echo "Local staging directory: $LOCAL_DIR"
echo "GCS destination:        $BUCKET/$GCS_PREFIX/"
echo ""

# --- Create local directories ---
mkdir -p "$DOWNLOADS_DIR"
mkdir -p "$LOCAL_DIR/my-notes"

# --- Step 1: Download GCP docs as text ---
echo "--- Step 1: Downloading GCP documentation ---"
echo ""

# Python script to fetch and clean HTML docs into text files
python3 << 'PYEOF'
import os
import sys
import time

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing required packages...")
    os.system("pip install beautifulsoup4 requests --quiet")
    import requests
    from bs4 import BeautifulSoup

DOWNLOADS_DIR = os.path.join(os.getcwd(), "gcp-docs")

# URL -> filename mapping
DOCS = {
    "vertex-ai-overview.txt": 
        "https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform",
    "vertex-ai-for-bigquery-users.txt": 
        "https://cloud.google.com/vertex-ai/docs/beginner/bqml",
    "hyperparameter-tuning-overview.txt": 
        "https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview",
    "hyperparameter-tuning-create-job.txt": 
        "https://cloud.google.com/vertex-ai/docs/training/using-hyperparameter-tuning",
    "feature-store-overview.txt": 
        "https://cloud.google.com/vertex-ai/docs/featurestore/latest/overview",
    "agent-builder-overview.txt": 
        "https://cloud.google.com/agent-builder/overview",
    "grounding-overview.txt": 
        "https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview",
    "grounding-with-vertex-ai-search.txt": 
        "https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-with-vertex-ai-search",
    "create-search-data-store.txt": 
        "https://cloud.google.com/generative-ai-app-builder/docs/create-data-store-es",
    "about-apps-and-data-stores.txt": 
        "https://cloud.google.com/generative-ai-app-builder/docs/create-datastore-ingest",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0"
}

for filename, url in DOCS.items():
    filepath = os.path.join(DOWNLOADS_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"  ✓ Already exists: {filename}")
        continue
    
    print(f"  Downloading: {filename}")
    print(f"    URL: {url}")
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove script, style, nav, footer elements
        for tag in soup(["script", "style", "nav", "footer", "header", 
                         "aside", "meta", "link"]):
            tag.decompose()
        
        # Try to find the main content area (GCP docs use article or devsite-content)
        main = (soup.find("article") or 
                soup.find("div", class_="devsite-article-body") or
                soup.find("main") or 
                soup.body or 
                soup)
        
        # Extract text with some structure preserved
        text = main.get_text(separator="\n", strip=True)
        
        # Clean up excessive blank lines
        lines = text.split("\n")
        cleaned = []
        prev_blank = False
        for line in lines:
            line = line.strip()
            if not line:
                if not prev_blank:
                    cleaned.append("")
                prev_blank = True
            else:
                cleaned.append(line)
                prev_blank = False
        
        # Add source URL header
        content = f"Source: {url}\nDownloaded: {time.strftime('%Y-%m-%d')}\n\n"
        content += "\n".join(cleaned)
        
        with open(filepath, "w") as f:
            f.write(content)
        
        size_kb = len(content) / 1024
        print(f"    ✓ Saved ({size_kb:.1f} KB)")
        
        time.sleep(1)  # Be polite
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")

print(f"\n  Done. {len(os.listdir(DOWNLOADS_DIR))} files in {DOWNLOADS_DIR}")
PYEOF

# --- Step 2: Prompt for local files ---
echo ""
echo "--- Step 2: Add your own files ---"
echo ""
echo "Copy your local files into: $LOCAL_DIR/my-notes/"
echo ""
echo "Suggested files:"
echo "  - Your certification plan (.md)"
echo "  - Lab notebooks from Labs 1-4 (.ipynb or .md)"
echo "  - Any Obsidian notes that are mostly text (.md)"
echo "  - Your agent pipeline prompt files (.md)"
echo ""
echo "Supported formats: TXT, PDF, HTML, DOCX, PPTX, XLSX"
echo ""

# Check if my-notes has files already
if [ "$(ls -A "$LOCAL_DIR/my-notes/" 2>/dev/null)" ]; then
    echo "Found files in my-notes/:"
    ls -la "$LOCAL_DIR/my-notes/"
    echo ""
else
    echo "No files in my-notes/ yet."
    echo ""
    read -p "Press Enter when you've added your files (or Enter to skip): "
fi

# Convert any .md files to .txt for Agent Builder compatibility
echo "Converting .md files to .txt..."
find "$LOCAL_DIR" -name "*.md" -exec bash -c 'mv "$1" "${1%.md}.txt"' _ {} \;

# --- Step 3: Upload to GCS ---
echo ""
echo "--- Step 3: Uploading to GCS ---"
echo ""

# Count files
TOTAL_FILES=0
for dir in "$DOWNLOADS_DIR" "$LOCAL_DIR/my-notes"; do
    if [ -d "$dir" ]; then
        COUNT=$(find "$dir" -type f | wc -l)
        TOT
        AL_FILES=$((TOTAL_FILES + COUNT))
    fi
done

echo "Total files to upload: $TOTAL_FILES"
echo "Destination: $BUCKET/$GCS_PREFIX/"
echo ""

# Upload GCP docs
if [ "$(ls -A "$DOWNLOADS_DIR" 2>/dev/null)" ]; then
    echo "Uploading GCP documentation..."
    gsutil -m cp -r "$DOWNLOADS_DIR"/* "$BUCKET/$GCS_PREFIX/gcp-docs/"
    echo "  ✓ GCP docs uploaded"
fi

# Upload personal notes
if [ "$(ls -A "$LOCAL_DIR/my-notes/" 2>/dev/null)" ]; then
    echo "Uploading your notes..."
    gsutil -m cp -r "$LOCAL_DIR/my-notes/"* "$BUCKET/$GCS_PREFIX/my-notes/"
    echo "  ✓ Personal notes uploaded"
fi

# --- Step 4: Verify ---
echo ""
echo "--- Step 4: Verify uploads ---"
echo ""
echo "Files in GCS:"
gsutil ls -l "$BUCKET/$GCS_PREFIX/**" 2>/dev/null || echo "  (no files found — check bucket name)"

echo ""
echo "============================================"
echo "  Knowledge base ready!"
echo "============================================"
echo ""
echo "GCS path for Agent Builder datastore:"
echo "  $BUCKET/$GCS_PREFIX/"
echo ""
echo "Next step: Create a datastore in the Agent Builder console"
echo "  https://console.cloud.google.com/gen-app-builder/data-stores"
echo ""
