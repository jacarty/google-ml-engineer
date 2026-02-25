# Convert to markdown then text
pip install nbconvert
find . -name "*.ipynb" -exec jupyter nbconvert --to markdown {} \;
gsutil -m cp *.txt gs://carty-470812-ml-census-data/agent-builder/knowledge-base/my-notes/