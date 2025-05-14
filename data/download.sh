# data/download.sh
mkdir -p data/raw
wget http://path-to-dataset.zip -O data/raw/visdrone.zip
unzip data/raw/visdrone.zip -d data/raw/
