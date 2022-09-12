You can run inference with [cell-tracking-challenge(ctc) datasets](http://celltrackingchallenge.net/3d-datasets/) or your own datasets.
The provided models were trained with [Fluo-N3DH-CE dataset](http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N3DH-CE.zip), which is a microscopy dataset of C.elegans.

[Segmentation models](http://celltrackingchallenge.net/participants/UCSB-US/#) were developed by [UCSB-US team](http://celltrackingchallenge.net/participants/UCSB-US/) and download from [cell tracking challenge homepage](http://celltrackingchallenge.net/latest-csb-results/).

1. Put ctc or your datasets to data directory.
1. Edit `<path to cell-tracker-gnn>/src/inference/config/config_inference.yaml` .
1. Run `<path to cell-tracker-gnn>/src/inference/Fluo-N3DH-CE.sh` .

example:
```
git clone https://github.com/jianglonghui/cell-tracker-gnn
cd cell-tracker-gnn
conda create -n ctc --file requirements-conda.txt
conda activate ctc
pip install -r requirements.txt
TEST_DIR="./data/CTC/Test"
mkdir -p ${TEST_DIR}
wget -P ${TEST_DIR} http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N3DH-CE.zip && unzip ${TEST_DIR}/Fluo-N3DH-CE.zip -d ${TEST_DIR}
sh ./src/inference/Fluo-N3DH-CE.sh
```
