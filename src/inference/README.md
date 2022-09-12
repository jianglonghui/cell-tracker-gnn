You can run inference with [cell-tracking-challenge(ctc) datasets](http://celltrackingchallenge.net/3d-datasets/) or your own datasets.
The provided [trained models](https://github.com/watarungurunnn/GSoC2022_submission/blob/main/all_params.pth) were trained with [Fluo-N3DH-CE dataset](http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N3DH-CE.zip), which is a microscopy dataset of C.elegans.

[Segmentation models](http://celltrackingchallenge.net/participants/UCSB-US/#) were developed by [UCSB-US team](http://celltrackingchallenge.net/participants/UCSB-US/) and download from [cell tracking challenge homepage](http://celltrackingchallenge.net/latest-csb-results/).
Evaluation protocols are based on [cell tracking challenge](http://celltrackingchallenge.net/evaluation-methodology/).

1. Put ctc or your datasets to data directory.
2. Put trained parameter file in `<path to cell-tracker-gnn>/outputs/example` .
3. Edit `<path to cell-tracker-gnn>/src/inference/config/config_inference.yaml` .
4. Run `<path to cell-tracker-gnn>/src/inference/Fluo-N3DH-CE.sh` .

example:
```
git clone https://github.com/jianglonghui/cell-tracker-gnn

# get trained params
git clone https://github.com/watarungurunnn/GSoC2022_submission tmp_dir
cp ./tmp_dir/all_params.pth ./cell-tracker-gnn/outputs/example/
rm -rf tmp_dir

cd cell-tracker-gnn
conda create -n ctc --file requirements-conda.txt
conda activate ctc
pip install -r requirements.txt
TEST_DIR="./data/CTC/Test"
mkdir -p ${TEST_DIR}
wget -P ${TEST_DIR} http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N3DH-CE.zip && unzip ${TEST_DIR}/Fluo-N3DH-CE.zip -d ${TEST_DIR}
sh ./src/inference/Fluo-N3DH-CE.sh
```
