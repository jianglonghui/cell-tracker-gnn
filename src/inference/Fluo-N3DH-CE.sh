#!/bin/bash
CONF_INFERENCE="./src/inference/config/config_inference.yaml"
CODE_TRA="./src/inference/tracking"

# seg prediction
python ./src/inference/segmentation/predict_stacks_N3DCH.py ${CONF_INFERENCE}

# Finish segmentation - start tracking

# our model needs CSVs, so let's create from image and segmentation.
python ${CODE_TRA%/}/preprocess_seq2graph.py ${CONF_INFERENCE}

# run the prediction
python ${CODE_TRA%/}/inference_clean.py ${CONF_INFERENCE}

# postprocess
python ${CODE_TRA%/}/postprocess_clean.py ${CONF_INFERENCE}

# evaluation
##### alert: GT for ctc test dataset is not provided. #####
# python ${CODE_TRA%/}/evaluation/cell_tracking_evaluation.py ${CONF_INFERENCE}

# rm -rf ./tmp
