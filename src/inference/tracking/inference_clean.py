import os
import warnings

import torch
import yaml

from datamodules.graph_dataset_inference import CellTrackDataset
from src.models.celltrack_plmodel import CellTrackLitModel

warnings.filterwarnings("ignore")


def predict(ckpt_path, path_csv_output, inf_out, ndim):
    """Inference with trained model.
    It loads trained model from checkpoint.
    Then it creates graph and make prediction.
    """

    CKPT_PATH = ckpt_path
    path_output = path_csv_output

    folder_path = os.path.dirname((os.path.dirname(CKPT_PATH)))

    config_path = os.path.join(folder_path, '.hydra/config.yaml')
    config = yaml.safe_load(open(config_path))

    print(f"load model from: {CKPT_PATH}")
    data_yaml = config['datamodule']

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    trained_model = CellTrackLitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # print model hyperparameters
    print(trained_model.hparams)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    data_yaml['dataset_params']['num_frames'] = 'all'
    data_yaml['dataset_params']['main_path'] = path_output

    data_yaml['dataset_params']['dirs_path']['test'] = [path_csv_output]
    data_yaml['dataset_params']['ndim'] = ndim

    data_train: CellTrackDataset = CellTrackDataset(**data_yaml['dataset_params'], split='test')
    data_list, df_list = data_train.all_data['test']
    test_data, df_data = data_list[0], df_list[0]
    x, x2, edge_index, edge_feature = test_data.x, test_data.x_2, test_data.edge_index, test_data.edge_feat

    outputs = trained_model((x, x2), edge_index, edge_feature.float())
    data_path = inf_out
    path_output_folder = data_path
    print(f"save path : {path_output_folder}")
    os.makedirs(path_output_folder, exist_ok=True)
    file1 = os.path.join(path_output_folder, 'pytorch_geometric_data.pt')
    file2 = os.path.join(path_output_folder, 'all_data_df.csv')
    file3 = os.path.join(path_output_folder, 'raw_output.pt')
    print(f"Save inference files: \n - {file1} \n - {file2} \n - {file3}")
    df_data.to_csv(file2)
    torch.save(test_data, file1)
    torch.save(outputs, file3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str, help='analysis configuration file')

    args = parser.parse_args()
    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    model_path = conf['checkpoint dir']
    output_csv = conf['csv output dir']
    ndim = conf['dimension']
    inf_out = conf['inference dir']
    if not output_csv:
        input_images = conf['input image dir']
        input_segmentation = os.path.join(
            os.path.dirname(input_images),
            os.path.basename(input_images) + "_CSV",
            )
    predict(model_path, output_csv, inf_out, ndim)
