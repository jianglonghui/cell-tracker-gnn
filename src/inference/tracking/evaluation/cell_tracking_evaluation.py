import argparse
import glob
import os
import subprocess
import tempfile

import yaml


def rms(s):
    while s.endswith('/'):
        s = s[-1:]
    return s

def ads(s):
    if not s.endswith('/'):
        s += "/"
    return s

def ln(d, s):
    subprocess.run(f'ln -s {ads(d)} {rms(s)}', shell=True, check=True)

def calculate(gt_dir, seq, tra_dir, os_type):
    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_tmp = os.path.join(tmp_dir, seq + "_GT")
        res_tmp = os.path.join(tmp_dir, seq + "_RES")
        segtra = os.path.join(os.path.dirname(__file__), 'segtra_measure', os_type)

        ln(gt_dir, gt_tmp)
        ln(tra_dir, res_tmp)

        if os_type != "Win":
            tra = subprocess.run(f'{os.path.join(segtra, "TRAMeasure")} {tmp_dir} {seq}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            seg = subprocess.run(f'{os.path.join(segtra, "SEGMeasure")} {tmp_dir} {seq}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        else:
            tra = subprocess.run(f'{os.path.join(segtra, "TRAMeasure.exe")} {tmp_dir} {seq}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            seg = subprocess.run(f'{os.path.join(segtra, "SEGMeasure.exe")} {tmp_dir} {seq}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        return tra.stdout.decode("utf8"), seg.stdout.decode("utf8")


if __name__ == '__main__':

    print(f'Refer to http://celltrackingchallenge.net/evaluation-methodology/ for the evaluation methodology.')
    parser = argparse.ArgumentParser()

    parser.add_argument('conf', type=str, help='analysis configuration file')

    args = parser.parse_args()

    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    gt_dir = conf['GT dir']
    seq = conf['seq']
    tra_dir = conf['tracking dir']
    os_type = conf['os type']
    assert os_type in ["Linux", "Win", "Mac"], "os type has to be Linux, Mac, or Win."

    if not isinstance(seq, list):
        seq = [seq]
    for s in seq:
        tra_stdout, seg_stdout = calculate(gt_dir, s, tra_dir, os_type)
        tra = tra_stdout.split()[-1]
        seg = seg_stdout.split()[-1]

        print(tra_stdout)
        print(seg_stdout)

        try:
            print(f'cell tracking measure: {float(tra) + float(seg)}')
        except ValueError:
            print('\033[31m' + f'Error exists in the process.' + '\033[0m')
