"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/restaurant/base_model/',
                    help='Directory containing params.json')


def launch_training_job(parent_dir, job_name, opt):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        opt: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    opt.save(json_path)

    # Launch training with this config
    cmd = "{} train2.py --model_dir={}".format(PYTHON, model_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    opt = utils.Params(json_path)

    # Perform hypersearch over one parameter
    batch_size = [4, 8, 32, 128, 256]
    dropout = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    for value in batch_size:
        print('*' *100)
        # Modify the relevant parameter in params
        opt.batch_size = value

        # Launch job (name has to be unique)
        job_name = "batch_size/{}".format(value)
        launch_training_job(args.parent_dir, job_name, opt)