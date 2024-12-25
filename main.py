import argparse
import os
import sys

import yaml

from implementations.srgan.srgan import srgan_train
from implementations.esrgan.esrgan import esrgan_train

def update_opt_from_dict(opt, config_dict):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            if not hasattr(opt, key):
                setattr(opt, key, argparse.Namespace())
            update_opt_from_dict(getattr(opt, key), value)
        else:
            setattr(opt, key, value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/srgan.yaml", help="dataset config")
    parser.add_argument("--test_mode", action="store_true", help="Enable test mode")
    opt = parser.parse_args()
    
    # Load the configuration file if provided
    if opt.config:
        with open(opt.config, 'r') as file:
            config_args = yaml.safe_load(file)
            update_opt_from_dict(opt, config_args)

    print(opt)

    save_path = os.path.join("./saved_models", opt.name, opt.main.save_path)
    log_path = os.path.join(save_path, "logging/train")
    os.makedirs(log_path, exist_ok=True)

    sys.stdout = open("{}/out.txt".format(log_path), "w")
    sys.stderr = open("{}/err.txt".format(log_path), "w")

    if opt.name == "srgan":
        print("INSIDE")
        srgan_train(opt.main)
    if opt.name == "esrgan":
        print("INSIDE")
        esrgan_train(opt.main)
    else:
        raise NotImplementedError

    sys.stdout.close()
    sys.stderr.close()


if __name__ == "__main__":
    main()
