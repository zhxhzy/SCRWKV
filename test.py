import numpy as np
import torch
import argparse
import os
import cv2
import time
from datasets import create_dataset
from models import build_model
from main import get_args_parser

parser = argparse.ArgumentParser('SCRWKV', parents=[get_args_parser()])
args = parser.parse_args()
args.phase = 'test'
args.dataset_path = ''

if __name__ == '__main__':
    args.batch_size = 1
    t_all = []
    device = torch.device(args.device)
    test_dl = create_dataset(args)
    load_model_file = "./checkpoints/weights/checkpoint_best.pth"
    data_size = len(test_dl)
    model, criterion = build_model(args)
    model.eval()
    state_dict = torch.load(load_model_file)
    model.load_state_dict(state_dict["model"])
    print("Load Model Successful!")

    model.cuda()
    save_root = 'Image_pre/' + args.dataset_path.split('/')[-1]
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_dl):
            x = data["image"].cuda()
            t1 = time.time()
            out = model(x)
            out = out[0, 0, ...].cpu().numpy()
            root_name = data["A_paths"][0].split("/")[-1][0:-4]
            out = 255 * (out / np.max(out))
            out[out >= 0.2] = 255
            out[out < 0.2] = 0
            print(os.path.join(save_root, "{}.png".format(root_name)))
            cv2.imwrite(os.path.join(save_root, "{}.png".format(root_name)), out)
            t2 = time.time()
            t_all.append(t2-t1)
        print('average time:', np.mean(t_all) / 1)
        print('average fps:', 1 / np.mean(t_all))

        print('fastest time:', min(t_all) / 1)
        print('fastest fps:', 1 / min(t_all))

        print('slowest time:', max(t_all) / 1)
        print('slowest fps:', 1 / max(t_all))

    print("All Test Finished!")
