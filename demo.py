from __future__ import print_function
import argparse

import cv2
import torch
import numpy as np

from mcnn.crowd_count import CrowdCounter
from mcnn import network

is_cuda = False
torch.backends.cudnn.enabled = is_cuda
torch.backends.cudnn.benchmark = False


def image_to_array(img):
    img = img.astype(np.float32, copy=False)
    ht, wd = img.shape[:2]
    img = img.reshape((1, 1, ht, wd))
    return img


def run(net, im_data):
    density_map = net(im_data)
    density_map = density_map.data.cpu().numpy()
    et_count = np.sum(density_map)
    return density_map, et_count


def show(origin, density_map, count):
    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]

    txt = 'cnt: %d' % count
    cv2.putText(origin, txt, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.imshow('density', density_map)
    cv2.imshow('origin', origin)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('image_files', nargs='+')
    p.add_argument('--model-path', '-M',
                   default='./final_models/mcnn_shtechB_110.h5')
    return p.parse_args()


def main():
    args = parse_args()

    net = CrowdCounter(is_cuda)
    network.load_net(args.model_path, net)
    if is_cuda:
        net.cuda()
    print('eval:', net.eval())

    for filename in args.image_files:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgarr = image_to_array(gray)
        den, cnt = run(net, imgarr)
        show(img, den, cnt)
        if cv2.waitKey(0) in (ord('q'), 27):  # q or ESC
            break


if __name__ == '__main__':
    main()
