from __future__ import print_function
import argparse

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

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
    return cv2.waitKey(0) in (ord('q'), 27)  # q or ESC


def display_results_count(input_img, density_map, crowd_count):
   density_map = 255 * density_map / np.max(density_map)
   density_map= density_map[0][0] # density_map 1,1,h,w
   show_map = cv2.cvtColor(density_map, cv2.COLOR_GRAY2RGB)

   print('>>>density ', type(input_img), input_img.shape, show_map.shape, density_map.shape)
   options = (cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,), 2)

   cv2.putText(input_img, 'total count: ' + str(int(crowd_count)), (20, 25), *options)

   input_img = cv2.resize(input_img, (show_map.shape[1], show_map.shape[0]))

   fig = plt.figure()
   ax = fig.add_subplot(121)
   ax.imshow(density_map, cmap=plt.cm.hot)

   ax1 = fig.add_subplot(122)
   ax1.imshow(input_img)

   plt.show()


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
        #if show(img, den, cnt): break
        display_results_count(img, den, cnt)


if __name__ == '__main__':
    main()
