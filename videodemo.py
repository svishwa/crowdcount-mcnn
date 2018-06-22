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
    den = net(im_data)
    den = den.data.cpu().numpy()
    et_count = np.sum(den)
    return den, et_count


def show(frame, den, count):
    den = den[0][0]
    den = 255 * den / np.max(den)
    den = den.astype(np.uint8, copy=False)
    den = cv2.equalizeHist(den)
    den = cv2.applyColorMap(den, cv2.COLORMAP_JET)
    for i, a in enumerate(den):
        for j, b in enumerate(a):
            if np.array_equal(b, [128, 0, 0]):
                den[i, j] = [255, 255, 255]

    overlay = cv2.resize(den, frame.shape[:2][::-1])
    alpha = .4
    print(overlay.shape, frame.shape)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    txt = 'cnt: %d' % count
    cv2.putText(frame, txt, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('density', den)
    return cv2.waitKey(1) in (ord('q'), 27)  # q or ESC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('--model-path', '-M',
                   default='./final_models/mcnn_shtechB_110.h5')
    p.add_argument('--estimate-rate', '-r', default=10)
    p.add_argument('--resize-fx', '-x', default=.5)
    return p.parse_args()


def main():
    args = parse_args()

    net = CrowdCounter(is_cuda)
    network.load_net(args.model_path, net)
    if is_cuda:
        net.cuda()
    print('eval:', net.eval())

    video = cv2.VideoCapture(args.video)
    nframe = 0
    hist = []
    while 1:
        ok, frame = video.read()
        frame = cv2.resize(frame, (0, 0), fx=args.resize_fx, fy=args.resize_fx)
        nframe += 1
        if nframe % args.estimate_rate != 0:
            continue
        print('nframe', nframe)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgarr = image_to_array(gray)

        den, cnt = run(net, imgarr)
        """
        if len(hist) > 5:
            hist.pop(0)
        hist.append(den)
        histden = sum(hist) / len(hist)
        """
        if show(frame, den, cnt):
            break


if __name__ == '__main__':
    main()
