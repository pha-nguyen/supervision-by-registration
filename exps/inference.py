
from __future__ import division

import sys, time, torch, random, argparse, PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
import numbers, numpy as np
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from config_utils import obtain_lk_args as obtain_args
from procedure import prepare_seed, save_checkpoint, lk_train as train, basic_eval_all as eval_all
from datasets import VideoDataset as VDataset, GeneralDataset as IDataset
from xvision import transforms
from log_utils import Logger, AverageMeter, time_for_file, convert_secs2time, time_string
from config_utils import load_configure
from models import obtain_LK as obtain_model, remove_module_dict
from optimizer import obtain_optimizer
from mtcnn import MTCNN
import cv2
import math

def preprocess(image, box):
    pass

def main(args):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    prepare_seed(args.rand_seed)

    logstr = 'seed-{:}-time-{:}'.format(args.rand_seed, time_for_file())
    logger = Logger(args.save_path, logstr)

    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    cpu = torch.device("cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    assert args.arg_flip == False, 'The flip is : {}, rotate is {}'.format(
        args.arg_flip, args.rotate_max)

    eval_transform = transforms.Compose([transforms.PreCrop(args.pre_crop_expand), transforms.TrainScale2WH(
        (args.crop_width, args.crop_height)),  transforms.ToTensor(), normalize])
    assert (args.scale_min+args.scale_max) / 2 == args.scale_eval, 'The scale is not ok : {},{} vs {}'.format(
        args.scale_min, args.scale_max, args.scale_eval)

    # Model Configure Load
    model_config = load_configure(args.model_config, logger)
    args.sigma = args.sigma * args.scale_eval

    fd = MTCNN(device=torch.device('cuda'))

    # Define network
    lk_config = load_configure(args.lk_config, logger)
    logger.log('model configure : {:}'.format(model_config))
    logger.log('LK configure : {:}'.format(lk_config))
    net = obtain_model(model_config, lk_config, args.num_pts + 1)

    net = net.cuda()
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load("snapshots/checkpoint/pfld-epoch-138-500.pth")
    net.load_state_dict(checkpoint['state_dict'], strict=False)

    detector = torch.nn.DataParallel(net.module.detector)
    detector.eval()


    cap = cv2.VideoCapture('/home/vinai/Desktop/VID_20200630_221121_970.mp4')
    out = cv2.VideoWriter('/home/vinai/Desktop/out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (int(1080),int(1920)))

    # image = cv2.imread("/home/vinai/Desktop/IMG_2363.JPG")
    first = True
    while True:
        ret, image = cap.read()
        if image is None:
            break
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ih, iw = image.shape[0], image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if first:
            bb, score = fd.detect(image)
            bb = list(map(int, bb[0]))
            first = False
            tracker = cv2.TrackerKCF_create()
            tracker.init(image, (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]))
        else:

            # ok, bb = tracker.update(image)
            # bb = list(bb)
            # bb[2] += bb[0]
            # bb[3] += bb[1]

            bb, score = fd.detect(image)
            bb = list(map(int, bb[0]))
        h, w = (bb[3] - bb[1])*float(args.pre_crop_expand), (bb[2] - bb[0])*float(args.pre_crop_expand)
        x1, y1 = int(max(math.floor(bb[0]-w), 0)), int(max(math.floor(bb[1]-h), 0))
        x2, y2 = int(min(math.ceil(bb[2]+w), iw)), int(min(math.ceil(bb[3]+h), ih))
        print((x1, y1), (x2, y2))

        _face = image[y1:y2,x1:x2]
        face = cv2.resize(_face, (112, 112))
        tensor = torch.from_numpy(face.transpose((2, 0, 1))).float().div(255)
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        tensor = tensor.view(1, 3, 112, 112)
        batch_locs = detector(tensor)
        np_batch_locs = batch_locs.detach().to(cpu).numpy()
        locations = np_batch_locs[0, :-1, :]
        scale_h, scale_w = _face.shape[0] * 1. / tensor.size(-2) , _face.shape[1] * 1. / tensor.size(-1)
        locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + x1, locations[:, 1] * scale_h + y1
        assert locations.shape[0] == 68, 'The number of points is {} vs {}'.format(68, locations.shape)

        for i in locations:
            cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)
    cap.release()
    logger.close()


if __name__ == '__main__':
    args = obtain_args()
    main(args)
