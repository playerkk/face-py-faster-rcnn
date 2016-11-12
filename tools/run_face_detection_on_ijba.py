from __future__ import division
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys

NETS = {'vgg16': ('VGG16',
				  # 'output/faster_rcnn_end2end/train/vgg16_faster_rcnn_iter_70000.caffemodel')}
				  'output/faster_rcnn_end2end/train/vgg16_faster_rcnn_imagenet_wider_ijba_iter_10000.caffemodel')}

def get_imdb_ijba(data_dir):
	imdb = []
	nfold = 10
	for n in xrange(nfold):
		file_name = '/home/hzjiang/Code/Faceness-HK-CNN/IJBA_detection/split%d/test_%d_imlist.txt' % ((n + 1), (n+1))
		fid = open(file_name, 'r')
		image_names = []
		for im_name in fid:
			image_names.append(im_name.strip('\n'))

		imdb.append(image_names)

	return imdb

def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
						default=0, type=int)
	parser.add_argument('--cpu', dest='cpu_mode',
						help='Use CPU mode (overrides --gpu)',
						action='store_true')
	parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]', default='vgg16')
	parser.add_argument('--weight', dest='weight_file', help='Network to use [vgg16]')
	parser.add_argument('--split', dest='split_id', help='split id of IJBA',
						default=1, type=int)	

	args = parser.parse_args()

	return args

if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals

	args = parse_args()

	# NETS['vgg16'][1] = args.solver

	# prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
	# 						'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
	# caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
	# 						  NETS[args.demo_net][1])

	prototxt = 'models/face/VGG16/faster_rcnn_end2end/test.prototxt'
	caffemodel = args.weight_file

	if not os.path.isfile(caffemodel):
		raise IOError(('{:s} not found.\nDid you run ./data/script/'
					   'fetch_faster_rcnn_models.sh?').format(caffemodel))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu_id)
		cfg.GPU_ID = args.gpu_id
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	print '\n\nLoaded network {:s}'.format(caffemodel)

	data_dir = '/data2/hzjiang/Data/CS2'
	work_dir = '/data2/hzjiang/WorkingData/CS2-faster-rcnn'
	force_new = True

	CONF_THRESH = 0.85
	NMS_THRESH = 0.15

	imdb = get_imdb_ijba(data_dir)

	# Warmup on a dummy image
	im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _= im_detect(net, im)

	split_id = args.split_id
	image_names = imdb[args.split_id - 1]

	# detection file
	dets_file_name = 'ijba_res/ijba-det-fold-%d.txt' % (split_id)
	fid = open(dets_file_name, 'w')
	sys.stdout.write('%s ' % (split_id))

	for idx, im_name in enumerate(image_names):
		# Load the demo image
		mat_name = os.path.splitext(im_name)[0] + '.mat'

		# print os.path.join(work_dir, mat_name)

		if (not force_new) and os.path.exists(os.path.join(work_dir, mat_name)):
			res = sio.loadmat(os.path.join(work_dir, mat_name))
			boxes = res['boxes']
			scores = res['scores']
		else:
			# im_path = im_name + '.jpg'
			im = cv2.imread(os.path.join(data_dir, im_name))

			# # Detect all object classes and regress object bounds
			# timer = Timer()
			# timer.tic()
			scores, boxes = im_detect(net, im)
			# timer.toc()
			# print ('Detection took {:.3f}s for '
			#        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

			dir_name, tmp_im_name = os.path.split(im_name)
			if not os.path.exists(os.path.join(work_dir, dir_name)):
				os.makedirs(os.path.join(work_dir, dir_name))

			res = {'boxes': boxes, 'scores': scores}
			sio.savemat(os.path.join(work_dir, mat_name), res)

		cls_ind = 1
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes,
						  cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]

		keep = np.where(dets[:, 4] > CONF_THRESH)
		dets = dets[keep]

		dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
		dets[:, 3] = dets[:, 3] - dets[:, 1] + 1

		fid.write(im_name + '\n')
		fid.write(str(dets.shape[0]) + '\n')
		for j in xrange(dets.shape[0]):
			fid.write('%f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], dets[j, 2], dets[j, 3], dets[j, 4]))


		if ((idx + 1) % 20) == 0:
			sys.stdout.write('%.3f ' % ((idx + 1) / len(image_names) * 100))
			sys.stdout.flush()

	print ''
	fid.close()

	# os.system('cp ./fddb_res/*.txt ~/Code/FDDB/results')

			