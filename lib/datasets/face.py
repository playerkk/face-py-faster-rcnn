# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# import datasets.face
# import os
# import datasets.imdb as imdb
# import xml.dom.minidom as minidom
# import numpy as np
# import scipy.sparse
# import scipy.io as sio
# import utils.cython_bbox
# import cPickle
# import subprocess

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import cv2
import PIL

class face(imdb):
    def __init__(self, image_set, split, devkit_path):
        imdb.__init__(self, 'wider')
        self._image_set = image_set         # {'train', 'test'}
        self._split = split                 # {1, 2, ..., 10}
        self._devkit_path = devkit_path     # /data2/hzjiang/Data/CS2
        # self._data_path = os.path.join(self._devkit_path, 'data')
        self._data_path = self._devkit_path;
        self._classes = ('__background__', # always index 0
                         'face')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.png']
        self._image_index, self._gt_roidb = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, index)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)

        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # # Example path to image set file:
        # # self._data_path + /ImageSets/val.txt
        # # read from file
        # image_set_file = 'split%d/%s_%d_annot.txt' % (self._fold, self._image_set, self._fold)
        # # image_set_file = os.path.join(self._devkit_path, image_set_file)
        # image_set_file = os.path.join('/home/hzjiang/Code/py-faster-rcnn/CS3-splits', image_set_file)
        image_set_file = self._name + '_face_' + self._image_set + '_annot.txt'
        image_set_file = os.path.join(self._devkit_path, image_set_file)

        # image_set_file = 'cs3_rand_train_annot.txt'
        # image_set_file = 'wider_dets_annot_from_cs3_model.txt'
        # image_set_file = 'wider_manual_annot.txt'

        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        image_index = []
        gt_roidb = []
        
        with open(image_set_file) as f:
            # print len(f.lines())
            lines = f.readlines()

            idx = 0
            while idx < len(lines):
                image_name = lines[idx].split('\n')[0]
                image_name = os.path.join('WIDER_%s/images' % self._image_set, image_name)
                # print image_name
                image_ext = os.path.splitext(image_name)[1].lower()
                # print image_ext
                assert(image_ext == '.png' or image_ext == '.jpg' or image_ext == '.jpeg')

                image = PIL.Image.open(os.path.join(self._data_path, image_name))
                imw = image.size[0]
                imh = image.size[1]

                idx += 1
                num_boxes = int(lines[idx])
                # print num_boxes

                boxes = np.zeros((num_boxes, 4), dtype=np.uint16)
                gt_classes = np.zeros((num_boxes), dtype=np.int32)
                overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

                for i in xrange(num_boxes):
                    idx += 1
                    coor = map(float, lines[idx].split())

                    x1 = min(max(coor[0], 0), imw - 1)
                    y1 = min(max(coor[1], 0), imh - 1)
                    x2 = min(max(x1 + coor[2] - 1, 0), imw - 1)
                    y2 = min(max(y1 + coor[3] - 1, 0), imh - 1)

                    if np.isnan(x1):
                        x1 = -1

                    if np.isnan(y1):
                        y1 = -1

                    if np.isnan(x2):
                        x2 = -1

                    if np.isnan(y2):
                        y2 = -1
                        
                    cls = self._class_to_ind['face']
                    boxes[i, :] = [x1, y1, x2, y2]
                    gt_classes[i] = cls
                    overlaps[i, cls] = 1.0

                widths = boxes[:, 2] - boxes[:, 0] + 1
                heights = boxes[:, 3] - boxes[:, 1] + 1
                keep_idx = np.where(np.bitwise_and(widths > 5, heights > 5))

                if len(keep_idx[0]) <= 0:
                    idx += 1
                    continue

                boxes = boxes[keep_idx]
                gt_classes = gt_classes[keep_idx[0]]
                overlaps = overlaps[keep_idx[0], :]



                if not (boxes[:, 2] >= boxes[:, 0]).all():
                    print boxes
                    print image_name

                # print boxes
                assert (boxes[:, 2] >= boxes[:, 0]).all()
                assert (boxes[:, 3] >= boxes[:, 1]).all()

                overlaps = scipy.sparse.csr_matrix(overlaps)
                gt_roidb.append({'boxes' : boxes,
                                'gt_classes': gt_classes,
                                'gt_overlaps' : overlaps,
                                'flipped' : False,
                                'image_name': image_name})
                image_index.append(image_name)

                idx += 1        

            assert(idx == len(lines))

        return image_index, gt_roidb

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        with open(cache_file, 'wb') as fid:
            cPickle.dump(self._gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return self._gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
            print len(roidb)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._devkit_path,
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)

        raw_data = sio.loadmat(filename)['all_boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_face_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of face.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.mat')

        data = sio.loadmat(filename)

        num_objs = data['gt'].shape[0]

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix in xrange(num_objs):
            # Make pixel indexes 0-based
            coor = data['gt'][ix, :]
            x1 = float(coor[0]) - 1
            y1 = float(coor[1]) - 1
            x2 = float(coor[2]) - 1
            y2 = float(coor[3]) - 1
            cls = self._class_to_ind['face']
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        if not (boxes[:, 2] >= boxes[:, 0]).all():
            print boxes
            print filename

        assert (boxes[:, 2] >= boxes[:, 0]).all()

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_inria_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_inria_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.inria('train', '')
    res = d.roidb
    from IPython import embed; embed()
