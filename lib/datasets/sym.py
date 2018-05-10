__author__ = 'yuxiang'

import os
import datasets
import datasets.sym
import datasets.imdb
import cPickle
import numpy as np
import cv2
import PIL
import sys
import scipy
from fcn.config import cfg
from utils.pose_error import *
from utils.se3 import *
from utils.cython_bbox import bbox_overlaps
from transforms3d.quaternions import quat2mat, mat2quat
from rpn_layer.generate_anchors import generate_anchors

class sym(datasets.imdb):
    def __init__(self, image_set, sym_path = None):
        datasets.imdb.__init__(self, 'sym_' + image_set)

        self._sym_path = self._get_default_path() if sym_path is None \
                            else sym_path
        self._data_path = os.path.join(self._sym_path, 'data')

        self._classes = ('__background__', 'cube')
        self._class_colors = [(255, 255, 255), (255, 0, 0)]
        self._class_weights = [1, 100]
        self._symmetry = [0, 1]
        self._cls_index = -1

        self._points, self._points_all = self._load_object_points()
        self._extents = self._load_object_extents()

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._sym_path), \
                'sym path does not exist: {}'.format(self._sym_path)


    # image
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path = os.path.join(self._data_path, index + '-color' + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # depth
    def depth_path_at(self, i):
        """
        Return the absolute path to depth i in the image sequence.
        """
        return self.depth_path_from_index(self.image_index[i])

    def depth_path_from_index(self, index):
        """
        Construct an depth path from the image's "index" identifier.
        """
        depth_path = os.path.join(self._data_path, index + '-depth' + self._image_ext)
        assert os.path.exists(depth_path), \
                'Path does not exist: {}'.format(depth_path)
        return depth_path

    # label
    def label_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.label_path_from_index(self.image_index[i])

    def label_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        label_path = os.path.join(self._data_path, index + '-label' + self._image_ext)
        assert os.path.exists(label_path), \
                'Path does not exist: {}'.format(label_path)
        return label_path

    # camera pose
    def metadata_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.metadata_path_from_index(self.image_index[i])

    def metadata_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        metadata_path = os.path.join(self._data_path, index + '-meta.mat')
        assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._sym_path, 'train.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index


    def _get_default_path(self):
        """
        Return the default path where KITTI is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'SYM')


    def _load_object_points(self):

        points = [[] for _ in xrange(len(self._classes))]
        num = np.inf

        for i in xrange(1, len(self._classes)):
            point_file = os.path.join(self._sym_path, 'models', self._classes[i] + '.xyz')
            print point_file
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((self.num_classes, num, 3), dtype=np.float32)
        for i in xrange(1, len(self._classes)):
            points_all[i, :, :] = points[i][:num, :]

        return points[1], points_all


    def _load_object_extents(self):

        extent_file = os.path.join(self._sym_path, 'extents.txt')
        assert os.path.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)

        extents = np.zeros((self.num_classes, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)

        return extents


    def compute_class_weights(self):

        print 'computing class weights'
        num_classes = self.num_classes
        count = np.zeros((num_classes,), dtype=np.int64)
        for index in self.image_index:
            # label path
            label_path = self.label_path_from_index(index)
            im = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            for i in xrange(num_classes):
                I = np.where(im == i)
                count[i] += len(I[0])

        for i in xrange(num_classes):
            self._class_weights[i] = min(float(count[0]) / float(count[i]), 10.0)
            print self._classes[i], self._class_weights[i]


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
            print 'class weights: ', roidb[0]['class_weights']
            return roidb

        # self.compute_class_weights()

        gt_roidb = [self._load_sym_annotation(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_sym_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # depth path
        depth_path = self.depth_path_from_index(index)

        # label path
        label_path = self.label_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)
        
        return {'image': image_path,
                'depth': depth_path,
                'label': label_path,
                'meta_data': metadata_path,
                'class_colors': self._class_colors,
                'class_weights': self._class_weights,
                'cls_index': -1,
                'flipped': False}


    def labels_to_image(self, im, labels):
        class_colors = self._class_colors
        height = labels.shape[0]
        width = labels.shape[1]
        image_r = np.zeros((height, width), dtype=np.float32)
        image_g = np.zeros((height, width), dtype=np.float32)
        image_b = np.zeros((height, width), dtype=np.float32)

        for i in xrange(len(class_colors)):
            color = class_colors[i]
            I = np.where(labels == i)
            image_r[I] = color[0]
            image_g[I] = color[1]
            image_b[I] = color[2]

        image = np.stack((image_r, image_g, image_b), axis=-1)

        return image.astype(np.uint8)


    def evaluate_result(self, im_ind, segmentation, gt_labels, meta_data, output_dir):

        # make matlab result dir
        import scipy.io
        mat_dir = os.path.join(output_dir, 'mat')
        if not os.path.exists(mat_dir):
            os.makedirs(mat_dir)

        # evaluate segmentation
        n_cl = self.num_classes
        hist = np.zeros((n_cl, n_cl))

        gt_labels = gt_labels.astype(np.float32)
        sg_labels = segmentation['labels']
        hist += self.fast_hist(gt_labels.flatten(), sg_labels.flatten(), n_cl)

        # per-class IU
        print 'per-class segmentation IoU'
        intersection = np.diag(hist)
        union = hist.sum(1) + hist.sum(0) - np.diag(hist)
        index = np.where(union > 0)[0]
        for i in range(len(index)):
            ind = index[i]
            print '{} {}'.format(self._classes[ind], intersection[ind] / union[ind])

        threshold = 0.1 * np.linalg.norm(self._extents[1, :])

        # evaluate pose
        if cfg.TEST.POSE_REG:
            rois = segmentation['rois']
            poses = segmentation['poses']
            poses_new = segmentation['poses_refined']
            poses_icp = segmentation['poses_icp']

            # save matlab result
            results = {'labels': sg_labels, 'rois': rois, 'poses': poses, 'poses_refined': poses_new, 'poses_icp': poses_icp}
            filename = os.path.join(mat_dir, '%04d.mat' % im_ind)
            print filename
            scipy.io.savemat(filename, results, do_compression=True)

            poses_gt = meta_data['poses']
            if len(poses_gt.shape) == 2:
                poses_gt = np.reshape(poses_gt, (3, 4, 1))
            num = poses_gt.shape[2]

            for j in xrange(num):
                if meta_data['cls_indexes'][j] != 1:
                    continue
                cls = self._classes[1]
                print cls
                print 'gt pose'
                print poses_gt[:, :, j]

                for k in xrange(rois.shape[0]):
                    if rois[k, 1] != meta_data['cls_indexes'][j]:
                        continue

                    print 'estimated pose'
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses[k, :4])
                    RT[:, 3] = poses[k, 4:7]
                    print RT

                    # quaternion loss
                    print mat2quat(poses_gt[:3, :3, j])
                    print mat2quat(RT[:3, :3])
                    d = mat2quat(poses_gt[:3, :3, j]).dot(mat2quat(RT[:3, :3]))
                    loss = 1 - d * d
                    print 'quaternion loss {}'.format(loss)

                    if cfg.TEST.POSE_REFINE:
                        print 'translation refined pose'
                        RT_new = np.zeros((3, 4), dtype=np.float32)
                        RT_new[:3, :3] = quat2mat(poses_new[k, :4])
                        RT_new[:, 3] = poses_new[k, 4:7]
                        print RT_new

                        print 'ICP refined pose'
                        RT_icp = np.zeros((3, 4), dtype=np.float32)
                        RT_icp[:3, :3] = quat2mat(poses_icp[k, :4])
                        RT_icp[:, 3] = poses_icp[k, 4:7]
                        print RT_icp

                    error_rotation = re(RT[:3, :3], poses_gt[:3, :3, j])
                    print 'rotation error: {}'.format(error_rotation)

                    error_translation = te(RT[:, 3], poses_gt[:, 3, j])
                    print 'translation error: {}'.format(error_translation)

                    if cls == 'eggbox' and error_rotation > 90:
                        RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                        RT_sym = se3_mul(RT, RT_z)
                        print 'eggbox rotation error after symmetry: {}'.format(re(RT_sym[:3, :3], poses_gt[:3, :3, j]))
                        error_reprojection = reproj(meta_data['intrinsic_matrix'], RT_sym[:3, :3], RT_sym[:, 3], \
                            poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                    else:
                        error_reprojection = reproj(meta_data['intrinsic_matrix'], RT[:3, :3], RT[:, 3], \
                            poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                    print 'reprojection error: {}'.format(error_reprojection)

                    # compute pose error
                    if cls == 'eggbox' or cls == 'glue':
                        error = adi(RT[:3, :3], RT[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                    else:
                        error = add(RT[:3, :3], RT[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                    print 'average distance error: {}\n'.format(error)

                    if cfg.TEST.POSE_REFINE:
                        error_rotation_new = re(RT_new[:3, :3], poses_gt[:3, :3, j])
                        print 'rotation error new: {}'.format(error_rotation_new)

                        error_translation_new = te(RT_new[:, 3], poses_gt[:, 3, j])
                        print 'translation error new: {}'.format(error_translation_new)

                        if cls == 'eggbox' and error_rotation_new > 90:
                            RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                            RT_sym = se3_mul(RT_new, RT_z)
                            print 'eggbox rotation error new after symmetry: {}'.format(re(RT_sym[:3, :3], poses_gt[:3, :3, j]))
                            error_reprojection_new = reproj(meta_data['intrinsic_matrix'], RT_sym[:3, :3], RT_sym[:, 3], \
                                poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        else:
                            error_reprojection_new = reproj(meta_data['intrinsic_matrix'], RT_new[:3, :3], RT_new[:, 3], \
                                poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        print 'reprojection error new: {}'.format(error_reprojection_new)

                        if cls == 'eggbox' or cls == 'glue':
                            error_new = adi(RT_new[:3, :3], RT_new[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        else:
                            error_new = add(RT_new[:3, :3], RT_new[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        print 'average distance error new: {}\n'.format(error_new)

                        error_rotation_icp = re(RT_icp[:3, :3], poses_gt[:3, :3, j])
                        print 'rotation error icp: {}'.format(error_rotation_icp)

                        error_translation_icp = te(RT_icp[:, 3], poses_gt[:, 3, j])
                        print 'translation error icp: {}'.format(error_translation_icp)

                        if cls == 'eggbox' and error_rotation_icp > 90:
                            RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                            RT_sym = se3_mul(RT_icp, RT_z)
                            print 'eggbox rotation error icp after symmetry: {}'.format(re(RT_sym[:3, :3], poses_gt[:3, :3, j]))
                            error_reprojection_icp = reproj(meta_data['intrinsic_matrix'], RT_sym[:3, :3], RT_sym[:, 3], \
                                poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        else:
                            error_reprojection_icp = reproj(meta_data['intrinsic_matrix'], RT_icp[:3, :3], RT_icp[:, 3], \
                                poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        print 'reprojection error icp: {}'.format(error_reprojection_icp)

                        if cls == 'eggbox' or cls == 'glue':
                            error_icp = adi(RT_icp[:3, :3], RT_icp[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        else:
                            error_icp = add(RT_icp[:3, :3], RT_icp[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        print 'average distance error icp: {}'.format(error_icp)

                    print 'threshold: {}'.format(threshold)


    def evaluate_result_detection(self, im_ind, detection, meta_data, output_dir):

        # make matlab result dir
        import scipy.io
        mat_dir = os.path.join(output_dir, 'mat')
        if not os.path.exists(mat_dir):
            os.makedirs(mat_dir)

        if 'few' in self._image_set:
            threshold = 0.1 * self._diameters[self._cls_index - 1]
        else:
            threshold = 0.1 * np.linalg.norm(self._extents[1, :])

        # evaluate pose
        if cfg.TEST.POSE_REG:
            rois = detection['rois']
            poses = detection['poses']

            # save matlab result
            results = {'rois': rois, 'poses': poses}
            filename = os.path.join(mat_dir, '%04d.mat' % im_ind)
            print filename
            scipy.io.savemat(filename, results, do_compression=True)

            poses_gt = meta_data['poses']
            if len(poses_gt.shape) == 2:
                poses_gt = np.reshape(poses_gt, (3, 4, 1))
            num = poses_gt.shape[2]

            for j in xrange(num):
                if meta_data['cls_indexes'][j] != 1:
                    continue
                cls = self._classes[1]
                print cls
                print 'gt pose'
                print poses_gt[:, :, j]

                for k in xrange(rois.shape[0]):
                    if rois[k, 0] != meta_data['cls_indexes'][j]:
                        continue

                    print 'estimated pose'
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses[k, :4])
                    RT[:, 3] = poses[k, 4:7]
                    print RT

                    # quaternion loss
                    print mat2quat(poses_gt[:3, :3, j])
                    print mat2quat(RT[:3, :3])
                    d = mat2quat(poses_gt[:3, :3, j]).dot(mat2quat(RT[:3, :3]))
                    loss = 1 - d * d
                    print 'quaternion loss {}'.format(loss)

                    error_rotation = re(RT[:3, :3], poses_gt[:3, :3, j])
                    print 'rotation error: {}'.format(error_rotation)

                    error_translation = te(RT[:, 3], poses_gt[:, 3, j])
                    print 'translation error: {}'.format(error_translation)

                    if cls == 'eggbox' and error_rotation > 90:
                        RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                        RT_sym = se3_mul(RT, RT_z)
                        print 'eggbox rotation error after symmetry: {}'.format(re(RT_sym[:3, :3], poses_gt[:3, :3, j]))
                        error_reprojection = reproj(meta_data['intrinsic_matrix'], RT_sym[:3, :3], RT_sym[:, 3], \
                            poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                    else:
                        error_reprojection = reproj(meta_data['intrinsic_matrix'], RT[:3, :3], RT[:, 3], \
                            poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                    print 'reprojection error: {}'.format(error_reprojection)

                    # compute pose error
                    if cls == 'eggbox' or cls == 'glue':
                        error = adi(RT[:3, :3], RT[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                    else:
                        error = add(RT[:3, :3], RT[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                    print 'average distance error: {}\n'.format(error)

                    print 'threshold: {}'.format(threshold)


    def evaluate_segmentations(self, segmentations, output_dir):
        print 'evaluating segmentations'
        # compute histogram
        n_cl = self.num_classes
        hist = np.zeros((n_cl, n_cl))

        # make image dir
        image_dir = os.path.join(output_dir, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # make matlab result dir
        import scipy.io
        mat_dir = os.path.join(output_dir, 'mat')
        if not os.path.exists(mat_dir):
            os.makedirs(mat_dir)

        count_all = 0
        count_correct = 0
        count_correct_refined = 0
        count_correct_icp = 0
        count_correct_pixel = 0
        count_correct_pixel_refined = 0
        count_correct_pixel_icp = 0
        if 'few' in self._image_set:
            threshold = 0.1 * self._diameters[self._cls_index - 1]
        else:
            threshold = 0.1 * np.linalg.norm(self._extents[1, :])

        # for each image
        for im_ind, index in enumerate(self.image_index):
            # read ground truth labels
            im = cv2.imread(self.label_path_from_index(index), cv2.IMREAD_UNCHANGED)
            gt_labels = im.astype(np.float32)
            if 'test' in self._image_set:
                I = np.where(gt_labels == self._cls_index)
                gt_labels[:, :] = 0
                gt_labels[I[0], I[1]] = 1

            # predicated labels
            if not segmentations[im_ind]:
                filename = os.path.join(mat_dir, '%04d.mat' % im_ind)
                results_mat = scipy.io.loadmat(filename)
                sg_labels = results_mat['labels']
            else:
                sg_labels = segmentations[im_ind]['labels']
            hist += self.fast_hist(gt_labels.flatten(), sg_labels.flatten(), n_cl)

            # evaluate pose
            if cfg.TEST.POSE_REG:
                # load meta data
                meta_data = scipy.io.loadmat(self.metadata_path_from_index(index))
                meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
                ind = np.where(meta_data['cls_indexes'] == self._cls_index)[0]
                meta_data['cls_indexes'][:] = 0
                meta_data['cls_indexes'][ind] = 1

                if not segmentations[im_ind]:
                    rois = results_mat['rois']
                    poses = results_mat['poses']
                    poses_new = results_mat['poses_refined']
                    poses_icp = results_mat['poses_icp']
                else:
                    rois = segmentations[im_ind]['rois']
                    poses = segmentations[im_ind]['poses']
                    poses_new = segmentations[im_ind]['poses_refined']
                    poses_icp = segmentations[im_ind]['poses_icp']

                # save matlab result
                results = {'labels': sg_labels, 'rois': rois, 'poses': poses, 'poses_refined': poses_new, 'poses_icp': poses_icp}
                filename = os.path.join(mat_dir, '%04d.mat' % im_ind)
                print filename
                scipy.io.savemat(filename, results, do_compression=True)

                poses_gt = meta_data['poses']
                if len(poses_gt.shape) == 2:
                    poses_gt = np.reshape(poses_gt, (3, 4, 1))
                num = poses_gt.shape[2]

                for j in xrange(num):
                    if meta_data['cls_indexes'][j] <= 0:
                        continue
                    cls = self._classes[int(meta_data['cls_indexes'][j])]
                    count_all += 1

                    for k in xrange(rois.shape[0]):
                        if rois[k, 1] != meta_data['cls_indexes'][j]:
                            continue

                        RT = np.zeros((3, 4), dtype=np.float32)
                        RT[:3, :3] = quat2mat(poses[k, :4])
                        RT[:, 3] = poses[k, 4:7]

                        if cfg.TEST.POSE_REFINE:
                            RT_new = np.zeros((3, 4), dtype=np.float32)
                            RT_new[:3, :3] = quat2mat(poses_new[k, :4])
                            RT_new[:, 3] = poses_new[k, 4:7]

                            RT_icp = np.zeros((3, 4), dtype=np.float32)
                            RT_icp[:3, :3] = quat2mat(poses_icp[k, :4])
                            RT_icp[:, 3] = poses_icp[k, 4:7]

                        error_rotation = re(RT[:3, :3], poses_gt[:3, :3, j])
                        error_translation = te(RT[:, 3], poses_gt[:, 3, j])

                        if cls == 'eggbox' and error_rotation > 90:
                            RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                            RT_sym = se3_mul(RT, RT_z)
                            error_reprojection = reproj(meta_data['intrinsic_matrix'], RT_sym[:3, :3], RT_sym[:, 3], \
                                poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        else:
                            error_reprojection = reproj(meta_data['intrinsic_matrix'], RT[:3, :3], RT[:, 3], \
                                poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)

                        if error_reprojection < 5:
                            count_correct_pixel += 1

                        # compute pose error
                        if cls == 'eggbox' or cls == 'glue':
                            error = adi(RT[:3, :3], RT[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        else:
                            error = add(RT[:3, :3], RT[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)

                        if error < threshold:
                            count_correct += 1

                        if cfg.TEST.POSE_REFINE:
                            error_rotation_new = re(RT_new[:3, :3], poses_gt[:3, :3, j])
                            error_translation_new = te(RT_new[:, 3], poses_gt[:, 3, j])

                            if cls == 'eggbox' and error_rotation_new > 90:
                                RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                                RT_sym = se3_mul(RT_new, RT_z)
                                error_reprojection_new = reproj(meta_data['intrinsic_matrix'], RT_sym[:3, :3], RT_sym[:, 3], \
                                    poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                            else:
                                error_reprojection_new = reproj(meta_data['intrinsic_matrix'], RT_new[:3, :3], RT_new[:, 3], \
                                    poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)

                            if error_reprojection_new < 5:
                                count_correct_pixel_refined += 1

                            if cls == 'eggbox' or cls == 'glue':
                                error_new = adi(RT_new[:3, :3], RT_new[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                            else:
                                error_new = add(RT_new[:3, :3], RT_new[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)

                            if error_new < threshold:
                                 count_correct_refined += 1

                            error_rotation_icp = re(RT_icp[:3, :3], poses_gt[:3, :3, j])
                            error_translation_icp = te(RT_icp[:, 3], poses_gt[:, 3, j])

                            if cls == 'eggbox' and error_rotation_icp > 90:
                                RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                                RT_sym = se3_mul(RT_icp, RT_z)
                                error_reprojection_icp = reproj(meta_data['intrinsic_matrix'], RT_sym[:3, :3], RT_sym[:, 3], \
                                    poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                            else:
                                error_reprojection_icp = reproj(meta_data['intrinsic_matrix'], RT_icp[:3, :3], RT_icp[:, 3], \
                                    poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)

                            if error_reprojection_icp < 5:
                                count_correct_pixel_icp += 1

                            if cls == 'eggbox' or cls == 'glue':
                                error_icp = adi(RT_icp[:3, :3], RT_icp[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                            else:
                                error_icp = add(RT_icp[:3, :3], RT_icp[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)

                            if error_icp < threshold:
                                 count_correct_icp += 1


            '''
            # label image
            rgba = cv2.imread(self.image_path_from_index(index), cv2.IMREAD_UNCHANGED)
            image = rgba[:,:,:3]
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            image[I[0], I[1], :] = 255
            label_image = self.labels_to_image(image, sg_labels)

            # save image
            filename = os.path.join(image_dir, '%04d.png' % im_ind)
            print filename
            cv2.imwrite(filename, label_image)
            '''
            '''
            # save matlab result
            labels = {'labels': sg_labels}
            filename = os.path.join(mat_dir, '%04d.mat' % im_ind)
            print filename
            scipy.io.savemat(filename, labels)
            #'''

        # overall accuracy
        acc = np.diag(hist).sum() / hist.sum()
        print 'overall accuracy', acc
        # per-class accuracy
        acc = np.diag(hist) / hist.sum(1)
        print 'mean accuracy', np.nanmean(acc)
        # per-class IU
        print 'per-class IU'
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        for i in range(n_cl):
            print '{} {}'.format(self._classes[i], iu[i])
        print 'mean IU', np.nanmean(iu)
        freq = hist.sum(1) / hist.sum()
        print 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum()

        filename = os.path.join(output_dir, 'segmentation.txt')
        with open(filename, 'wt') as f:
            for i in range(n_cl):
                f.write('{:f}\n'.format(iu[i]))

        filename = os.path.join(output_dir, 'confusion_matrix.txt')
        with open(filename, 'wt') as f:
            for i in range(n_cl):
                for j in range(n_cl):
                    f.write('{:f} '.format(hist[i, j]))
                f.write('\n')

        # pose accuracy
        if cfg.TEST.POSE_REG:

            print 'correct poses reprojection: {}, all poses: {}, accuracy: {}'.format(count_correct_pixel, count_all, float(count_correct_pixel) / float(count_all))

            if cfg.TEST.POSE_REFINE:
                print 'correct poses reprojection after refinement: {}, all poses: {}, accuracy: {}'.format(count_correct_pixel_refined, count_all, float(count_correct_pixel_refined) / float(count_all))

                print 'correct poses reprojection after ICP: {}, all poses: {}, accuracy: {}'.format(count_correct_pixel_icp, count_all, float(count_correct_pixel_icp) / float(count_all))


            print 'correct poses: {}, all poses: {}, accuracy: {}'.format(count_correct, count_all, float(count_correct) / float(count_all))

            if cfg.TEST.POSE_REFINE:
                print 'correct poses after refinement: {}, all poses: {}, accuracy: {}'.format(count_correct_refined, count_all, float(count_correct_refined) / float(count_all))

                print 'correct poses after ICP: {}, all poses: {}, accuracy: {}'.format(count_correct_icp, count_all, float(count_correct_icp) / float(count_all))


    def evaluate_detections(self, detections, output_dir):
        print 'evaluating detections'

        # make matlab result dir
        import scipy.io
        mat_dir = os.path.join(output_dir, 'mat')
        if not os.path.exists(mat_dir):
            os.makedirs(mat_dir)

        count_all = 0
        count_correct = 0
        count_correct_pixel = 0
        if 'few' in self._image_set:
            threshold = 0.1 * self._diameters[self._cls_index - 1]
        else:
            threshold = 0.1 * np.linalg.norm(self._extents[1, :])

        # for each image
        for im_ind, index in enumerate(self.image_index):

            # evaluate pose
            if cfg.TEST.POSE_REG:
                # load meta data
                meta_data = scipy.io.loadmat(self.metadata_path_from_index(index))
                meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
                ind = np.where(meta_data['cls_indexes'] == self._cls_index)[0]
                meta_data['cls_indexes'][:] = 0
                meta_data['cls_indexes'][ind] = 1

                if not detections[im_ind]:
                    rois = results_mat['rois']
                    poses = results_mat['poses']
                else:
                    rois = detections[im_ind]['rois']
                    poses = detections[im_ind]['poses']

                # save matlab result
                results = {'rois': rois, 'poses': poses}
                filename = os.path.join(mat_dir, '%04d.mat' % im_ind)
                print filename
                scipy.io.savemat(filename, results, do_compression=True)

                poses_gt = meta_data['poses']
                if len(poses_gt.shape) == 2:
                    poses_gt = np.reshape(poses_gt, (3, 4, 1))
                num = poses_gt.shape[2]

                for j in xrange(num):
                    if meta_data['cls_indexes'][j] <= 0:
                        continue
                    cls = self._classes[int(meta_data['cls_indexes'][j])]
                    count_all += 1

                    for k in xrange(rois.shape[0]):
                        if rois[k, 0] != meta_data['cls_indexes'][j]:
                            continue

                        RT = np.zeros((3, 4), dtype=np.float32)
                        RT[:3, :3] = quat2mat(poses[k, :4])
                        RT[:, 3] = poses[k, 4:7]

                        error_rotation = re(RT[:3, :3], poses_gt[:3, :3, j])
                        error_translation = te(RT[:, 3], poses_gt[:, 3, j])

                        if cls == 'eggbox' and error_rotation > 90:
                            RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                            RT_sym = se3_mul(RT, RT_z)
                            error_reprojection = reproj(meta_data['intrinsic_matrix'], RT_sym[:3, :3], RT_sym[:, 3], \
                                poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        else:
                            error_reprojection = reproj(meta_data['intrinsic_matrix'], RT[:3, :3], RT[:, 3], \
                                poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)

                        if error_reprojection < 5:
                            count_correct_pixel += 1

                        # compute pose error
                        if cls == 'eggbox' or cls == 'glue':
                            error = adi(RT[:3, :3], RT[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)
                        else:
                            error = add(RT[:3, :3], RT[:, 3], poses_gt[:3, :3, j], poses_gt[:, 3, j], self._points)

                        if error < threshold:
                            count_correct += 1

        # pose accuracy
        if cfg.TEST.POSE_REG:

            print 'correct poses reprojection: {}, all poses: {}, accuracy: {}'.format(count_correct_pixel, count_all, float(count_correct_pixel) / float(count_all))
            print 'correct poses: {}, all poses: {}, accuracy: {}'.format(count_correct, count_all, float(count_correct) / float(count_all))

if __name__ == '__main__':
    d = datasets.sym('test')
    res = d.roidb
    from IPython import embed; embed()
