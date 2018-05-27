import rospy
import message_filters
import cv2
import numpy as np
from fcn.config import cfg
from utils.blob import im_list_to_blob, pad_im, unpad_im, add_noise
from normals import gpu_normals
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
from synthesizer.msg import PoseCNNMsg

class ImageListener:

    def __init__(self, sess, network, imdb, meta_data, cfg):

        self.sess = sess
        self.net = network
        self.imdb = imdb
        self.meta_data = meta_data
        self.cfg = cfg
        self.cv_bridge = CvBridge()
        self.count = 0

        # initialize a node
        rospy.init_node("image_listener")
        self.posecnn_pub = rospy.Publisher('posecnn_result', PoseCNNMsg, queue_size=1)
        self.label_pub = rospy.Publisher('posecnn_label', Image, queue_size=1)
        rgb_sub = message_filters.Subscriber('/camera/rgb/image_color', Image, queue_size=2)
        depth_sub = message_filters.Subscriber('/camera/depth_registered/image', Image, queue_size=2)
        # depth_sub = message_filters.Subscriber('/camera/depth_registered/sw_registered/image_rect_raw', Image, queue_size=2)

        queue_size = 1
        slop_seconds = 0.025
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

    def callback(self, rgb, depth):
        if depth.encoding == '32FC1':
            depth_32 = self.cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        # write images
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        filename = 'images/%06d-color.png' % self.count
        cv2.imwrite(filename, im)

        filename = 'images/%06d-depth.png' % self.count
        cv2.imwrite(filename, depth_cv)
        print filename
        self.count += 1

        # run network
        labels, probs, vertex_pred, rois, poses = self.im_segment_single_frame(self.sess, self.net, im, depth_cv, self.meta_data, \
            self.imdb._extents, self.imdb._points_all, self.imdb._symmetry, self.imdb.num_classes)

        im_label = self.imdb.labels_to_image(im, labels)

        # publish
        msg = PoseCNNMsg()
        msg.height = int(im.shape[0])
        msg.width = int(im.shape[1])
        msg.roi_num = int(rois.shape[0])
        msg.roi_channel = int(rois.shape[1])
        msg.fx = float(self.meta_data['intrinsic_matrix'][0, 0])
        msg.fy = float(self.meta_data['intrinsic_matrix'][1, 1])
        msg.px = float(self.meta_data['intrinsic_matrix'][0, 2])
        msg.py = float(self.meta_data['intrinsic_matrix'][1, 2])
        msg.factor = float(self.meta_data['factor_depth'])
        msg.znear = float(0.25)
        msg.zfar = float(6.0)
        msg.label = self.cv_bridge.cv2_to_imgmsg(labels.astype(np.uint8), 'mono8')
        msg.depth = self.cv_bridge.cv2_to_imgmsg(depth_cv, 'mono16')
        msg.rois = rois.astype(np.float32).flatten().tolist()
        msg.poses = poses.astype(np.float32).flatten().tolist()
        self.posecnn_pub.publish(msg)

        label_msg = self.cv_bridge.cv2_to_imgmsg(im_label)
        label_msg.header.stamp = rospy.Time.now()
        label_msg.header.frame_id = rgb.header.frame_id
        label_msg.encoding = 'rgb8'
        self.label_pub.publish(label_msg)

    def get_image_blob(self, im, im_depth, meta_data):
        """Converts an image into a network input.

        Arguments:
            im (ndarray): a color image in BGR order

        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
               in the image pyramid
        """

        # RGB
        im_orig = im.astype(np.float32, copy=True)
        # mask the color image according to depth
        if self.cfg.EXP_DIR == 'rgbd_scene':
            I = np.where(im_depth == 0)
            im_orig[I[0], I[1], :] = 0

        processed_ims_rescale = []
        im_scale = self.cfg.TEST.SCALES_BASE[0]
        im_rescale = cv2.resize(im_orig / 127.5 - 1, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_rescale.append(im_rescale)

        im_orig -= self.cfg.PIXEL_MEANS
        processed_ims = []
        im_scale_factors = []
        assert len(self.cfg.TEST.SCALES_BASE) == 1

        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

        # depth
        im_orig = im_depth.astype(np.float32, copy=True)
        # im_orig = im_orig / im_orig.max() * 255
        im_orig = np.clip(im_orig / 2000.0, 0, 1) * 255
        im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
        im_orig -= self.cfg.PIXEL_MEANS

        processed_ims_depth = []
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_depth.append(im)

        if cfg.INPUT == 'NORMAL':
            # meta data
            K = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            # normals
            depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
            nmap = gpu_normals.gpu_normals(depth, fx, fy, cx, cy, 20.0, cfg.GPU_ID)
            im_normal = 127.5 * nmap + 127.5
            im_normal = im_normal.astype(np.uint8)
            im_normal = im_normal[:, :, (2, 1, 0)]
            im_normal = cv2.bilateralFilter(im_normal, 9, 75, 75)

            processed_ims_normal = []
            im_orig = im_normal.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_normal = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            processed_ims_normal.append(im_normal)
            blob_normal = im_list_to_blob(processed_ims_normal, 3)
        else:
            blob_normal = []

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims, 3)
        blob_rescale = im_list_to_blob(processed_ims_rescale, 3)
        blob_depth = im_list_to_blob(processed_ims_depth, 3)
        
        return blob, blob_rescale, blob_depth, blob_normal, np.array(im_scale_factors)


    def im_segment_single_frame(self, sess, net, im, im_depth, meta_data, extents, points, symmetry, num_classes):
        """segment image
        """

        # compute image blob
        im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = self.get_image_blob(im, im_depth, meta_data)
        im_scale = im_scale_factors[0]

        # construct the meta data
        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        mdata = np.zeros(48, dtype=np.float32)
        mdata[0:9] = K.flatten()
        mdata[9:18] = Kinv.flatten()
        # mdata[18:30] = pose_world2live.flatten()
        # mdata[30:42] = pose_live2world.flatten()
        meta_data_blob = np.zeros((1, 1, 1, 48), dtype=np.float32)
        meta_data_blob[0,0,0,:] = mdata

        # use a fake label blob of ones
        height = int(im_depth.shape[0] * im_scale)
        width = int(im_depth.shape[1] * im_scale)
        label_blob = np.ones((1, height, width), dtype=np.int32)

        pose_blob = np.zeros((1, 13), dtype=np.float32)
        vertex_target_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)
        vertex_weight_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)

        # forward pass
        if self.cfg.INPUT == 'RGBD':
            data_blob = im_blob
            data_p_blob = im_depth_blob
        elif self.cfg.INPUT == 'COLOR':
            data_blob = im_blob
        elif self.cfg.INPUT == 'DEPTH':
            data_blob = im_depth_blob
        elif self.cfg.INPUT == 'NORMAL':
            data_blob = im_normal_blob

        if self.cfg.INPUT == 'RGBD':
            if self.cfg.TEST.VERTEX_REG_2D or self.cfg.TEST.VERTEX_REG_3D:
                feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                             net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                             net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.poses: pose_blob}
            else:
                feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}
        else:
            if self.cfg.TEST.VERTEX_REG_2D or self.cfg.TEST.VERTEX_REG_3D:
                feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                             net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                             net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.symmetry: symmetry, net.poses: pose_blob}
            else:
                feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}

        sess.run(net.enqueue_op, feed_dict=feed_dict)

        if self.cfg.TEST.VERTEX_REG_2D:
            if self.cfg.TEST.POSE_REG:
                labels_2d, probs, vertex_pred, rois, poses_init, poses_pred = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), \
                              net.get_output('rois'), net.get_output('poses_init'), net.get_output('poses_tanh')])

                # non-maximum suppression
                # keep = nms(rois, 0.5)
                # rois = rois[keep, :]
                # poses_init = poses_init[keep, :]
                # poses_pred = poses_pred[keep, :]
                print rois

                # combine poses
                num = rois.shape[0]
                poses = poses_init
                for i in xrange(num):
                    class_id = int(rois[i, 1])
                    if class_id >= 0:
                        poses[i, :4] = poses_pred[i, 4*class_id:4*class_id+4]
            else:
                labels_2d, probs, vertex_pred, rois, poses = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'), net.get_output('poses_init')])
                print rois
                print rois.shape
                # non-maximum suppression
                # keep = nms(rois[:, 2:], 0.5)
                # rois = rois[keep, :]
                # poses = poses[keep, :]

                #labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
                #vertex_pred = []
                #rois = []
                #poses = []
            vertex_pred = vertex_pred[0, :, :, :]
        else:
            labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
            vertex_pred = []
            rois = []
            poses = []

        return labels_2d[0,:,:].astype(np.int32), probs[0,:,:,:], vertex_pred, rois, poses
