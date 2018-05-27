import rospy
import message_filters
import cv2
import numpy as np
from fcn.config import cfg
from utils.blob import im_list_to_blob, pad_im, unpad_im, add_noise
from normals import gpu_normals
from std_msgs.msg import String
from sensor_msgs.msg import Image


def test_ros(sess, network, imdb, meta_data, cfg, rgb, depth, cv_bridge, count):
    if depth.encoding == '32FC1':
        depth_32 = cv_bridge.imgmsg_to_cv2(depth) * 1000
        depth_cv = np.array(depth_32, dtype=np.uint16)
    elif depth.encoding == '16UC1':
        depth_cv = cv_bridge.imgmsg_to_cv2(depth)
    else:
        rospy.logerr_throttle(1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(depth.encoding))
        return

    # image
    im = cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

    # write images
    filename = 'images/%06d-color.png' % count
    cv2.imwrite(filename, im)

    filename = 'images/%06d-depth.png' % count
    cv2.imwrite(filename, depth_cv)
    print filename

    # run network
    labels, probs, vertex_pred, rois, poses = im_segment_single_frame(sess, network, im, depth_cv, meta_data, \
            imdb._extents, imdb._points_all, imdb._symmetry, imdb.num_classes, cfg)
    poses_icp = []

    im_label = imdb.labels_to_image(im, labels)

    if cfg.TEST.VISUALIZE:
        vertmap = extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
        vis_segmentations_vertmaps(im, depth_cv, im_label, imdb._class_colors, \
                    vertmap, labels, rois, poses, poses_icp, meta_data['intrinsic_matrix'], \
                    imdb.num_classes, imdb._points_all, cfg)


def extract_vertmap(im_label, vertex_pred, extents, num_classes):
    height = im_label.shape[0]
    width = im_label.shape[1]
    vertmap = np.zeros((height, width, 3), dtype=np.float32)

    for i in xrange(1, num_classes):
        I = np.where(im_label == i)
        if len(I[0]) > 0:
            start = 3 * i
            end = 3 * i + 3
            vertmap[I[0], I[1], :] = vertex_pred[I[0], I[1], start:end]
    vertmap[:, :, 2] = np.exp(vertmap[:, :, 2])
    return vertmap


def get_image_blob(im, im_depth, meta_data, cfg):
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
    im_orig -= cfg.PIXEL_MEANS
    processed_ims = []
    im_scale_factors = []
    assert len(cfg.TEST.SCALES_BASE) == 1
    im_scale = cfg.TEST.SCALES_BASE[0]

    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # depth
    im_orig = im_depth.astype(np.float32, copy=True)
    im_orig = im_orig / im_orig.max() * 255
    im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
    im_orig -= cfg.PIXEL_MEANS

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
    blob_depth = im_list_to_blob(processed_ims_depth, 3)
        
    return blob, blob_depth, blob_normal, np.array(im_scale_factors)


def im_segment_single_frame(sess, net, im, im_depth, meta_data, extents, points, symmetry, num_classes, cfg):
    """segment image
    """

    # compute image blob
    im_blob, im_depth_blob, im_normal_blob, im_scale_factors = get_image_blob(im, im_depth, meta_data, cfg)
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
    if cfg.INPUT == 'RGBD':
        data_blob = im_blob
        data_p_blob = im_depth_blob
    elif cfg.INPUT == 'COLOR':
        data_blob = im_blob
    elif cfg.INPUT == 'DEPTH':
        data_blob = im_depth_blob
    elif cfg.INPUT == 'NORMAL':
        data_blob = im_normal_blob

    if cfg.INPUT == 'RGBD':
        if cfg.TEST.VERTEX_REG_2D or cfg.TEST.VERTEX_REG_3D:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}
    else:
        if cfg.TEST.VERTEX_REG_2D or cfg.TEST.VERTEX_REG_3D:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.symmetry: symmetry, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}

    sess.run(net.enqueue_op, feed_dict=feed_dict)

    if cfg.TEST.VERTEX_REG_2D:
        if cfg.TEST.POSE_REG:
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



def vis_segmentations_vertmaps(im, im_depth, im_labels, colors, center_map, 
  labels, rois, poses, poses_new, intrinsic_matrix, num_classes, points, cfg):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(3, 4, 1)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')

    # show class label
    ax = fig.add_subplot(3, 4, 9)
    plt.imshow(im_labels)
    ax.set_title('class labels')      

    if cfg.TEST.VERTEX_REG_2D:
        # show centers
        for i in xrange(rois.shape[0]):
            if rois[i, 1] == 0:
                continue
            cx = (rois[i, 2] + rois[i, 4]) / 2
            cy = (rois[i, 3] + rois[i, 5]) / 2
            w = rois[i, 4] - rois[i, 2]
            h = rois[i, 5] - rois[i, 3]
            if not np.isinf(cx) and not np.isinf(cy):
                plt.plot(cx, cy, 'yo')

                # show boxes
                plt.gca().add_patch(
                    plt.Rectangle((cx-w/2, cy-h/2), w, h, fill=False,
                                   edgecolor='g', linewidth=3))
        
    # show vertex map
    ax = fig.add_subplot(3, 4, 10)
    plt.imshow(center_map[:,:,0])
    ax.set_title('centers x')

    ax = fig.add_subplot(3, 4, 11)
    plt.imshow(center_map[:,:,1])
    ax.set_title('centers y')
    
    ax = fig.add_subplot(3, 4, 12)
    plt.imshow(center_map[:,:,2])
    ax.set_title('centers z: {:6f}'.format(poses[0, 6]))

    # show projection of the poses
    if cfg.TEST.POSE_REG:

        ax = fig.add_subplot(3, 4, 3, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(rois.shape[0]):
            cls = int(rois[i, 1])
            if cls > 0:
                # extract 3D points
                x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                x3d[0, :] = points[cls,:,0]
                x3d[1, :] = points[cls,:,1]
                x3d[2, :] = points[cls,:,2]

                # projection
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(poses[i, :4])
                RT[:, 3] = poses[i, 4:7]
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)
                # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)

        ax.set_title('projection')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

        if cfg.TEST.POSE_REFINE:
            ax = fig.add_subplot(3, 4, 4, aspect='equal')
            plt.imshow(im)
            ax.invert_yaxis()
            for i in xrange(rois.shape[0]):
                cls = int(rois[i, 1])
                if cls > 0:
                    # extract 3D points
                    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                    x3d[0, :] = points[cls,:,0]
                    x3d[1, :] = points[cls,:,1]
                    x3d[2, :] = points[cls,:,2]

                    # projection
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses_new[i, :4])
                    RT[:, 3] = poses_new[i, 4:7]
                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)

            ax.set_title('projection refined by ICP')
            ax.invert_yaxis()
            ax.set_xlim([0, im.shape[1]])
            ax.set_ylim([im.shape[0], 0])

    plt.show()
