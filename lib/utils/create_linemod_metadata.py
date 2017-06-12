import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import savemat

visualize = False
generate_metadata = True

# camera intrinsic matrix
k = np.matrix([[ 572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])


def main(dataset):
    base_path = "data/linemod/" + dataset + "/"
    # mesh_file = open(base_path+"OLDmesh.ply", "r")
    point_cloud_file = open(base_path + "object.xyz", "r")
    # transform_file = open(base_path + "transform.dat", "r")

    num_lines, voxelsize = point_cloud_file.readline().split(" ")
    point_cloud = np.ones([int(num_lines), 4, 1], dtype=np.float32)
    for i in xrange(int(num_lines)):
        line = point_cloud_file.readline().split(" ")[0:3]
        for j in xrange(len(line)):
            point_cloud[i, j, 0] = line[j]

    for i in xrange(1236):
        image_path = base_path + "data/color" + str(i) + ".jpg"
        depth_path = base_path + "data/depth" + str(i) + ".dpt"
        rotation_path = base_path + "data/rot" + str(i) + ".rot"
        translation_path = base_path + "data/tra" + str(i) + ".tra"

        rot_matrix = read_from_file(rotation_path)
        translation_matrix = read_from_file(translation_path)
        transformation_matrix = np.concatenate((rot_matrix, translation_matrix), 1)

        center = transform_to_2d(k, transformation_matrix, np.array([np.mean(point_cloud, axis=0)]))
        final_points = transform_to_2d(k, transformation_matrix, point_cloud)

        if visualize:
            fig = plt.figure()
            image = Image.open(image_path)
            ax = fig.add_subplot(111)
            ax.scatter(final_points.transpose()[0], final_points.transpose()[1], marker='.', s=4, alpha=0.5, edgecolors='none')
            ax.scatter(center.transpose()[0], center.transpose()[1], c="w", s=15, alpha=1)
            ax.imshow(image)

            # ax2 = fig.add_subplot(122)
            # ax2.imshow(image)

            fig.tight_layout()
            plt.show()

        if generate_metadata:
            pose_matrix = np.zeros([3,4,1], dtype=np.float32)
            for one in range(3):
                for two in range(4):
                    pose_matrix[one, two, 0] = transformation_matrix[one, two]

            mat_file = {}
            mat_file['factor_depth'] = [[1000]]
            mat_file['poses'] = pose_matrix
            mat_file['intrinsic_matrix'] = k
            mat_file['rotation_translation_matrix'] = np.identity(4)[0:3]
            mat_file['center'] = center
            mat_file['cls_indexes'] = [[1]]
            mat_file['vertmap'] = np.zeros([6, 6, 3])

            savemat(("/home/davidm/DA-RNN/DA-RNN/data/linemod/ape/data/%.5i-meta.mat" % i), mat_file)


def transform_to_2d(k, transformation_matrix, point_cloud):
    transformed_points = np.matmul(np.array(np.matmul(k, transformation_matrix)), point_cloud)
    final_points = np.zeros([transformed_points.shape[0], 2], dtype=np.float32)
    for i in range(final_points.shape[0]):
        final_points[i, 0] = transformed_points[i, 0, 0] / transformed_points[i, 2, 0]
        final_points[i, 1] = transformed_points[i, 1, 0] / transformed_points[i, 2, 0]
    return final_points


def read_from_file(filename):
    file_obj = open(filename, "r")
    num_columns, num_rows = file_obj.readline().replace("\r", "").replace("\n", "").split(" ")
    matrix = np.zeros([int(num_rows), int(num_columns)], dtype=np.float32)
    for i in xrange(int(num_rows)):
        line = file_obj.readline().replace("\r", "").replace("\n", "").split(" ")
        for j in xrange(int(num_columns)):
            matrix[i][j] = float(line[j])
    return matrix


if __name__ == "__main__":
    main("ape_full")
    print "done"