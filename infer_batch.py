import os
import csv
import argparse
import tensorflow as tf
import keras.backend as K

from glob import glob

from lib.io import openpose_from_file, read_segmentation, write_mesh
from model.octopus import Octopus


def main(weights, num, batch_file, opt_pose_steps, opt_shape_steps):
    K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

    model = Octopus(num=num)

    with open(batch_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')

        for name, segm_dir, pose_dir, out_dir in reader:
            print('Processing {}...'.format(name))
            model.load(weights)

            segm_files = sorted(glob(os.path.join(segm_dir, '*.png')))
            pose_files = sorted(glob(os.path.join(pose_dir, '*.json')))

            if len(segm_files) != len(pose_files) or len(segm_files) == len(pose_files) == 0:
                print('> Inconsistent input.')
                continue

            segmentations = [read_segmentation(f) for f in segm_files]
            joints_2d, face_2d = [], []
            for f in pose_files:
                j, f = openpose_from_file(f)

                if len(j) != 25 or len(f) != 70:
                    print('> Invalid keypoints.')
                    continue

                joints_2d.append(j)
                face_2d.append(f)

            if opt_pose_steps:
                print('> Optimizing for pose...')
                model.opt_pose(segmentations, joints_2d, opt_steps=opt_pose_steps)

            if opt_shape_steps:
                print('> Optimizing for shape...')
                model.opt_shape(segmentations, joints_2d, face_2d, opt_steps=opt_shape_steps)

            print('> Estimating shape...')
            pred = model.predict(segmentations, joints_2d)

            write_mesh('{}/{}.obj'.format(out_dir, name), pred['vertices'][0], pred['faces'])

            print('> Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'batch_file',
        type=str,
        help="Batch file")

    parser.add_argument(
        'num',
        type=int,
        help="Number of views per subject")

    parser.add_argument(
        '--opt_steps_pose', '-p', default=10, type=int,
        help="Optimization steps pose")

    parser.add_argument(
        '--opt_steps_shape', '-s', default=25, type=int,
        help="Optimization steps")

    parser.add_argument(
        '--weights', '-w',
        default='weights/octopus_weights.hdf5',
        help='Model weights file (*.hdf5)')

    args = parser.parse_args()
    main(args.weights, args.num, args.batch_file, args.opt_steps_pose, args.opt_steps_shape)
