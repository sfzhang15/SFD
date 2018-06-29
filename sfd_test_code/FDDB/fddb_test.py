import numpy as np
import cv2


# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {sfd_root}/sfd_test_code/FDDB
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
import caffe


caffe.set_device(2)
caffe.set_mode_gpu()
model_def = 'models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt'
model_weights = 'models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)


count = 0
Path = 'Path to the images of FDDB'
f = open('./sfd_test_code/FDDB/sfd_fddb_dets.txt', 'wt')
for Name in open('./sfd_test_code/FDDB/fddb_img_list.txt'):
    Image_Path = Path + Name[:-1] + '.jpg'
    image = caffe.io.load_image(Image_Path)
    heigh = image.shape[0]
    width = image.shape[1]

    net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    detections = net.forward()['detection_out']
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    keep_index = np.where(det_conf >= 0)[0]
    det_conf = det_conf[keep_index]
    det_xmin = det_xmin[keep_index]
    det_ymin = det_ymin[keep_index]
    det_xmax = det_xmax[keep_index]
    det_ymax = det_ymax[keep_index]

    f.write('{:s}\n'.format(Name[:-1]))
    f.write('{:.1f}\n'.format(det_conf.shape[0]))
    for i in xrange(det_conf.shape[0]):
        xmin = det_xmin[i] * width
        ymin = det_ymin[i] * heigh
        xmax = det_xmax[i] * width
        ymax = det_ymax[i] * heigh
        score = det_conf[i]
        f.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.
                format(xmin, ymin, (xmax-xmin+1), (ymax-ymin+1), score))
    count += 1
    print('%d/2845' % count)
