import numpy as np
import cv2


# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {sfd_root}/sfd_test_code/AFW
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
import caffe


caffe.set_device(0)
caffe.set_mode_gpu()
model_def = 'models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt'
model_weights = 'models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)


count = 0
Path = 'Path to the images of AFW'
f = open('./sfd_test_code/AFW/sfd_afw_dets.txt', 'wt')
for Name in open('./sfd_test_code/AFW/afw_img_list.txt'):
    Image_Path = Path + Name[:-1] + '.jpg'
    image = caffe.io.load_image(Image_Path)
    heigh = image.shape[0]
    width = image.shape[1]

    im_shrink = 640.0 / max(image.shape[0], image.shape[1])
    image = cv2.resize(image, None, None, fx=im_shrink, fy=im_shrink, interpolation=cv2.INTER_LINEAR)

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

    for i in xrange(det_conf.shape[0]):
        xmin = int(round(det_xmin[i] * width))
        ymin = int(round(det_ymin[i] * heigh))
        xmax = int(round(det_xmax[i] * width))
        ymax = int(round(det_ymax[i] * heigh))
        # simple fitting to AFW, because the gt box of training data (i.e., WIDER FACE) is longer than the gt box of AFW
        ymin += 0.2 * (ymax - ymin + 1)   
        score = det_conf[i]
        if score < 0:
            continue
        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                format(Name[:-1], score, xmin, ymin, xmax, ymax))
    count += 1
    print('%d/205' % count)
