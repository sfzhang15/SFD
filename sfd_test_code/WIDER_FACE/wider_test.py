import numpy as np
import cv2
import scipy.io as sio


# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/sfd_test_code
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
import caffe


def detect_face(net, image, shrink):
    if shrink != 1:
        image = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

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
    det_xmin = image.shape[1] * detections[0, 0, :, 3] / shrink
    det_ymin = image.shape[0] * detections[0, 0, :, 4] / shrink
    det_xmax = image.shape[1] * detections[0, 0, :, 5] / shrink
    det_ymax = image.shape[0] * detections[0, 0, :, 6] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


def multi_scale_test(net, image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(net, image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets


def write_to_txt(f, det):
    f.write('{:s}\n'.format(event[0][0].encode('utf-8') + '/' + im_name + '.jpg'))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in xrange(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4]
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))


if __name__ == '__main__':

    caffe.set_device(3)
    caffe.set_mode_gpu()
    model_def = 'models/VGGNet/WIDER_FACE/SFD/deploy.prototxt'
    model_weights = 'models/VGGNet/WIDER_FACE/SFD/SFD.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    subset = 'val' # val or test
    if subset is 'val':
        wider_face = sio.loadmat('./sfd_test_code/WIDER_FACE/wider_face_val.mat')    # Val set
    else:
        wider_face = sio.loadmat('./sfd_test_code/WIDER_FACE/wider_face_test.mat')   # Test set
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    Path = 'Path to the images of WIDER FACE'
    save_path = './sfd_test_code/WIDER_FACE/eval_tools_old-version/sfd' + '_' + subset + '/'
    for index, event in enumerate(event_list):
        filelist = file_list[index][0]
        if not os.path.exists(save_path + event[0][0].encode('utf-8')):
            os.makedirs(save_path + event[0][0].encode('utf-8'))

        for num, file in enumerate(filelist):
            im_name = file[0][0].encode('utf-8')
            Image_Path = Path + im_name[:] + '.jpg'
            image = caffe.io.load_image(Image_Path)

            max_im_shrink = (0x7fffffff / 577.0 / (image.shape[0] * image.shape[1])) ** 0.5 # the max size of input image for caffe
            shrink = max_im_shrink if max_im_shrink < 1 else 1

            det0 = detect_face(net, image, shrink)  # origin test
            det1 = flip_test(net, image, shrink)    # flip test
            [det2, det3] = multi_scale_test(net, image, max_im_shrink)  #multi-scale test

            # merge all test results via bounding box voting
            det = np.row_stack((det0, det1, det2, det3))
            dets = bbox_vote(det)

            f = open(save_path + event[0][0].encode('utf-8') + '/' + im_name + '.txt', 'w')
            write_to_txt(f, dets)
            print('event:%d num:%d' % (index + 1, num + 1))
