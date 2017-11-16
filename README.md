# S³FD: Single Shot Scale-invariant Face Detector

By [Shifeng Zhang](http://www.cbsr.ia.ac.cn/users/sfzhang/)

### Introduction

S³FD is a real-time face detector, which performs superiorly on various scales of faces with a single deep neural network, especially for small faces. For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1708.05237).


### Contents
1. [Preparation](#preparation)
2. [Eval](#eval)
3. [Train](#train)

### Preparation
1. Get the [SSD](https://github.com/weiliu89/caffe/tree/ssd) code. We will call the directory that you cloned Caffe into `$SFD_ROOT`
  ```Shell
  git clone https://github.com/weiliu89/caffe.git
  cd $SFD_ROOT
  git checkout ssd
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional)
  make runtest -j8
  ```

3. Download our pre-trained [model](https://drive.google.com/open?id=17Tlop86wpA87BOKsWRPIzak2dwUYxcB5) and merge it with the folder `$SFD_ROOT/models`.

4. Download our above [sfd_test_code](https://github.com/sfzhang15/SFD/archive/master.zip) folder and put it in the `$SFD_ROOT`.

5. Download [AFW](http://www.ics.uci.edu/~xzhu/face/), [PASCAL face](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html), [FDDB](http://vis-www.cs.umass.edu/fddb/index.html) and [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) datasets.

6. Download the [EVALUATION TOOLBOX](https://bitbucket.org/marcopede/face-eval) for evaluation.

### Eval
1. Evaluate our model on AFW.
  ```Shell
  cd $SFD_ROOT/sfd_test_code/AFW
  # You must modify the "Path" in the afw_test.py to your AFW path. 
  # It will creat sfd_afw_dets.txt and put it in the EVALUATION TOOLBOX to evalute.
  python afw_test.py
  ```

2. Evaluate our model on PASCAL face.
  ```Shell
  cd $SFD_ROOT/sfd_test_code/PASCAL_face
  # You must modify the "Path" in the pascal_test.py to your PASCAL_face path. 
  # It will creat sfd_pascal_dets.txt and put it in the EVALUATION TOOLBOX to evalute.
  python pascal_test.py
  ```

3. Evaluate our model on FDDB.
  ```Shell
  cd $SFD_ROOT/sfd_test_code/FDDB
  # You must modify the "Path" in the fddb_test.py to your FDDB path.
  # It will creat sfd_fddb_dets.txt.
  python fddb_test.py
  # Fitting the dets from rectangle box to ellipse box.
  # It will creat sfd_fddb_dets_fit.txt and put it in the FDDB evalution code to evalute.
  cd fddb_from_rectangle_to_ellipse
  matlab -nodesktop -nosplash -nojvm -r "run fitting.m;quit;"
  # If you want to get the results of FDDB in our paper, you should use our 'FDDB_annotation_ellipseList_new.txt'
  ```

4. Evaluate our model on WIDER FACE.
  ```Shell
  cd $SFD_ROOT/sfd_test_code/WIDER_FACE
  # You must modify the path in the wider_test.py to your WIDERFACE path. 
  # It will creat detection results in the "eval_tools_old-version" folder.
  python wider_test.py
  # If you want to get the results of val set in our paper, you should use the provided "eval_tools_old-version". 
  # Or you can use latest eval_tools of WIDER FACE.
  # There is a slight difference between them, since the annotation used for the evaluation is slightly change around March 2017.
  ```

### Train

1. Follow the intruction of SSD to create the lmdb of WIDER FACE.

2. Modify the data augmentation code of SSD to make sure that it does not change the image ratio.

3. Modify the anchor match code of SSD to implement the 'scale compensation anchor matching strategy'.

4. Train the model.
