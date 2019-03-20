from model_pipeline.config import Config 
from model_pipeline.moeld_utils import rpn_layer, classifier_layer
from utils.util import rpn_to_roi,calc_iou,get_img_output_length
from data_pipeline.data_pipeline import get_data,get_anchor_gt
import pandas as pd
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy
from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from model_pipeline.loss import rpn_loss_regr,rpn_loss_cls,class_loss_regr,class_loss_cls
import numpy as np
import argparse
import time
from model_pipeline.vgg_16_model import Vgg16Model

import cv2
import os

C = Config()

epoch_length = 1000

# 构建网络
def build_model(classes_count):
    # 输入节点
    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))

    # vgg-16特征提取层
    vgg16Model = Vgg16Model()
    shared_layers = vgg16Model.nn_base(img_input,True)# shape : (600/16.1000/16,512)

    # RPN层
    rpn = rpn_layer(shared_layers, C.num_anchors)# x_class.shape: (600/16,1000/16,9), x_regr.shape(600/16,1000/16,9*4), base_layers.shape (600/16.1000/16,512)
    classifier = classifier_layer(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count))# RPN中的二分类：是否是目标还是背景

    model_rpn = Model(img_input, rpn[:2])# rpn网络
    model_classifier = Model([img_input, roi_input], classifier)

    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    if not os.path.isfile(C.model_path):
        #If this is the begin of the training, load the pre-traind base network such as vgg-16
        try:
            print('This is the first time of your training')
            print('loading weights from {}'.format(C.base_net_weights))
            model_rpn.load_weights(C.base_net_weights, by_name=True)
            model_classifier.load_weights(C.base_net_weights, by_name=True)
        except:
            print('Could not load pretrained model weights. Weights can be found in the keras application folder \
                https://github.com/fchollet/keras/tree/master/keras/applications')

        # Create the record.csv file to record losses, acc and mAP
        record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
    else:
        # If this is a continued training, load the trained model from before
        print('Continue training based on previous trained model')
        print('Loading weights from {}'.format(C.model_path))
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)

        # Load the records
        record_df = pd.read_csv(record_path)

        r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
        r_class_acc = record_df['class_acc']
        r_loss_rpn_cls = record_df['loss_rpn_cls']
        r_loss_rpn_regr = record_df['loss_rpn_regr']
        r_loss_class_cls = record_df['loss_class_cls']
        r_loss_class_regr = record_df['loss_class_regr']
        r_curr_loss = record_df['curr_loss']
        r_elapsed_time = record_df['elapsed_time']
        r_mAP = record_df['mAP']

        print('Already train %dK batches'% (len(record_df)))
    return model_rpn,model_classifier,model_all ,record_df

def log(losses,iter_num,record_df,model_all,rpn_accuracy_for_epoch):
    result = False
    if iter_num == epoch_length:
        loss_rpn_cls = np.mean(losses[:, 0])
        loss_rpn_regr = np.mean(losses[:, 1])
        loss_class_cls = np.mean(losses[:, 2])
        loss_class_regr = np.mean(losses[:, 3])
        class_acc = np.mean(losses[:, 4])
        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
        rpn_accuracy_for_epoch = []
        if C.verbose:
            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
            print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
            print('Loss RPN classifier: {}'.format(loss_rpn_cls))
            print('Loss RPN regression: {}'.format(loss_rpn_regr))
            print('Loss Detector classifier: {}'.format(loss_class_cls))
            print('Loss Detector regression: {}'.format(loss_class_regr))
            print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
            print('Elapsed time: {}'.format(time.time() - start_time))
            elapsed_time = (time.time()-start_time)/60
        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
        iter_num = 0
        start_time = time.time()
        if curr_loss < best_loss:
            if C.verbose:
                print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
            best_loss = curr_loss
            model_all.save_weights(C.model_path)
        new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3), 
                   'class_acc':round(class_acc, 3), 
                   'loss_rpn_cls':round(loss_rpn_cls, 3), 
                   'loss_rpn_regr':round(loss_rpn_regr, 3), 
                   'loss_class_cls':round(loss_class_cls, 3), 
                   'loss_class_regr':round(loss_class_regr, 3), 
                   'curr_loss':round(curr_loss, 3), 
                   'elapsed_time':round(elapsed_time, 3), 
                   'mAP': 0}
        record_df = record_df.append(new_row, ignore_index=True)
        record_df.to_csv(record_path, index=0)

        result = True
    
    return result

def main():
    #1.获取训练数据
    train_imgs, classes_count, class_mapping = get_data(C.train_path)
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    C.class_mapping = class_mapping

    #2.构造数据生成器
    data_gen_train = get_anchor_gt(train_imgs, C, get_img_output_length, mode='train')
    '''
    X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)
    print(X)
    '''

    #3.拼装faster-crnn模型
    model_rpn,model_classifier,model_all,record_df = build_model(classes_count)

    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(C.num_anchors), rpn_loss_regr(C.num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls, class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')


    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    if len(record_df)==0:
        best_loss = np.Inf
    else:
        best_loss = np.min(r_curr_loss)

    #4.开始训练
    start_time = time.time()
 
    iter_num = 0
    for epoch_num in range(C.num_epochs):
        progbar = generic_utils.Progbar(epoch_length)
        print('epoch_num：' + str(epoch_num))
        while True:
            try:
                X, Y, img_data, debug_img, debug_num_pos = next(data_gen_train)
                
                #4.1 训练rnp模型 得到rpn分类损失和rpn分类损失 [,loss_rpn_cls,loss_rpn_regr]
                loss_rpn = model_rpn.train_on_batch(X, Y)
                

                #4.2 获取rpn的预测结果 [rpn_cls, rpn_regr]
                P_rpn = model_rpn.predict_on_batch(X)

                #将rpn的输出转化成ROI bboxes
                R = rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300) # shape=(300,4)

                # 将 bbox从 (x1,y1,x2,y2) 格式 转化成 (x,y,w,h) 
                # X2: 转化后的bbox
                # Y1: 分类信息
                # Y2: 回归信息
                # IouS：iou
                X2, Y1, Y2, IouS = calc_iou(R, img_data, C, class_mapping)

                # If X2 is None means there are no matching bboxes
                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue
                # 
                neg_samples = np.where(Y1[0, :, -1] == 1)#反例anchors
                pos_samples = np.where(Y1[0, :, -1] == 0)#正例anchors
                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []
                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))
                #均衡正例和负例数据
                if C.num_rois > 1:
                    # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                    
                    # Randomly choose (num_rois - num_pos) neg samples
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                    
                    # Save all the pos and neg samples in sel_samples
                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
                
                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                            ('final_cls', np.mean(losses[:iter_num, 2])), ('final_regr', np.mean(losses[:iter_num, 3]))])
            
                if log(losses,iter_num,record_df,model_all,rpn_accuracy_for_epoch):
                    break
            except Exception as e:
                print('Exception: {}'.format(e))
                continue

    
    print('Training complete, exiting.')
            



        
        


    


if __name__ == "__main__":
    C.use_horizontal_flips = True
    C.use_vertical_flips = True
    C.rot_90 = True

    C.record_path = 'model/record.csv'
    C.model_path = 'model/model_frcnn_vgg.h5'
    C.num_rois = 4

    C.base_net_weights = '../model/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    C.train_path = '../input/faster_rcnn_by_keras/train.txt' 
    
    main()
    #img = cv2.imread('../input/faster_rcnn_by_keras/image/110487ec7e9be60a.jpg')
    #print(img.shape)
    