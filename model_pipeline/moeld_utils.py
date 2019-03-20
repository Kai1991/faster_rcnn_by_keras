from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from model_pipeline.roi_pooling_conv_lay import RoiPoolingConv


#获取rpn(区域提案网络)层
# 输入
    #base_layers 特征提取层
    #num_anchors anchor的数量
# 输出
    #  x_class 感兴趣的内的二分类，是否有物体
    #  x_regr  bboxes regression
    #  base_layers 特征提取层
def rpn_layer(base_layers,num_anchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers) #shape : (none,600/16,1000/16,512)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x) #shape : (none,600/16,1000/16,9)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x) #shape : (none,600/16,1000/16,9*4)

    return [x_class, x_regr, base_layers]

#获取rpn(区域提案网络)层
# 输入
    # base_layers 特征提取层
    # input_rois (1,num_rois,4)` list of rois, with ordering (x,y,w,h) 感兴趣区域
    # num_rois  感兴趣数量
    # nb_classes 分类数量
# 输出
    #  list(out_class, out_regr) 感兴趣的内的二分类，是否有物体
    #  out_class: classifier layer output  
    #  out_regr: regression layer output 
def classifier_layer(base_layers, input_rois, num_rois, nb_classes = 4):
    """Create a classifier layer"""

    input_shape = (num_rois,7,7,512)

    pooling_regions = 7

    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    # Flatten the convlutional layer and connected to 2 FC and 2 dropout
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]
