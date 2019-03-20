import cv2
import os
import numpy as np
import pandas as pd

'''
with open('F:/input/faster_rcnn_by_keras/tmp.txt','w') as w:
    with open('F:/input/faster_rcnn_by_keras/train.txt','r') as f:
    	for line in f:
            (filename,x1,y1,x2,y2,class_name) = line.strip().split(',')
            img = cv2.imread(filename)
            (height,width) = img.shape[:2]
            x1 = int(np.around(float(x1) * width))
            x2 = int(np.around(float(x2) * width))
            y1 = int(np.around(float(y1)* height))
            y2 = int(np.around(float(y2) * height))
            w.write(filename + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + class_name + '\n')

'''

def main():
    class_descriptions_boxable_df = pd.read_csv('../input/faster_rcnn_by_keras/class-descriptions-boxable.csv',names = ['LabelName','class_name'] )
    validation_annotations_bbox_df = pd.read_csv('../input/faster_rcnn_by_keras/validation-annotations-bbox.csv')
    validation_images_df = pd.read_csv('../input/faster_rcnn_by_keras/validation-images.csv')

    tmp = pd.merge(class_descriptions_boxable_df,validation_annotations_bbox_df,on=['LabelName'])
    #tmp = pd.merge(tmp,validation_images_df,on=['LabelName'])
    tmp = tmp[(tmp['class_name'] == 'Person' )|( tmp['class_name'] == 'Mobile phone') | (tmp['class_name'] == 'Car')]

    with open('F:/input/faster_rcnn_by_keras/train.txt','w') as w:
        for index, row in tmp.iterrows():
            filename = '../input/faster_rcnn_by_keras/image/' + row['ImageID'] + '.jpg'
            if not os.path.isfile(filename):
                print(filename)
                continue
            x1 = row['XMin']
            x2 = row['XMax']
            y1 = row['YMin']
            y2 = row['YMax']

            class_name = row['class_name']

            img = cv2.imread(filename)
            (height,width) = img.shape[:2]
            #print(height)
            x1 = int(np.around(float(x1) * width))
            x2 = int(np.around(float(x2) * width))
            y1 = int(np.around(float(y1)* height))
            y2 = int(np.around(float(y2) * height))
            w.write(filename + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + class_name + '\n')

if __name__ == "__main__":
    main()


