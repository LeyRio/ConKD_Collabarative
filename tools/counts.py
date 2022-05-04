# generate adj file
import os
import xml.dom.minidom
import numpy as np
import pickle
 
xml_path = os.path.join(os.path.dirname(os.path.abspath("__file__")),'appendix/VOCdevkit/voc2012/VOC2012/Annotations/')
files = os.listdir(xml_path)

gt_dict = {}
cocurrent_matrix =  np.zeros((20,20))
counts_matrix = np.zeros(20)
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']
class_idx = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4, 'bus':5, 'car':6,
            'cat':7, 'chair':8, 'cow':9, 'diningtable':10, 'dog':11, 'horse':12,
            'motorbike':13, 'person':14, 'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18,
            'tvmonitor':19}
countss = 0
idx = list()
if __name__ == '__main__':
    for xm in files:
        xmlfile = xml_path + xm
        dom = xml.dom.minidom.parse(xmlfile)  # 读取xml文档
        root = dom.documentElement  # 得到文档元素对象
        filenamelist = root.getElementsByTagName("filename")
        filename = filenamelist[0].childNodes[0].data
        objectlist = root.getElementsByTagName("object")
        ##
        countss = 0
        idx = list()
        for objects in objectlist:
            namelist = objects.getElementsByTagName("name")
            objectname = namelist[0].childNodes[0].data
            if objectname == '-':
                print(filename)
            # 统计每个类各自出现的数量
            if objectname in gt_dict:
                gt_dict[objectname] += 1
            else:
                gt_dict[objectname] = 1
            if class_idx[objectname] not in idx:
                countss = countss + 1
                idx.append(class_idx[objectname])
        if countss > 1:
            #print(idx)
            for i in range(0,len(idx)-1):
                for j in range(i+1,len(idx)):
                    #print(i,j)
                    cocurrent_matrix[idx[i]][idx[j]] +=1
                    cocurrent_matrix[idx[j]][idx[i]]+=1
    for i,j in gt_dict.items():
        counts_matrix[class_idx[i]] = j
    ans = dict()
    ans['nums'] = counts_matrix
    ans['adj'] = cocurrent_matrix

 
    f = open('voc_adj_2012.pkl', 'wb')
    pickle.dump(ans, f)
    f.close()

    
    