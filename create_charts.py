import os
import numpy as np
import matplotlib.pyplot as plot
from utils import get_dataset
from tqdm import tqdm

classes = []
number_objects = []
box_height = []
box_width = []
box_area = []


def get_info_from_dataset(dataset):

    #for idx, batch in enumerate(dataset):
    for batch in tqdm(dataset, desc="Calculating..."):
        image = batch['image'].numpy()
        height, width, channel = image.shape
        boxes = batch['groundtruth_boxes'].numpy()
        boxes[:, [0,2]] *= height
        boxes[:, [1,3]] *= width
        bh = boxes[:,2] - boxes[:,0]
        bw = boxes[:,3] - boxes[:,1]

        classes.extend(batch['groundtruth_classes'].numpy())
        number_objects.append(len(batch['groundtruth_classes'].numpy()))
        box_height.extend(bh)
        box_width.extend(bw)
        box_area.extend(bh * bw)
#        if(idx % 10000 == 0):
#            print('Idx ', idx)
        

def plot_charts():
    figure = plot.figure()
    figure.suptitle('Charts')
    
    f, ax = plot.subplots(2, 2)
    f.delaxes(ax[1, 1])
    
    object_class_distribution = np.array(classes)
    object_class_distribution = np.array([(object_class_distribution == 1).sum(), (object_class_distribution == 2).sum(), (object_class_distribution == 4).sum()])
    object_class_distribution = object_class_distribution / float(object_class_distribution.sum())
    ax[0, 0].bar(['vehicle', 'pedestrian', 'cyclist'], object_class_distribution)
    ax[0, 0].set_title("Object class distribution")
    ax[0, 0].set_ylabel("Percentage")
    
    ax[0, 1].hist(box_area)
    ax[0, 1].set_title("Boxes area")
    ax[0, 1].set_xlabel("GroundTruth boxes size (pixels)")
    ax[0, 1].set_ylabel("Number of GroundTruth boxes")
    
    ax[1, 0].hist(number_objects)
    ax[1, 0].set_title("Obj num per img")
    ax[1, 0].set_xlabel("Number of objects per image")
    ax[1, 0].set_ylabel("Number of images")
    plot.tight_layout()
    plot.autoscale()
    plot.savefig('charts.png', bbox_inches="tight")


if __name__ == "__main__":
    try:
        dataset = get_dataset("/home/workspace/data/test/*.tfrecord")
        #print("Dataset size: ", len(list(dataset)))
        get_info_from_dataset(dataset)
        plot_charts()
    except KeyboardInterrupt:
        print("Ploting data")
        plot_charts()
