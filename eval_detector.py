import os
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    if ((len(box_1) > 0) & (len(box_2) > 0)):
        box_2= [int(x) for x in box_2]
        intTLX = np.max([box_1[0], box_2[0]])
        intTLY = np.max([box_1[1], box_2[1]])
        intBRX = np.min([box_1[2], box_2[2]])
        intBRY = np.min([box_1[3], box_2[3]])
        intersect = np.max([0, intBRX - intTLX + 1]) * np.max([0, intBRY - intTLY + 1])
        box_1A = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
        box_2A = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)
        iou = intersect / float(box_1A + box_2A - intersect)
    else:
        if ((len(box_1) == 0) & (len(box_2) == 0)):
            iou=1
        else:
            iou=0
    assert (iou >= 0) and (iou <= 1.0)
    return(iou)


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        for i in range(len(gt)):
            initTP=TP
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if ((iou > iou_thr) & ((pred[j][4]-.995)/.005 >= conf_thr)):
                    TP+=1
                else:
                    if ((j == len(pred) - 1) & ( TP == initTP)):
                        FN+=1
                    else:
                        if (pred[j][4]-.995)/.005 >= conf_thr:
                            FP+=1




    '''
    END YOUR CODE
    '''

    return TP, FP, FN


# set a path for predictions and annotations:
preds_path = 'data/hw02_preds'
gts_path = 'data/hw02_annotations'

# load splits:
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data.
'''
with open(os.path.join(preds_path, 'preds_train_weakened.json'), 'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'), 'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''

    with open(os.path.join(preds_path, 'preds_test_weakened.json'), 'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'), 'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.


# using (ascending) list of confidence scores as thresholds
# Range scale confidence scores to be 0-1 rather than .975-1
def gen_pr_curves(preds,gt,title=False):
    def parse_bb(bb):
        if len(bb) == 0:
            return(1)
        else:
            return((bb[4]-.995)/.005)
    bbs = [preds[fname] for fname in preds]
    ctlist=[]
    for subbbs in bbs:
        candcts=[parse_bb(x) for x in subbbs]
        ct=np.mean(candcts)
        ctlist.append(ct)
    confidence_thrs=np.sort(np.array(ctlist))
    tp = np.zeros(len(confidence_thrs))
    fp= np.zeros(len(confidence_thrs))
    fn = np.zeros(len(confidence_thrs))
    for iou_thresh in [0.5,0.25,0.75]:
        for i, conf_thr in enumerate(confidence_thrs):
            print(i)
            tp[i], fp[i], fn[i] = compute_counts(
                preds, gt, iou_thr=iou_thresh, conf_thr=conf_thr)
        print(tp,fp,fn)
        prec=tp/(tp+fp)
        rec=tp/(tp+fn)
        print(prec,rec)
        plt.scatter(rec,prec,label='IOU Thresh ' + str(iou_thresh) + ' Max Recall is ' + str(np.around(np.max(rec),2)),linestyle='--',marker='x',s=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()
gen_pr_curves(preds_train,gts_train,title='PR Curve Train, Weakened')
# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')
    gen_pr_curves(preds_test,gts_test,title='PR Curve Test, Weakened')
