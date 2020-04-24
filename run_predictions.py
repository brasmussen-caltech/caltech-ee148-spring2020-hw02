import os
import numpy as np
import json
from PIL import Image
from glob import glob


def compute_convolution(I, T, candrows, candcols, stride=2):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''

    '''
    BEGIN YOUR CODE
    '''
    Tx, Ty, Tz = T.shape
    flatT = T.flatten()
    outheat = np.zeros(I.shape[:2])
    mag = np.linalg.norm(flatT)**2
    Tredrange = (np.max(T[:, :, 0])-np.min(T[:, :, 0]))*.8
    for ind, xval in enumerate(candrows):
        yval = candcols[ind]
        if ((xval == 0) | (yval == 0)):
            continue
        extract = I[xval:xval+Tx, yval:yval+Ty, :]
        flatext = extract.flatten()
        if len(flatext) != len(flatT):
            continue
        redrange = np.max(extract[:, :, 0])-np.min(extract[:, :, 0])
        if redrange < Tredrange:
            continue
        outval = np.dot(flatext, flatT)/mag
        outheat[xval, yval] = outval
    '''
    END YOUR CODE
    '''

    return(np.nan_to_num(outheat))


def predict_boxes(heatmap, T, weakened=False):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    '''
    BEGIN YOUR CODE
    '''
    detbbs, detmvals = [], []
    Tx, Ty, Tz = T.shape
    if weakened:
        thr = .995
    else:
        thr = .975
    for nit in range(10000):
        if np.max(heatmap) >= thr:
            goodinds = np.where(heatmap == np.max(heatmap))
            goodrow, goodcol = goodinds
            goodrow, goodcol = goodrow[0], goodcol[0]
            boundbox = [goodrow, goodcol, goodrow+Tx, goodcol+Ty]
            detmvals.append(np.max(heatmap))
            detbbs.append(boundbox)
            botrowind, botcolind = int(
                np.clip(goodrow-12, 0, np.inf)), int(np.clip(goodcol-12, 0, np.inf))
            heatmap[botrowind:goodrow+12, botcolind:goodcol+12] = 0
        else:
            break
    return(detbbs, detmvals)


def trim_results(detbbs, detmvals):
    badinds = []
    #Remove duplicate detection and maintain the highest score in overlapping sets

    for bbind in range(len(detbbs)):
        thisbb = np.array(detbbs[bbind])
        diffs = np.array([np.linalg.norm(thisbb[:2] - x[:2]) for x in detbbs])
        if len(diffs[diffs < 24]) > 0:
            initdupinds = np.where(diffs < 24)[0]
            bestind = initdupinds[np.argmax(np.array(detmvals)[initdupinds])]
            badinds.extend([x for x in initdupinds if x != bestind])
    badinds = list(set(badinds))
    #add in confidences
    goodinds = np.array([x for x in np.arange(
        0, len(detbbs), 1) if x not in badinds])
    for bbind in range(len(detbbs)):
        detbbs[bbind] = [int(x) for x in detbbs[bbind]]
    for gind in goodinds:
        detbbs[gind].append(detmvals[gind])
    for ele in sorted(badinds, reverse=True):
        del detbbs[ele]

    return(detbbs)


def detect_red_light_mf(I, weakened=False):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    masterbbs, masterdetmvals = [], []
    I = np.clip(I[:, :, :3], 1, 255).astype('float32')
    I = np.divide(I, np.expand_dims(
        np.linalg.norm(I, axis=-1), axis=-1))
    redim = I[:, :, 0]
    shiftdiffax0 = abs(redim-np.roll(redim, shift=1, axis=0))
    shiftdiffax1 = abs(redim-np.roll(redim, shift=1, axis=1))
    avdiff = (shiftdiffax0+shiftdiffax1)/2
    avdiff = avdiff**2
    per = np.percentile(avdiff, q=99)
    avdiff /= per
    avdiff = np.clip(avdiff, 0, 1)
    avdiff[avdiff < .08] = 0
    candrows, candcols = np.where(avdiff > 0)
    for Tname in glob('data/templates/*.png'):
        rawT = np.asarray(Image.open(Tname))[:, :, :3]
        T = np.clip(rawT, 1, 255)
        T = np.divide(T, np.expand_dims(np.linalg.norm(T, axis=-1), axis=-1))
        heatmap = compute_convolution(I, T, candrows, candcols, stride=None)
        bbs, detmvals = predict_boxes(heatmap, T, weakened=weakened)
        masterbbs.extend(bbs)
        masterdetmvals.extend(detmvals)
    output = trim_results(masterbbs, masterdetmvals)
    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return(output)


# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# load splits:
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'data/hw02_preds'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I, weakened=True)
    print('Finished ' + file_names_train[i])

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds_train_weakened.json'), 'w') as f:
    json.dump(preds_train, f)

if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path, file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I, weakened=True)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path, 'preds_test_weakened.json'), 'w') as f:
        json.dump(preds_test, f)
