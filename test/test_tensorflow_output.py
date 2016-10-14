
from scipy import misc
import copy
import numpy as np
from squeezenet import get_squeezenet
import time


if __name__ == '__main__':

    model = get_squeezenet(1000, dim_ordering='tf')
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.load_weights('../model/sqn_tf.h5', by_name=True)


    start = time.time()
    im = misc.imread('../images/cat.jpeg')

    im = misc.imresize(im, (227, 227)).astype(np.float32)
    aux = copy.copy(im)
    im[:, :, 0] = aux[:, :, 2]
    im[:, :, 2] = aux[:, :, 0]

    # Remove image mean
    im[:, :, 0] -= 104.006
    im[:, :, 1] -= 116.669
    im[:, :, 2] -= 122.679

    #im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)

    res = model.predict(im)
    classes = []
    with open('classes.txt', 'r') as list_:
        for line in list_:
            classes.append(line.rstrip('\n'))
    duration = time.time() - start
    print "{} s to get output".format(duration)

    print 'class: ' + classes[np.argmax(res[0])] + ' acc: ' + str(res[0][np.argmax(res[0])])
