import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import time


if __name__ == '__main__':

    model = SqueezeNet()
    model.load_weights('../model/squeezenet_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)


    start = time.time()

    img = image.load_img('../images/cat.jpeg', target_size=(227, 227))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    duration = time.time() - start
    print('Predicted:', decode_predictions(preds))
