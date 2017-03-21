# keras-squeezenet
SqueezeNet v1.1 Implementation using Keras Functional Framework 2.0

This [network model](https://github.com/rcmalli/keras-squeezenet/blob/master/images/SqueezeNet.png) has AlexNet accuracy with small footprint (5.1 MB)
Pretrained models are converted from original Caffe network.

~~~bash

pip install keras_squeezenet

~~~

### News

- Project is now up-to-date with the new Keras version (2.0).

- Old Implementation is still available at 'keras1' branch.

### Library Versions

- Keras v2.0+
- Tensorflow 1.0+

### Example Usage

~~~python
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

model = SqueezeNet()

img = image.load_img('../images/cat.jpeg', target_size=(227, 227))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))

~~~


### References

1) [Keras Framework](www.keras.io)

2) [SqueezeNet Official Github Repo](https://github.com/DeepScale/SqueezeNet)

3) [SqueezeNet Paper](http://arxiv.org/abs/1602.07360)


### Licence 

MIT License 