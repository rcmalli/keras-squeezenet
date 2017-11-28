import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import keras
import unittest



class SqueezeNetTests(unittest.TestCase):

    def testModelInit(self):
        model = SqueezeNet()
        self.assertIsNotNone(model)
    def testTFwPrediction(self):
        keras.backend.set_image_dim_ordering('tf')
        model = SqueezeNet()
        img = image.load_img('images/cat.jpeg', target_size=(227, 227))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds)
        #print('Predicted:', decoded_preds)
        self.assertIn(decoded_preds[0][0][1], 'tabby')
        #self.assertAlmostEqual(decode_predictions(preds)[0][0][2], 0.82134342)
    def testTHPrediction(self):
        keras.backend.set_image_dim_ordering('th')
        model = SqueezeNet()
        img = image.load_img('images/cat.jpeg', target_size=(227, 227))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds)
        #print('Predicted:', decoded_preds)
        self.assertIn(decoded_preds[0][0][1], 'tabby')
        #self.assertAlmostEqual(decode_predictions(preds)[0][0][2], 0.82134342)

if __name__ == '__main__':
    unittest.main()