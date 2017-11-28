from setuptools import setup
exec(open('keras_squeezenet/version.py').read())
setup(name='keras_squeezenet',
      version=__version__,
      description='Squeezenet implementation with Keras framework',
      url='https://github.com/rcmalli/keras-squeezenet',
      author='Refik Can MALLI',
      author_email = "mallir@itu.edu.tr",
      license='MIT',
      packages=['keras_squeezenet'],
      zip_safe=False,
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'h5py',
                        'tensorflow',
                        'keras',
                        'six>=1.9.0',
                        'pyyaml'])
