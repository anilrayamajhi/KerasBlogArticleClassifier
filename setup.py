from setuptools import setup

setup(
    name='Keras Blog Article Classifier',
    version='1.0',
    description='Keras Blog Article Classifier',
    include_package_data=True,
    install_requires=[
        'tensorflow==1.3.0',
        'Keras==2.0.9',
        'numpy==1.13.1',
        'scikit-learn==0.19.0',
        'pandas'
    ],
)
