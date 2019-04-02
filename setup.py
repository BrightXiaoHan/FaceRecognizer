import numpy
from distutils.core import setup
from Cython.Build import cythonize


setup(
    name="torch_insight",
    version="0.1",
    description='This repo is a re-implementation of Arcface[(paper)',
    author='HanBing',
    author_email='beatmight@gmail.com',
    packages=['insight_face', 'insight_face.datasets', 'insight_face.deploy', 'insight_face.network', 'insight_face.train', 'insight_face.utils'],
    install_requires=[
        "torch_mtcnn=0.1"
    ]
)