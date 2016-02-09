from distutils.core import setup

setup(
    name = 'semisup-metrics',
    version = '0.1',
    author = 'Marc Claesen',
    author_email = 'marc.claesen@esat.kuleuven.be',
    py_modules = ['semisup_metrics'],
    url = 'http://semisup-metrics.readthedocs.org',
    license = 'LICENSE.txt',
    description = 'Performance metrics for semi-supervised classification',
    long_description = open('README.rst').read(),
    classifiers = ['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Education',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python',
                   'License :: OSI Approved :: BSD License',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Information Analysis'
                   ],
    platforms = ['any'],
    keywords = ['machine learning', 'semi-supervised learning', 'classification',
                'performance metrics', 'roc'],
    install_requires = ['numpy', 'scipy', 'optunity']
)
