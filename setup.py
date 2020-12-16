from setuptools import setup, find_packages

setup(
  name = 'cross-transformers-pytorch',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Cross Transformers - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/cross-transformers-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'cross attention',
    'few shot learning'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
