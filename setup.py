from setuptools import setup, find_packages

setup(name='santander',
      version='1.0',
      description='Model for Kaggle project',
      author='Alexey',
      author_email='nomail@nowhere.com',
      url='https://github.com/AlexLuka/SanKagProj',
      packages=find_packages(),
      install_requires=[
          "lightgbm",
          "scikit-learn"
      ]
      )
