from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='AMA',
    url='https://github.com/dherrera1911/accuracy_maximization_analysis',
    author='Daniel Herrera-Esposito',
    author_email='dherrera1911@gmail.com',
    # Needed to actually package something
    packages=['ama_library'],
    # Needed for dependencies
    install_requires=[
      'numpy',
      'torch',
      'matplotlib',
      'pycircstat'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Python implementation of accuracy maximization analysis',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
