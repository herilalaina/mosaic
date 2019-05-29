from setuptools import find_packages, setup
from package import Package

setup(name='mosaic',
      version='0.1',
      description='General algorithm configurator',
      url='https://github.com/herilalaina/mosaic',
      author='Herilalaina Rakotoarison',
      author_email='heri@lri.fr',
      license='BSD-3',
      packages=find_packages(),
      include_package_data=True,
      cmdclass={
        "package": Package
      }
)
