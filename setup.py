from setuptools import setup, find_packages

setup(
    name='libf1tenth',
    version='0.0.1',
    description='F1Tenth library',
    url='https://github.com/JChunX/libf1tenth',
    author='JChunX',
    author_email='jchunx[at]seas[dot]upenn[dot]edu',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'pandas'],
)