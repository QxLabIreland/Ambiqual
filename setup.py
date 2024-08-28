from setuptools import setup, find_packages

setup(
    name='Ambiqual',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.7.2',
        'numpy>=1.23.5',
        'opencv_python>=4.10.0.84',
        'scipy>=1.14.0',
        'soundfile>=0.12.1',
        'pandas>=2.2.2'
    ],
    extras_require={
        'dev': [
        ],
    },
    entry_points={
        'console_scripts': [
            # 'my-command=mypackage.module:function',
        ],
    },
    author='Davoud Shariat Panah',
    author_email='davoodsp@gmail.com',
    description='A brief description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dspanah/pyAmbiqual',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
