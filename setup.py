from setuptools import setup, find_packages
from codecs import open
from os import path
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='lucifer-ml-dev',
    packages=['luciferml', 'luciferml.supervised',
              'luciferml.supervised.utils'],
    version='0.0.04',
    license='MIT',
    description="Automated ML by d4rk-lucif3r",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Arsh Anwar',
    author_email="lucifer78908@gmail.com",
    url="https://github.com/d4rk-lucif3r/LuciferML",
    keywords=['luciferML', 'AutoML', 'Python'],
    install_requires=open("requirements.txt").readlines(),
    # install_requires=[
        # "numpy",
        # "pandas",
        # "scipy",
        # "seaborn",
        # "matplotlib",
        # "scikit-learn",
        # "imblearn",
        # "xgboost",
        # "tensorflow",
    # ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
