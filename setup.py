from setuptools import find_packages, setup

setup(
    name='quantminer',
    version='0.4.1', 
    description='Data/Pattern Mining Algorithms for Financial Data',
    author='Jerry Inyang',
    author_email='jerprog0@gmail.com',
    packages=find_packages(),  # Automatically finds your package
    install_requires=[  # List any dependencies here
        "kneed",
        "quantstats",
        "pandas",
        "numpy",
        "scikit-learn",
        "sktime",
        "dtaidistance",
        "matplotlib",
        "tslearn ",
        "plotly"
    ]
)

# /Users/jerryinyang/anaconda3/bin/python