from setuptools import find_packages, setup


# Function to read the requirements from the requirements.txt file
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip()]

setup(
    name='quantminer',
    version='0.5.2',
    description='Data/Pattern Mining Algorithms for Financial Data',
    author='Jerry Inyang',
    author_email='jerprog0@gmail.com',
    packages=find_packages(),  # Automatically finds your package
    install_requires=read_requirements()  # Reads requirements dynamically from requirements.txt
)