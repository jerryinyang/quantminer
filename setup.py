from setuptools import find_packages, setup


# Function to read the requirements from the requirements.txt file
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]


# Read the README file
def read_readme():
    with open("README.rst", "r") as f:
        return f.read()


setup(
    name="quantminer",
    version="1.0.0",
    description="Data/Pattern Mining Algorithms for Financial Data",
    long_description=read_readme(),
    long_description_content_type="text/x-rst",
    author="Jerry Inyang",
    author_email="jerprog0@gmail.com",
    packages=find_packages(),  # Automatically finds your package
    install_requires=read_requirements(),  # Reads requirements dynamically from requirements.txt
)
