from setuptools import setup, find_packages
long_description = open('README.md').read()

setup(
        name='nlptools',
        version='0.1',
        description='Some tools for NLP',
        long_description=long_description,
        author='outenki',
        author_email='outenki@gmail.com',
        url='https://github.com/outenki/nlptools',
        python_requires='>=3.6',
        packages=find_packages(
                exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
        )
)
