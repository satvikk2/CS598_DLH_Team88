from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='CS598-DL4H-TransformEHR-Team88',
    version='1.0.0',
    author='Anikesh Haran , Satvik Kulkarni, Changhua Zhan',
    author_email='anikesh2@illinois.edu, satvikk2@illinois.edu, zhan36@illinois.edu',
    description='CS598-DL4H-TransformEHR-Team88',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    install_requires=[
        'pyhealth',
        'torch',
        'numpy',
        'datetime'
    ],
    entry_points={
        'console_scripts': [
            'transform_ehr_cli=transform_ehr:run',
        ],
    }
)