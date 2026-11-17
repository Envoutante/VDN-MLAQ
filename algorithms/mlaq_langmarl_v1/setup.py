from setuptools import setup, find_packages

setup(
    name='robotcollab',
    version='1.0',
    package_dir={"": "environments"},
    packages=find_packages(where="environments", include=["rocobench", "rocobench.*"]),
    description='Code for Paper: RoCo: Multi-Robot Collaboration with Large Language Models', 
    url='https://github.com/MandiZhao/robot-collab.git',
    author='Mandi Zhao',
    install_requires=[],
)