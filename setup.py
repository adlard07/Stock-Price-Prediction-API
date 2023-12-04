from setuptools import find_packages, setup
from typing import List


hyfen_e_dot = '-e.'
def get_requirements(file_path:str)->List[str]:
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n', '')for req in requirements]
        
        if hyfen_e_dot in requirements:
            requirements.remove(hyfen_e_dot)
    return requirements

setup(
    name='Stock-Price-Prediction', 
    version='0.0.1',
    author='Adlard',
    author_email='adelarddcunha07@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)