from setuptools import setup, find_packages

# def get_packages(filepath: str = './requirements.txt'):
#     with open(filepath, 'r') as file:
#        lines = file.readlines()
#     packages = []
#     for line in lines:
#        stripped_line = line.strip()
#        if (len(stripped_line) > 0) and (not stripped_line.startswith('#')):
#           packages.append(stripped_line)
#     return packages
    
setup(
    name="jlab_datascience_toolkit",
    version="0.1",
    description="JLab datascience toolkit for composable workflows",
    author="JLab DataScience Department",
    author_email="schram@jlab.org, kishan@jlab.org, dianam@jlab.org, dlersch@jlab.org",
    # packages=[find_packages('jlab_datascience_toolkit'),find_packages('jlab_datascience_toolkit/keras')],
    packages=["jlab_datascience_toolkit"],
    # install_requires=get_packages(),
    # python_requires='==3.10.*'
)
