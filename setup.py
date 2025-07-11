from setuptools import setup, find_packages
#from src.guv_calcs.io import get_version
from pathlib import Path
import re
VERSION_RE = re.compile(r'__version__\s*=\s*"([^"]+)"')

def read_version(path) -> str:
    return VERSION_RE.search(path.read_text()).group(1)
    
with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as thefile: 
    requirements = thefile.readlines()
setup(
        name="guv_calcs",
        url="https://github.com/jvbelenky/guv-calcs",
        version=read_version(Path(__file__).parent / 'src' / 'guv_calcs' / '_version.py'),
        author="J. Vivian Belenky",
        author_email="j.vivian.belenky@outlook.com",
        description="A library for carrying out fluence and irradiance calculations for germicidal UV (GUV) applications.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "License :: OSI Approved :: MIT License",
        ],
        packages=find_packages('src'),
        package_dir={'': 'src'},
        package_data={'guv_calcs': ['data/*', 'data/*/*']},
        zip_safe=True,
        python_requires=">=3.8",
        install_requires=requirements,
    )
