from Cython.Build import cythonize
import numpy
from setuptools.extension import Extension
from setuptools import setup, find_packages


# define an extension that will be cythonized and compiled
ext = Extension(name="GADS", sources=["gads/GADS.pyx"],include_dirs=[numpy.get_include()])
with open("requirements.txt", "r") as fh:
   requirements = fh.readlines()
reqs = [req for req in requirements if req[:2] != "# "]
setup(
    name='GADS',
    packages=find_packages(),
    version="0.0.3",
    description="graph algorithm for seams detection. and tool for plastron detection.",
    ext_modules=cythonize(ext),
    #install_requires=reqs,
    include_package_data=True,
package_data={
        "": ["*.h5", "*.rst"],
    }
)