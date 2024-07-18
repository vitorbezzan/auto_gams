"""
setup_.py - Installs dependencies and creates architecture dependant wheel file.
"""
import sys
from os import environ, path

from setuptools import setup
from setuptools_cythonize import get_cmdclass

from bezzanlabs import __version__

here = path.abspath(path.dirname(__file__))


def publish():
    """
    Function to forbid publishing of this repo in the PyPi default library
    """
    argv = sys.argv
    blacklist = ["register", "upload"]

    for command in blacklist:
        if command in argv:
            values = {"command": command}
            print(
                "[ERROR] -This package cant be uploaded or registered on PyPi, exiting..."
                % values
            )
            sys.exit(-1)


publish()

# get the dependencies and installs
with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = [x.strip() for x in f if x.strip()]

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    readme = f.read()

# If compiling, uses the best optimisation available
environ["CFLAGS"] = "-Ofast"

setup(
    cmdclass=get_cmdclass(),
    name="bezzanlabs.auto_gams",
    version=__version__,
    description="An AutoML companion to fit GAM models easily",
    long_description=readme,
    author="Vitor Bezzan",
    author_email="vitor@bezzan.com",
    include_package_data=True,
    install_requires=requires,
)
