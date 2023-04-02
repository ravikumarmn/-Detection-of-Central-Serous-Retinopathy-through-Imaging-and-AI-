# put this in the setup.py file.


from setuptools import setup,find_packages

setup(
    name="csr", # name of the package
    packages=find_packages()
)

# run pip install -e . 