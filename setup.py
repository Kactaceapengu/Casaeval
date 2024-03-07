from setuptools import setup, find_packages

setup(
    name='casaval',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'casaval = main.__main__:main',
            'casaval-convert = main.convert:main',
        ]
    },
    install_requires=[
        'click',  # Add Click as a dependency
    ],
)