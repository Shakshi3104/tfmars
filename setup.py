import setuptools

setuptools.setup(
    name="tfmars",
    version="1.0.0",
    author="Shakshi3104",
    description="MarNASNet and famous CNN models for Sensor-based Human Activity Recognition",
    url="https://github.com/Shakshi3104/tfmars",
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires=">=3.6, <4",
    package_dir={'': 'src'},
)
