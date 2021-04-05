import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="neurodiffeq_conditions",
    version="0.1.0",
    author="Shuheng Liu",
    author_email="shuheng_liu@g.harvard.edu",
    description="Extra conditions for neurodiffeq",
    url="https://github.com/odegym/neurodiffeq-conditions",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    setup_requires=['setuptools_scm'],
    include_package_data=True,
)
