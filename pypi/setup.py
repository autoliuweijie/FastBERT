import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages()
print(packages)

package_data = {'fastbert.files': ['fastbert/files/*.json', 'fastbert/files/*.txt']}

setuptools.setup(
    name="fastbert", # Replace with your own username
    version="0.0.9",
    author="Weijie Liu",
    author_email="autoliuweijie@163.com",
    description="The pipy version of FastBERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autoliuweijie/FastBERT",
    packages=packages,
    package_data=package_data,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
    install_requires=[
        'torch>=1.0.0',
        ]
)
