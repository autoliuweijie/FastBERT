# Load to Pypi

Load/Update the FastBERT to [Pypi](https://pypi.org/).

@reference: [official tutorials](https://packaging.python.org/tutorials/packaging-projects/)


## Step 1

Make sure the ``setup.py`` file is correct, and modify the ``version`` number if need.


## Step 2

Now run this command from the same directory where setup.py is located:

```sh
python3 setup.py sdist bdist_wheel
```

This command should output a lot of text and once completed should generate two files in the dist directory:

```
dist/
    fastbert-0.0.1-py3-none-any.whl
    fastbert-0.0.1.tar.gz
```


## Step 1

it’s time to upload the package to the Python Package Index. Run thie command:

```sh
python3 -m twine upload --repository pypi dist/*
```

Then, input your ``username`` and ``password`` of [Pypi](https://pypi.org/).

After the command completes, you should see [fastbert](https://pypi.org/project/fastbert/)

