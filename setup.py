import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='func_conn_predict',
    version='0.0.3',
    author='Ben Griffin',
    author_email='ben.griffin@keble.ox.ac.uk',
    description='Prediction of netmats',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bengriffin3/func_conn_predict',
    project_urls = {
        "Bug Tracker": "https://github.com/bengriffin3/func_conn_predict/issues"
    },
    license='MIT',
    packages=['func_conn_predict'],
    install_requires=['requests'],
)
