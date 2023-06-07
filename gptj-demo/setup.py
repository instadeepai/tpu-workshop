import setuptools

setuptools.setup(
    name="src",
    version="0.0.1",
    author="Thomas D Barrett",
    author_email="t.barrett@instadeep.com",
    description="Google TPU Workshop",
    url="https://github.com/instadeep/tpu-workshop",
    packages=["src"],
    package_dir={'src':'src'},
    python_requires='>=3.8'
)
