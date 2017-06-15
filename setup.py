from setuptools import setup

setup(name="xgbutils",
      description="Utilities for xgboost",
      author="Andrew Hah",
      author_email="hahdawg@yahoo.com",
      license="MIT",
      packages=["xgbutils"],
      install_requires=[
        "hyperopt",
        "scikit-learn",
        "xgboost",
      ],
      zip_safe=False)
