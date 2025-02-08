from setuptools import setup, find_packages

setup(
    name="Ampire",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "pytest",
        # سایر وابستگی‌ها...
    ],
    include_package_data=True,
)
