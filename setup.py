from setuptools import setup, find_packages


def requirements():
    with open('requirements.txt') as f:
        return f.read().strip().split('\n')

def version():
    with open('VERSION') as f:
        return f.read().strip()

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='scikit-info',
    version=version(),
    license='MIT',
    description='Scipy-styled expansion for information theory.',
    long_description=readme(),
    long_description_content_type = 'text/markdown',
    author='Mirko Mälicke',
    author_email='mirko.maelicke@kit.edu',
    install_requires=requirements(),
    test_require=['nose'],
    test_suite='nose.collector',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False
)
