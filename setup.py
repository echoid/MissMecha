from setuptools import setup, find_packages

setup(
    name='missmecha-py',
    version='0.1.0',
    description='All about missing data mechanisms: simulation, analysis, and visualization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Youran Zhou',
    author_email='youranzhou0724@gmail.com',
    url='https://github.com/echoid/MissMecha',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'scipy',
        'seaborn'
    ],
    include_package_data=True,
    zip_safe=False,
)
