from setuptools import setup, find_packages

setup(
    name='PhysDock',
    packages=find_packages(),
    include_package_data=True,
    version='0.0.1',
    description='Physics-Guided All-Atom Diffusion Model for Accurate Protein-Ligand Complex Prediction',
    author='Kexin Zhang',
    author_email='zhangkx2022@shanghaitech.edu.cn',
    license='MIT',
    keywords=['Molecular Docking'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
)
