from setuptools import setup, Extension
import sys
import subprocess
import os

try:
    import pybind11
except Exception:
    raise SystemExit("pybind11 is required: run 'pip install pybind11'")

include_dirs = [pybind11.get_include()]
library_dirs = []
extra_compile_args = ['-std=c++14', '-O3']
extra_link_args = []

if sys.platform == 'darwin':
    # macOS uses clang; OpenMP support requires libomp from Homebrew.
    # Use -Xpreprocessor -fopenmp when compiling and link with -lomp.
    extra_compile_args += ['-Xpreprocessor', '-fopenmp']
    extra_link_args += ['-lomp']
    # Try to find Homebrew libomp prefix
    try:
        brew_prefix = subprocess.check_output(['brew', '--prefix', 'libomp']).decode().strip()
        include_dirs.append(os.path.join(brew_prefix, 'include'))
        library_dirs.append(os.path.join(brew_prefix, 'lib'))
    except Exception:
        # Allow users to set LIBOMP_DIR env var if brew isn't available
        libomp_dir = os.environ.get('LIBOMP_DIR')
        if libomp_dir:
            include_dirs.append(os.path.join(libomp_dir, 'include'))
            library_dirs.append(os.path.join(libomp_dir, 'lib'))

else:
    # Linux/others: use -fopenmp
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

ext_modules = [
    Extension(
        'homography_cpp',
        sources=['homography.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    )
]

setup(
    name='homography_cpp',
    version='0.0.1',
    author='auto',
    description='Pybind11 wrapper for homography',
    ext_modules=ext_modules,
)
