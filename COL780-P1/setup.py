import sys
import os
import subprocess
from setuptools import setup, Extension
import pybind11

def get_mac_libomp_paths():
    """
    Attempts to find libomp prefix on macOS using Homebrew.
    Returns (include_path, lib_path) or (None, None).
    """
    # 1. Check environment variable first
    if 'LIBOMP_DIR' in os.environ:
        base = os.environ['LIBOMP_DIR']
        return os.path.join(base, 'include'), os.path.join(base, 'lib')

    # 2. Try Homebrew
    try:
        prefix = subprocess.check_output(['brew', '--prefix', 'libomp'], 
                                       stderr=subprocess.STDOUT).decode().strip()
        return os.path.join(prefix, 'include'), os.path.join(prefix, 'lib')
    except (OSError, subprocess.CalledProcessError):
        return None, None

# --- Compiler Configuration ---

include_dirs = [pybind11.get_include()]
library_dirs = []
extra_compile_args = []
extra_link_args = []
define_macros = []

if sys.platform == 'win32':
    # Windows (MSVC)
    extra_compile_args += ['/O2', '/openmp', '/std:c++14']
    # MSVC doesn't need extra_link_args for OpenMP usually, implied by /openmp
else:
    # Linux / macOS (GCC / Clang)
    extra_compile_args += ['-O3', '-std=c++14']
    
    if sys.platform == 'darwin':
        # macOS specific OpenMP configuration
        extra_compile_args += ['-Xpreprocessor', '-fopenmp']
        extra_link_args += ['-lomp']
        
        # Find libomp location
        inc_path, lib_path = get_mac_libomp_paths()
        if inc_path and lib_path:
            include_dirs.append(inc_path)
            library_dirs.append(lib_path)
            # Rpath ensures the library is found at runtime without setting DYLD_LIBRARY_PATH
            extra_link_args += [f'-Wl,-rpath,{lib_path}']
        else:
            print("[WARNING] Could not find libomp via Homebrew or LIBOMP_DIR.")
            print("          Compilation may fail if <omp.h> is not in standard paths.")
            
    else:
        # Linux (GCC)
        extra_compile_args += ['-fopenmp']
        extra_link_args += ['-fopenmp']

# --- Module Definition ---

ext_modules = [
    Extension(
        'custom_cv2_cpp',               
        sources=['cpp/custom_cv2.cpp'],     
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
        language='c++'
    )
]

setup(
    name='custom_cv2_cpp',
    version='1.0.0',
    author='User',
    description='High-performance C++ implementation of homography and CV utils using OpenMP',
    ext_modules=ext_modules,
    zip_safe=False,
)