from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

extra_compile_args = ['-std=c++17']
if os.name == 'posix':
    extra_compile_args += ['-fopenmp']

setup(
    name='cpp_extensions',
    ext_modules=[
        CppExtension(
            'gaussian_mixture_cpp',
            ['gaussian_mixture.cpp', 'gaussian_mixture_bindings.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=['-fopenmp'] if os.name == 'posix' else []
        ),
        CppExtension(
            name="qem_cpp",
            sources=[
                "qem_binding.cpp",
                "qem_partitioning.cpp"
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=['-fopenmp'] if os.name == 'posix' else []
        ),
        CppExtension(
            name="voxel_partition_cpp",
            sources=[
                "voxel_partition.cpp"
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=['-fopenmp'] if os.name == 'posix' else []
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)