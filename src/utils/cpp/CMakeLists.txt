cmake_minimum_required(VERSION 3.1)
project(gaussian_mixture_cpp)

# Set CMake policies
cmake_policy(SET CMP0057 NEW)

# Set C++17 as required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE Python3_SITEPACKAGES_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the path to PyTorch's CMake files
set(CMAKE_PREFIX_PATH "/home/moussabendjilali/miniforge3/envs/spt/lib/python3.8/site-packages/torch/share/cmake")

# Add conda environment path to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "/home/moussabendjilali/miniforge3/envs/spt/share/cmake")

# Find PyTorch
find_package(Torch REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Create a static library for the core GMM implementation
add_library(gmm_core STATIC gaussian_mixture.cpp)
target_link_libraries(gmm_core PUBLIC ${TORCH_LIBRARIES})
target_include_directories(gmm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET gmm_core PROPERTY POSITION_INDEPENDENT_CODE ON)

# Python module (using explicit name)
pybind11_add_module(gaussian_mixture_cpp gaussian_mixture_bindings.cpp)
target_link_libraries(gaussian_mixture_cpp PRIVATE gmm_core ${TORCH_LIBRARIES})
target_include_directories(gaussian_mixture_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Test executable
add_executable(test_gmm test_gmm.cpp)
target_link_libraries(test_gmm PRIVATE gmm_core ${TORCH_LIBRARIES})
target_include_directories(test_gmm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Print Python paths for debugging
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_SITEPACKAGES_DIR: ${Python3_SITEPACKAGES_DIR}")

# Install the Python module
install(TARGETS gaussian_mixture_cpp 
        LIBRARY DESTINATION ${Python3_SITEPACKAGES_DIR}
        RUNTIME DESTINATION ${Python3_SITEPACKAGES_DIR})