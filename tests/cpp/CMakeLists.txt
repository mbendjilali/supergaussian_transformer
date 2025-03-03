cmake_minimum_required(VERSION 3.1)
project(gaussian_mixture_tests)

# Set C++17 as required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Link to main project
add_subdirectory(../../src/utils/cpp gmm_core_build)

# Add test executable
add_executable(test_gmm test_gmm.cpp)
target_link_libraries(test_gmm PRIVATE gmm_core ${TORCH_LIBRARIES})
target_include_directories(test_gmm 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/utils/cpp
)

# Enable testing
enable_testing()
add_test(NAME test_gmm COMMAND test_gmm) 