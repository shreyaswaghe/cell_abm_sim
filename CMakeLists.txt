option(USE_LINALG_BLAS "Use BLAS for linear algebra operations" OFF)

cmake_minimum_required(VERSION 3.20)
project(numeric LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type only for main project
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -larmpl -march=native -Wall -Rpass-missed=loop-vectorize")

include(GNUInstallDirs)

add_executable(OnePhenoCellSim ${CMAKE_SOURCE_DIR}/src/OnePhenoCellSim.cpp)

add_subdirectory(third_party/swnumeric)
target_link_libraries(OnePhenoCellSim PRIVATE swnumeric)
target_include_directories(OnePhenoCellSim PRIVATE third_party/swnumeric)
