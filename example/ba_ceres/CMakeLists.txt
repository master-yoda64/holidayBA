set (CMAKE_CXX_STANDARD 20)
cmake_minimum_required(VERSION 3.16)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#=========================#
# build examples          #
#=========================#

project(ba_ceres_example)

find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)

set(HOLIDAY_OPT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE  ba_ceres_example_src 
    ${CMAKE_CURRENT_LIST_DIR}/ba_ceres.cpp
)
add_executable(ba_ceres_example 
    ${ba_ceres_example_src}
)
target_include_directories(ba_ceres_example PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${HOLIDAY_OPT_INCLUDE_DIR}
)

target_link_libraries(ba_ceres_example PUBLIC 
    Eigen3::Eigen
    holiday_opt
    Ceres::ceres
)
