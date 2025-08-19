set (CMAKE_CXX_STANDARD 20)
cmake_minimum_required(VERSION 3.16)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#=========================#
# build examples          #
#=========================#

project(ba_lie_example)

find_package(OpenMP)
find_package(Sophus REQUIRED)

set(HOLIDAY_OPT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE  ba_lie_example_src 
    ${CMAKE_CURRENT_LIST_DIR}/ba_lie_testy.cpp
)
add_executable(ba_lie_example 
    ${ba_lie_example_src}
)
target_include_directories(ba_lie_example PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${HOLIDAY_OPT_INCLUDE_DIR}  # holiday_opt のインクルードディレクトリを追加
)

target_link_libraries(ba_lie_example PUBLIC 
    Eigen3::Eigen
    Sophus::Sophus
    OpenMP::OpenMP_CXX
    holiday_opt
)
