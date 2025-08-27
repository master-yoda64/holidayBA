set (CMAKE_CXX_STANDARD 20)
cmake_minimum_required(VERSION 3.16)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)


#=========================#
# find dependencies      #
#=========================#
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

#=========================#
# set compile options     #
#=========================#
if(CPACK_DEBIAN_PACKAGE_ARCHITECTURE STREQUAL "arm64")
  add_compile_options(-Wno-error=attributes -Wno-strict-aliasing)
endif()

add_compile_options( -Wno-array-bounds )

#=========================#
# Add executable          #
#=========================#
file(GLOB_RECURSE  sources 
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_library(holiday_opt STATIC
    ${sources}
)

#=========================#
# set include directories #
#=========================#
target_include_directories(holiday_opt PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)
#=========================#
# link libraries          #
#=========================#
target_link_libraries(holiday_opt PUBLIC 
    ${OpenCV_LIBS}
)

add_subdirectory(example/ba_lie)
add_subdirectory(example/ba_ceres)
