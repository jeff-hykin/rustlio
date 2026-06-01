cmake_minimum_required(VERSION 3.16)
project(fastlio_cpp_ref CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io kdtree search filters)
find_package(fmt REQUIRED)   # newer Sophus uses fmt

# The in-repo C++ FAST-LIO algorithm (no ROS2 dependency in map_builder/).
set(MB "${CMAKE_CURRENT_SOURCE_DIR}/../../fastlio2/src/map_builder")

add_executable(ref_run
  ref_run.cpp
  ${MB}/map_builder.cpp
  ${MB}/ieskf.cpp
  ${MB}/imu_processor.cpp
  ${MB}/lidar_processor.cpp
  ${MB}/commons.cpp
  ${MB}/ikd_Tree.cpp)

target_include_directories(ref_run PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../fastlio2/src
  ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS}
  /opt/homebrew/include)               # Sophus (header-only)
target_link_libraries(ref_run PRIVATE ${PCL_LIBRARIES} fmt::fmt)
target_compile_definitions(ref_run PRIVATE ${PCL_DEFINITIONS})
