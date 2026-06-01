cmake_minimum_required(VERSION 3.16)
project(reloc_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io kdtree registration filters)

# The relocalizer under test lives in the ROS2 localizer package; compile it
# directly (icp_localizer.cpp) with no ROS2 dependency. commons.cpp is empty.
set(LOC_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../localizer/src")

add_executable(reloc_bench
  reloc_bench.cpp
  ${LOC_SRC_DIR}/localizers/icp_localizer.cpp)

target_include_directories(reloc_bench PRIVATE
  ${LOC_SRC_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS})

target_link_libraries(reloc_bench PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(reloc_bench PRIVATE ${PCL_DEFINITIONS})
