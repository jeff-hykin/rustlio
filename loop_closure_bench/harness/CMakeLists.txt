cmake_minimum_required(VERSION 3.16)
project(pgo_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3)

# GTSAM: under `nix develop` it is on CMAKE_PREFIX_PATH already. Otherwise fall
# back to a locally-built install if one exists (legacy brew path). Override
# with -DGTSAM_DIR=...
if(NOT GTSAM_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.deps/gtsam-install/lib/cmake/GTSAM")
  set(GTSAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.deps/gtsam-install/lib/cmake/GTSAM")
endif()
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io kdtree registration filters)

# The C++ PGO under test lives in the ROS2 package; we compile it directly
# (commons.cpp + simple_pgo.cpp) with no ROS2 dependency.
set(PGO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../pgo/src")

add_executable(pgo_bench
  pgo_bench.cpp
  ivan_pgo.cpp
  ${PGO_SRC_DIR}/pgos/commons.cpp
  ${PGO_SRC_DIR}/pgos/simple_pgo.cpp)

target_include_directories(pgo_bench PRIVATE
  ${PGO_SRC_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS})

target_link_libraries(pgo_bench PRIVATE gtsam ${PCL_LIBRARIES})
target_compile_definitions(pgo_bench PRIVATE ${PCL_DEFINITIONS})
