cmake_minimum_required(VERSION 3.15)
project(gtbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(gtbench SHARED
  src/stereo_postprocess.cpp
  src/metrics.cpp
)
target_include_directories(gtbench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(gtbench PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
