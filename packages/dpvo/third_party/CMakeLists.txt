# Build dpretrieval: a pybind11 module wrapping DBoW2 for ORB-based image retrieval.
#
# This builds DBoW2 as a static library and links it directly into the
# pybind11 module, avoiding a separate install step.
#
# Usage (from packages/dpvo/):
#   mkdir -p third_party/build && cd third_party/build
#   cmake .. -DPYTHON_EXECUTABLE=$(which python)
#   make -j$(nproc)
#   # produces dpretrieval*.so in third_party/build/

cmake_minimum_required(VERSION 3.14...3.28)
project(dpretrieval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── OpenCV ──────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core features2d highgui)

# ── pybind11 ────────────────────────────────────────────────────────────
# Try system/conda pybind11 first, fall back to fetching it
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# ── DBoW2 (static library) ─────────────────────────────────────────────
set(DBOW2_SRCS
  DBoW2/src/BowVector.cpp
  DBoW2/src/FBrief.cpp
  DBoW2/src/FORB.cpp
  DBoW2/src/FSurf64.cpp
  DBoW2/src/FeatureVector.cpp
  DBoW2/src/QueryResults.cpp
  DBoW2/src/ScoringObject.cpp
)

add_library(DBoW2 STATIC ${DBOW2_SRCS})
target_include_directories(DBoW2 PUBLIC DBoW2/include DBoW2/include/DBoW2)
target_link_libraries(DBoW2 PUBLIC ${OpenCV_LIBS})

# ── dpretrieval pybind11 module ─────────────────────────────────────────
pybind11_add_module(dpretrieval dpretrieval/src/main.cpp)
target_include_directories(dpretrieval PRIVATE DBoW2/include/DBoW2)
target_link_libraries(dpretrieval PRIVATE DBoW2 ${OpenCV_LIBS})
