# Build dpretrieval: a pybind11 module wrapping DBoW2 for ORB-based image retrieval.
#
# DBoW2 is built as a static library and linked directly into the pybind11
# module, avoiding a separate shared library install step.
#
# DBoW2/ is cloned at build time by the recipe.yaml build script.

cmake_minimum_required(VERSION 3.14...3.28)
project(dpretrieval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── OpenCV ──────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core features2d highgui)

# ── pybind11 (provided by conda) ─────────────────────────────────────
find_package(pybind11 REQUIRED)

# ── Python (for install destination) ──────────────────────────────────
find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE SP_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

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
pybind11_add_module(dpretrieval src/main.cpp)
target_include_directories(dpretrieval PRIVATE DBoW2/include/DBoW2)
target_link_libraries(dpretrieval PRIVATE DBoW2 ${OpenCV_LIBS})

install(TARGETS dpretrieval LIBRARY DESTINATION ${SP_DIR})
