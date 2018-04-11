cmake_minimum_required(VERSION 2.8)

project(Tracy)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Tracy_SOURCE_DIR}/bin/)

include_directories(. ./src)

file(GLOB_RECURSE INC_LIST "./src/*.hpp" "./src/*.h" "./src/shapes/*.hpp" "./src/shapes/modifiers/*.hpp" "./src/materials/*.hpp" "./src/textures/*.hpp")
aux_source_directory(./src SRC_LIST)

if (MSVC)
    message(STATUS "Found MSVC: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "/nologo /W3 /D_CRT_SECURE_NO_WARNINGS /std:c++17")
    set(CMAKE_CXX_FLAGS_DEBUG "/EHsc /W3 /Zi /Od /D_DEBUG /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE "/EHsc- /GR- /GL /Gv /Ox /Ob1 /Oi /Ot /Oy /fp:fast /openmp /MT /DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /LTCG /INCREMENTAL:NO /OPT:REF")
else()
    message(STATUS "Found C++ Compiler: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "-Wall -no-pie -std=c++1z")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fopenmp -ffast-math -march=native -flto")
endif()

if (${CMAKE_BUILD_TYPE} MATCHES "Release")
    message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
elseif (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
else()
    message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
endif()
add_executable(${PROJECT_NAME} ${SRC_LIST} ${INC_LIST})
