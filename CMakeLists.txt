cmake_minimum_required(VERSION 2.8)

project(Tracy)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Tracy_SOURCE_DIR}/bin/)

#
# sources + headers (grouped)
#
file(GLOB INC_LIST "./src/*.hpp" "./src/*.h")
source_group("Includes" FILES ${INC_LIST})

file(GLOB INC_SHAPES "./src/shapes/*.hpp")
source_group("Includes\\Shapes" FILES ${INC_SHAPES})

file(GLOB INC_SHAPES_MOD "./src/shapes/modifiers/*.hpp")
source_group("Includes\\Shapes\\modifiers" FILES ${INC_SHAPES_MOD})

file(GLOB INC_MATERIALS "./src/materials/*.hpp")
source_group("Includes\\Materials" FILES ${INC_MATERIALS})

file(GLOB INC_TEXTURES "./src/textures/*.hpp")
source_group("Includes\\Textures" FILES ${INC_TEXTURES})

file(GLOB INC_CCLIB "./ext/cclib/*.h")
source_group("Ext\\CCLib" FILES ${INC_CCLIB})

file(GLOB SRC_MAIN "./src/*.cpp")

include_directories(. ./src)

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
add_executable(${PROJECT_NAME} ${SRC_MAIN} ${INC_LIST} ${INC_SHAPES} ${INC_SHAPES_MOD} ${INC_MATERIALS} ${INC_TEXTURES} ${INC_CCLIB})
