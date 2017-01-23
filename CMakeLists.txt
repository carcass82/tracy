project(Tracy)

cmake_minimum_required(VERSION 2.8)

include_directories(.)
include_directories(./src)

if (WIN32)
include_directories(../SharedLibs)
endif()

file(GLOB_RECURSE INC_LIST "./src/*.hpp" "./src/*.h")
aux_source_directory(./src SRC_LIST)

if (MSVC)
	message(STATUS "Found MSVC: " ${CMAKE_CXX_COMPILER})
	set(CMAKE_CXX_FLAGS_DEBUG "/nologo /EHsc /W3 /Zi /Od /D_DEBUG /MDd")
	set(CMAKE_CXX_FLAGS_RELEASE "/nologo /EHsc- /Ox /fp:fast /openmp /W3 /DNDEBUG")
else()
	message(STATUS "Found C++ Compiler: " ${CMAKE_CXX_COMPILER})
	set(CMAKE_CXX_FLAGS_DEBUG "-Wall -O0 -g -fopenmp -std=c++11")
	set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -fopenmp -ffast-math -march=native -std=c++11 ")
endif()

if (${CMAKE_BUILD_TYPE} MATCHES "Release")
	message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS_RELEASE})
elseif (${CMAKE_BUILD_TYPE} MATCHES "Debug")
	message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS_DEBUG})
else()
	message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
endif()
add_executable(${PROJECT_NAME} ${SRC_LIST} ${INC_LIST})