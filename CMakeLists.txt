cmake_minimum_required(VERSION 3.10)

project(Tracy LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Tracy_SOURCE_DIR}/bin/)

option(BUILD_WITH_CUDA "Build CUDA support." ON)
set(CUDA_COMPUTECAPABILITY "-arch=sm_30" CACHE STRING "Cuda Arch Compile Option")

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

file(GLOB INC_PDF "./src/pdf/*.hpp")
source_group("Includes\\PDF" FILES ${INC_PDF})

file(GLOB INC_CCLIB "./ext/cclib/*.h")
source_group("Ext\\CCLib" FILES ${INC_CCLIB})

file(GLOB SRC_MAIN "./src/*.cpp")

include_directories(. ./src)

if (MSVC)
    message(STATUS "Found MSVC: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "/nologo /W3 /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} "/W3 /EHsc /Zi /Od /D_DEBUG /D_ALLOW_RTCc_IN_STL")
    set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "/EHsc- /GR- /Gv /GL /Ox /Ob1 /Oi /Ot /Oy /fp:fast /openmp /DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /LTCG /INCREMENTAL:NO /OPT:REF")
else()
    message(STATUS "Found C++ Compiler: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "-Wall -no-pie")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fopenmp -ffast-math -march=native -flto")
endif()

if (${CMAKE_BUILD_TYPE} MATCHES Release)
    message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
elseif (${CMAKE_BUILD_TYPE} MATCHES Debug)
    message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
else()
    message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
endif()

add_executable(${PROJECT_NAME} ${SRC_MAIN} ${INC_LIST} ${INC_SHAPES} ${INC_SHAPES_MOD} ${INC_MATERIALS} ${INC_TEXTURES} ${INC_PDF} ${INC_CCLIB})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

if(BUILD_WITH_CUDA)
	add_definitions(-DUSE_CUDA)
	set(CMAKE_CUDA_FLAGS ${CUDA_COMPUTECAPABILITY})
	
	file(GLOB SRC_CUDA "./src/cuda/*.cu" "./src/cuda/*.cuh")

	add_library(CUDA_CODE STATIC ${SRC_CUDA})
	target_link_libraries(${PROJECT_NAME} CUDA_CODE)
	
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_PROPAGATE_HOST_FLAGS OFF)
endif(BUILD_WITH_CUDA)
