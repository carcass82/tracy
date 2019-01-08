cmake_minimum_required(VERSION 3.10)

project(Tracy LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Tracy_SOURCE_DIR}/bin/)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
	option(BUILD_WITH_CUDA "Build with CUDA support" ON)
else()
	option(BUILD_WITH_CUDA "Build with CUDA support" OFF)
endif()

option(USE_GLM "Use GLM instead of cclib" OFF)
option(CUDA_USE_STREAMS "Use async streams" OFF)
option(CUDA_USE_MULTIGPU "Use all available CUDA enabled GPUs" OFF)
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

include_directories(. ./src ./ext)

#
# Compiler options
#
if (MSVC)
    message(STATUS "Found MSVC: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "/nologo /W3 /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS_DEBUG "/EHsc /Zi /Od /D_DEBUG /DNOVTABLE")
    set(CMAKE_CXX_FLAGS_RELEASE "/EHsc /Zi /GR- /GL /O2 /Ob1 /Oi /Ot /fp:fast /openmp /DNDEBUG /DNOVTABLE=__declspec(novtable)")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /LTCG /INCREMENTAL:NO /OPT:REF")
else()
    message(STATUS "Found C++ Compiler: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "-Wall -no-pie -DNOVTABLE= ")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -fopenmp -march=native -flto -fno-rtti -funroll-loops -D_GLIBCXX_PARALLEL")
endif()

if (${CMAKE_BUILD_TYPE} MATCHES Release)
    message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
elseif (${CMAKE_BUILD_TYPE} MATCHES Debug)
    message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
else()
    message(STATUS "Current CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
endif()

add_executable(${PROJECT_NAME} ${SRC_MAIN} ${INC_LIST} ${INC_SHAPES} ${INC_SHAPES_MOD} ${INC_MATERIALS} ${INC_TEXTURES} ${INC_PDF} ${INC_CCLIB})

#
# CUDA related settings
#
if (BUILD_WITH_CUDA)
	enable_language(CUDA)
	add_definitions(-DUSE_CUDA)
	set(CMAKE_CUDA_FLAGS ${CUDA_COMPUTECAPABILITY})
	set(CMAKE_CUDA_FLAGS_DEBUG "-G -D_DEBUG")
	set(CMAKE_CUDA_FLAGS_RELEASE "--use_fast_math -O2 -lineinfo -DNDEBUG")

	if (CUDA_USE_STREAMS)
		add_definitions(-DCUDA_USE_STREAMS)
	endif()

	if (CUDA_USE_MULTIGPU)
		add_definitions(-DCUDA_USE_MULTIGPU)
	endif()
	
	file(GLOB SRC_CUDA "./src/cuda/*.cu" "./src/cuda/*.cuh")

	add_library(CUDA_CODE STATIC ${SRC_CUDA})
	target_link_libraries(${PROJECT_NAME} CUDA_CODE)
	
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

if (USE_GLM)
	add_definitions(-DUSE_GLM)
endif()
