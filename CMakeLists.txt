cmake_minimum_required(VERSION 3.8)
include(CMakeDependentOption)

project(Tracy LANGUAGES CXX)

#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Tracy_SOURCE_DIR}/bin/)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
check_language(CUDA)

option(BUILD_WITH_CUDA "Build with CUDA support" ${CMAKE_CUDA_COMPILER})
option(CUDA_USE_STREAMS "Use async streams" OFF)
option(CUDA_USE_MULTIGPU "Use all available CUDA enabled GPUs" OFF)
set(CUDA_COMPUTECAPABILITY "-arch=sm_30" CACHE STRING "Cuda Arch Compile Option")

option(USE_CPU_RECURSIVE "Select between recursive and iterative trace() implementation for CPU" OFF)
option(USE_GLM "Use GLM instead of cclib" OFF)
option(DEBUG_BVH "BVH Debug Visualization" OFF)

option(USE_GUI "Build GUI Application" OFF)
option(USE_OPENGL "Use OpenGL in GUI Application" OFF)

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

file(GLOB INC_EXT "./ext/*.h")
source_group("Ext\\" FILES ${INC_EXT})

file(GLOB SRC_MAIN "./src/raytracer.cpp")

if (USE_GUI)
	if (UNIX)
		file(GLOB SRC_GUI "./src/lnx_raytracer.cpp")
	elseif(WIN32)
		file(GLOB SRC_GUI "./src/win_raytracer.cpp")
	endif()
endif()

include_directories(. ./src ./ext)

if (USE_GUI)
	if (UNIX)
		find_package(X11 REQUIRED)
		include_directories(${X11_INCLUDE_DIR})
	endif()

	if (USE_OPENGL)
		find_package(OpenGL REQUIRED)
	endif()
endif()

#
# Compiler options
#
if (MSVC)
    message(STATUS "Found MSVC: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "/nologo /W3 /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc /Zi /Od /D_DEBUG /DNOVTABLE= ")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /GR- /GL /Gs- /O2 /fp:fast /DNDEBUG /DNOVTABLE=__declspec(novtable)")
	
	if(MSVC_VERSION GREATER_EQUAL 1920)
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp:experimental")
	else()
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp")
	endif()
    
	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /LTCG /INCREMENTAL:NO /OPT:REF")
else()
    message(STATUS "Found C++ Compiler: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "-Wall -no-pie -DNOVTABLE= ")
    set(CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG -O0 -g")
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Ofast -march=native -fsingle-precision-constant -fopenmp -fno-rtti -funroll-loops -D_GLIBCXX_PARALLEL")
	else()
		set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Ofast -march=native -fsingle-precision-constant -fopenmp -flto -fno-rtti -funroll-loops -D_GLIBCXX_PARALLEL")
	endif()
endif()

message(STATUS "Debug CMAKE_CXX_FLAGS:  ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Release CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Debug CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
message(STATUS "Release CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")

if (USE_GUI AND WIN32)
 add_executable(${PROJECT_NAME} WIN32 ${SRC_MAIN} ${SRC_GUI} ${INC_LIST} ${INC_SHAPES} ${INC_SHAPES_MOD} ${INC_MATERIALS} ${INC_TEXTURES} ${INC_PDF} ${INC_CCLIB} ${INC_EXT})
 target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES})
elseif (USE_GUI AND UNIX)
 add_executable(${PROJECT_NAME} ${SRC_MAIN} ${SRC_GUI} ${INC_LIST} ${INC_SHAPES} ${INC_SHAPES_MOD} ${INC_MATERIALS} ${INC_TEXTURES} ${INC_PDF} ${INC_CCLIB} ${INC_EXT})
 target_link_libraries(${PROJECT_NAME} ${X11_LIBRARIES} ${OPENGL_LIBRARIES})
else()
 add_executable(${PROJECT_NAME} ${SRC_MAIN} ${INC_LIST} ${INC_SHAPES} ${INC_SHAPES_MOD} ${INC_MATERIALS} ${INC_TEXTURES} ${INC_PDF} ${INC_CCLIB} ${INC_EXT})
 if (UNIX)
  target_link_libraries(${PROJECT_NAME} pthread)
 endif()
endif()

#
# CUDA related settings
#
if (BUILD_WITH_CUDA)
	enable_language(CUDA)
	add_definitions(-DUSE_CUDA)
	set(CMAKE_CUDA_FLAGS "-std=c++11 -D_FORCE_INLINES ${CUDA_COMPUTECAPABILITY}")
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

if (USE_CPU_RECURSIVE)
	add_definitions(-DCPU_RECURSIVE)
endif()

if (USE_GLM)
	add_definitions(-DUSE_GLM)
endif()

if (DEBUG_BVH)
	add_definitions(-DDEBUG_BVH)
endif()

if (USE_GUI)
	add_definitions(-DBUILD_GUI)
endif()

if (USE_OPENGL)
	add_definitions(-DUSE_OPENGL)
endif()
