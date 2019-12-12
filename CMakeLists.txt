cmake_minimum_required(VERSION 3.8)
include(CMakeDependentOption)

project(Tracy LANGUAGES CXX)

#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Tracy_SOURCE_DIR}/bin/)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DEBUG_SHOW_NORMALS "Draw scene normals only (no lighting)" OFF)
if(DEBUG_SHOW_NORMALS)
	add_definitions(-DDEBUG_SHOW_NORMALS)
endif()

#
# math library selection
#
set(MATH_LIBRARY "cclib" CACHE STRING "Math Lib")
set_property(CACHE MATH_LIBRARY PROPERTY STRINGS cclib glm)
if(MATH_LIBRARY STREQUAL "glm")
	add_definitions(-DUSE_GLM)
else()
	file(GLOB INC_CCLIB "./ext/cclib/*.h")
	source_group("Ext\\CCLib" FILES ${INC_CCLIB})
endif()

#
# kernel selection
#
set(RENDER_MODULE "CPU (RayTracing)" CACHE STRING "Render Module")
set_property(CACHE RENDER_MODULE PROPERTY STRINGS "CPU (RayTracing)" "CUDA (RayTracing)" "OpenGL (Raster)")

#
# default to Release build
#
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

#
# default sources + headers (grouped)
#
file(GLOB INC_LIST "./src/*.h")
source_group("Includes" FILES ${INC_LIST})

file(GLOB INC_EXT "./ext/*.h")
source_group("Ext\\" FILES ${INC_EXT})

include_directories(. ./src ./ext)
file(GLOB SRC_GUI "./src/*.cpp" "./src/*.h")

#
# Compiler options
#
if(MSVC)
    message(STATUS "Found MSVC: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "/nologo /W3 /D_CRT_SECURE_NO_WARNINGS /MP")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc /Zi /Od /D_DEBUG /DNOVTABLE= ")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHsc /Zi /GR- /GL /Gs- /O2 /fp:fast /DNDEBUG /DNOVTABLE=__declspec(novtable)")
	
	if(MSVC_VERSION GREATER_EQUAL 1920)
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp:experimental")
		#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Qvec-report:2")
	else()
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp")
	endif()
    
	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /LTCG /INCREMENTAL:NO /OPT:REF")
else()
    message(STATUS "Found C++ Compiler: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "-Wall -no-pie -DNOVTABLE= ")
    set(CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG -O0 -g")
	set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Ofast -march=native -fsingle-precision-constant -fopenmp -fno-rtti -funroll-loops -D_GLIBCXX_PARALLEL")
	if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
	endif()
endif()

#
# kernels
#
if(RENDER_MODULE STREQUAL "CPU (RayTracing)")

	add_definitions(-DCPU_KERNEL=1)

	include_directories(./src/kernels/raytracing/software/)

	file(GLOB SRC_MAIN "./src/kernels/raytracing/software/*.cpp" "./src/kernels/raytracing/software/*.h")
	source_group("Module\\CPU" FILES ${SRC_MAIN})
	
	set(CPU_RAND_ALGORITHM "Default" CACHE STRING "Random Algorithm to use")
	set_property(CACHE CPU_RAND_ALGORITHM PROPERTY STRINGS "Default" "XorShift" "PCG" "Intel")
	
	if(CPU_RAND_ALGORITHM STREQUAL "XorShift")
		add_definitions(-DRANDOM_XORSHIFT=1)
	elseif(CPU_RAND_ALGORITHM STREQUAL "PCG")
		add_definitions(-DRANDOM_PCG=1)
	elseif(CPU_RAND_ALGORITHM STREQUAL "Intel")
		add_definitions(-DRANDOM_INTEL=1)
	endif()

elseif(RENDER_MODULE STREQUAL "CUDA (RayTracing)")

	find_package(OpenGL REQUIRED)

	add_definitions(-DGLEW_STATIC)
	enable_language(C) # for glew.c
	include_directories(./ext/glew/include)

elseif(RENDER_MODULE STREQUAL "OpenGL (Raster)")

	find_package(OpenGL REQUIRED)
	
	add_definitions(-DOPENGL_KERNEL=1)
	
	add_definitions(-DGLEW_STATIC)
	enable_language(C) # for glew.c
	
	include_directories(./src/kernels/rasterization/opengl/)
	include_directories(./ext/glew/include)

	file(GLOB SRC_MAIN "./src/kernels/rasterization/opengl/*.cpp" "./src/kernels/rasterization/opengl/*.h" "./ext/glew/glew.c")
	source_group("Module\\OpenGL" FILES ${SRC_MAIN})

endif()

#
# show build flags
#
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	message(STATUS "Debug CMAKE_CXX_FLAGS:  ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
	message(STATUS "Debug CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
	message(STATUS "Release CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
	message(STATUS "Release CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
endif()

#
# main app build rule
#
if(WIN32)
 add_executable(${PROJECT_NAME} WIN32 ${SRC_MAIN} ${SRC_GUI} ${INC_LIST} ${INC_CCLIB} ${INC_EXT})
 target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES})
elseif(UNIX)
 find_package(X11 REQUIRED)
 include_directories(${X11_INCLUDE_DIR})
 add_executable(${PROJECT_NAME} ${SRC_MAIN} ${SRC_GUI} ${INC_LIST} ${INC_CCLIB} ${INC_EXT})
 target_link_libraries(${PROJECT_NAME} ${X11_LIBRARIES} ${OPENGL_LIBRARIES})
endif()

#
# CUDA must be defined here as it needs to be appended to existing target executable
#
if(RENDER_MODULE STREQUAL "CUDA (RayTracing)")

	enable_language(CUDA)
	
	add_definitions(-DCUDA_KERNEL=1)

	set(CUDA_COMPUTECAPABILITY "-arch=sm_30" CACHE STRING "Cuda Arch Compile Option")

	set(CMAKE_CUDA_FLAGS "-std=c++14 -D_FORCE_INLINES ${CUDA_COMPUTECAPABILITY}")
	set(CMAKE_CUDA_FLAGS_DEBUG "-G -D_DEBUG")
	set(CMAKE_CUDA_FLAGS_RELEASE "--use_fast_math -O2 -g -lineinfo -DNDEBUG")

	include_directories(../src/kernels/raytracing/cuda/)

	file(GLOB SRC_CUDA "./src/kernels/raytracing/cuda/*.cpp" "./src/kernels/raytracing/cuda/*.h" "./src/kernels/raytracing/cuda/*.cu" "./src/kernels/raytracing/cuda/*.cuh" "./ext/glew/glew.c")
	source_group("CUDA\\" FILES ${SRC_CUDA})

	add_library(CUDA_CODE STATIC ${SRC_CUDA})
	target_link_libraries(${PROJECT_NAME} CUDA_CODE)

	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()
