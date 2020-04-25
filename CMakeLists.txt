cmake_minimum_required(VERSION 3.8)
project(Tracy LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/")

###############################################################################
# Defaults to Release Build                                                   #
###############################################################################
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()



###############################################################################
# CMP0072: FindOpenGL prefers GLVND by default when available.                #
#          (OLD policy forces legacy mode)                                    #
###############################################################################
if(POLICY CMP0072)
	cmake_policy(SET CMP0072 OLD)
endif()



###############################################################################
# Debug Options                                                               #
###############################################################################
option(DEBUG_VERBOSE_MAKEFILE "Debug CMake Build (Verbose Makefile)" OFF)
if(DEBUG_VERBOSE_MAKEFILE)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

option(DEBUG_SHOW_NORMALS "Draw scene normals only (no lighting)" OFF)
if(DEBUG_SHOW_NORMALS)
	add_definitions(-DDEBUG_SHOW_NORMALS)
endif()

option(USE_ACCEL_STRUCTURE "Use KDTree to speed up tracing" ON)
if(USE_ACCEL_STRUCTURE)
	add_definitions(-DUSE_KDTREE)
endif()



###############################################################################
# Math library selection                                                      #
###############################################################################
set(MATH_LIBRARY "cclib" CACHE STRING "Math Library")
set_property(CACHE MATH_LIBRARY PROPERTY STRINGS cclib glm)

if(MATH_LIBRARY STREQUAL "glm")
	add_definitions(-DUSE_GLM)
else()
	add_definitions(-DUSE_CCLIB)
	set(cclib_includes "./ext/cclib/cclib.h")
	source_group("ext\\cclib\\" FILES ${cclib_includes})
	list(APPEND tracy_includes ${cclib_includes})
endif()



###############################################################################
# Common Sources                                                              #
###############################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(. ./src ./ext)

set(ext_includes "./ext/stb_image.h"
				 "./ext/tiny_obj_loader.h")
source_group("ext\\" FILES ${ext_includes})
list(APPEND tracy_includes ${ext_includes})

set(main_sources "./src/win_raytracer.cpp"
                 "./src/mesh.cpp"
                 "./src/scene.cpp"
                 "./src/material.cpp")

source_group("src\\" FILES ${main_sources})

list(APPEND tracy_sources ${main_sources})


set(main_includes "./src/mesh.h"
                  "./src/scene.h"
                  "./src/material.h"
                  "./src/common.h"
                  "./src/ray.h"
                  "./src/camera.h"
                  "./src/timer.h"
                  "./src/random.h"
                  "./src/vertex.h"
				  "./src/aabb.h"
				  "./src/kdtree.h"
				  "./src/log.h"
				  "./src/input.h")

source_group("include\\" FILES ${main_includes})

list(APPEND tracy_includes ${main_includes})



###############################################################################
# Kernel selection                                                            #
###############################################################################
set(RENDER_MODULE "CPU (RayTracing)" CACHE STRING "Render Module")
set_property(CACHE RENDER_MODULE PROPERTY STRINGS "CPU (RayTracing)" "CUDA (RayTracing)" "OpenGL (Raster)")

if(RENDER_MODULE STREQUAL "CPU (RayTracing)")

	add_definitions(-DCPU_KERNEL=1)

	include_directories(./src/kernels/raytracing/software/)

	set(CPU_RAND_ALGORITHM "LCG" CACHE STRING "Random Algorithm to use")
	set_property(CACHE CPU_RAND_ALGORITHM PROPERTY STRINGS "rand()" "XorShift" "LCG")
	option(NO_OPENMP "Disable OpenMP in CPU Trace loop" OFF)
	
	set(USE_SIMD "Off" CACHE STRING "Use SIMD in triangle-ray test function")
	set_property(CACHE USE_SIMD PROPERTY STRINGS "Off" "SSE" "AVX")

	set(DATA_LAYOUT "SoA" CACHE STRING "Layout Triangle Data as SoA (Structure of Arrays) or AoS (Array of Structures)")
	set_property(CACHE DATA_LAYOUT PROPERTY STRINGS "SoA" "AoS")

	set(cpu_includes "./src/kernels/raytracing/software/cpu_trace.h")

	if (NOT USE_SIMD STREQUAL "Off")
		add_definitions(-DUSE_SIMD=1)
		list(APPEND cpu_includes "./src/kernels/raytracing/software/simdhelper.h")
		source_group("kernel\\cpu\\include" FILES ${cpu_includes})

		if (USE_SIMD STREQUAL "SSE")
			add_definitions(-DSIMD_SSE=1)
		elseif(USE_SIMD STREQUAL "AVX")
			add_definitions(-DSIMD_AVX=1)
		#TODO: Autodetect
		endif()
	endif()

	if (DATA_LAYOUT STREQUAL "AoS")
		add_definitions(-DAOS=1)
	endif()

	source_group("kernel\\cpu\\include" FILES ${cpu_includes})
	list(APPEND tracy_includes ${cpu_includes})

	set(cpu_sources "./src/kernels/raytracing/software/cpu_trace.cpp")
	source_group("kernel\\cpu\\src" FILES ${cpu_sources})
	list(APPEND tracy_sources ${cpu_sources})
	
	if(CPU_RAND_ALGORITHM STREQUAL "XorShift")
		add_definitions(-DRANDOM_XORSHIFT=1)
	elseif(CPU_RAND_ALGORITHM STREQUAL "LCG")
		add_definitions(-DRANDOM_LCG=1)
	endif()

	

elseif(RENDER_MODULE STREQUAL "CUDA (RayTracing)")

	add_definitions(-DCUDA_KERNEL=1)
	add_definitions(-DRANDOM_CUDA=1)

	find_package(OpenGL REQUIRED)
	list(APPEND tracy_libraries ${OPENGL_LIBRARIES})

	enable_language(C)
	add_definitions(-DGLEW_STATIC)
	include_directories(./ext/glew/include)
	set(glew_source "./ext/glew/glew.c")
	source_group("ext\\glew" FILES ${glew_source})
	list(APPEND tracy_sources ${glew_source})

	enable_language(CUDA)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_EXTENSIONS OFF)

	set(target_properties "CUDA_SEPARABLE_COMPILATION ON")
	list(APPEND target_properties "CUDA_RESOLVE_DEVICE_SYMBOLS ON")

	include_directories(./src/kernels/raytracing/cuda/)
	include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

	set(CUDA_ARCH Auto CACHE STRING "Cuda Arch Compile Option (Default: Autodetected)")
	
	# deprecated but currently the only way to detect cuda arch
	include(FindCUDA)
	cuda_select_nvcc_arch_flags(CUDA_COMPUTE_ARCH_FLAGS ${CUDA_ARCH})
	string (REPLACE ";" " " CUDA_COMPUTE_ARCH_FLAGS_STRING "${CUDA_COMPUTE_ARCH_FLAGS}")
	message(STATUS "CUDA architecture Compile Flags: ${CUDA_COMPUTE_ARCH_FLAGS_STRING}")
	
	set(CMAKE_CUDA_FLAGS "-DCUDA_CALL=\"__host__ __device__\" -DCUDA_DEVICE_CALL=__device__ -D_FORCE_INLINES ${CUDA_COMPUTE_ARCH_FLAGS_STRING}")
	set(CMAKE_CUDA_FLAGS_DEBUG "-G -D_DEBUG")
	set(CMAKE_CUDA_FLAGS_RELEASE "--use_fast_math -O2 -g -lineinfo -DNDEBUG")

	set(cuda_sources "./src/kernels/raytracing/cuda/cuda_trace.cpp"
	                 "./src/kernels/raytracing/cuda/cuda_trace.cu")
	source_group("kernel\\cuda\\src" FILES ${cuda_sources})
	list(APPEND tracy_sources ${cuda_sources})

	set(cuda_includes "./src/kernels/raytracing/cuda/cuda_trace.h"
	                  "./src/kernels/raytracing/cuda/cuda_vertex.h"
					  "./src/kernels/raytracing/cuda/cuda_material.h"
	                  "./src/kernels/raytracing/cuda/cuda_mesh.h"
					  "./src/kernels/raytracing/cuda/cuda_scene.h"
	                  "./src/kernels/raytracing/cuda/cuda_log.h")
	source_group("kernel\\cuda\\include" FILES ${cuda_includes})
	list(APPEND tracy_includes ${cuda_includes})

	set_source_files_properties("./src/material.cpp" PROPERTIES LANGUAGE CUDA)

elseif(RENDER_MODULE STREQUAL "OpenGL (Raster)")

	add_definitions(-DOPENGL_KERNEL=1)

	find_package(OpenGL REQUIRED)
	list(APPEND tracy_libraries ${OPENGL_LIBRARIES})

	enable_language(C)
	add_definitions(-DGLEW_STATIC)
	include_directories(./ext/glew/include)
	set(glew_source "./ext/glew/glew.c")
	source_group("ext\\glew" FILES ${glew_source})
	list(APPEND tracy_sources ${glew_source})
	
	include_directories(./src/kernels/rasterization/opengl/)

	set(opengl_sources "./src/kernels/rasterization/opengl/opengl_render.cpp")
	source_group("kernel\\opengl\\src" FILES ${opengl_sources})
	list(APPEND tracy_sources ${opengl_sources})

	set(opengl_includes "./src/kernels/rasterization/opengl/opengl_render.h"
	                    "./src/kernels/rasterization/opengl/gl_material.h"
	                    "./src/kernels/rasterization/opengl/gl_mesh.h"
	                    "./src/kernels/rasterization/opengl/materials.h")
	source_group("kernel\\opengl\\include" FILES ${opengl_includes})
	list(APPEND tracy_includes ${opengl_includes})

endif()



###############################################################################
# Compiler options                                                            #
###############################################################################
if(MSVC)

    message(STATUS "Found MSVC: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "/nologo /W3 /D_CRT_SECURE_NO_WARNINGS /MP /Gm- /EHsc /Zi")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /D_DEBUG /DNOVTABLE= ")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GR- /Gw /GL /O2 /fp:fast /Qpar /DNDEBUG /DNOVTABLE=__declspec(novtable)")
	
	# enable auto vectorizer report
	#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Qvec-report:2")

	# enable auto parallelizer report
	#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Qpar-report:2")

	if(NO_OPENMP)
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp-")
	else()
		# choose "experimental" openmp implementation if available
		# (and suppress warning C4849 about 'collapse' clause ignored)
		if(MSVC_VERSION GREATER_EQUAL 1920)
			set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp:experimental /wd4849")
		else()
			set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp /wd4849")
		endif()
	endif()
    
	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /LTCG /INCREMENTAL:NO")

else()

    message(STATUS "Found C++ Compiler: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "-Wall -no-pie -g -DNOVTABLE= ")
    set(CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Ofast -march=native -fsingle-precision-constant -fno-rtti -funroll-loops -D_GLIBCXX_PARALLEL")
	
	if(NOT NO_OPENMP)
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
	endif()
	
	if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
	endif()

endif()


###############################################################################
# Build Rules                                                                 #
###############################################################################
if(WIN32)
	set(WIN32_MAIN WIN32)
elseif(UNIX)
	find_package(X11 REQUIRED)
	include_directories(${X11_INCLUDE_DIR})
	list(APPEND tracy_libraries ${X11_LIBRARIES})
endif()

add_executable(${PROJECT_NAME} ${WIN32_MAIN} ${tracy_sources} ${tracy_includes})
target_link_libraries(${PROJECT_NAME} ${tracy_libraries})

foreach(target_property IN LISTS target_properties)
    
	string(REPLACE " " ";" property_pair ${target_property})
	list(GET property_pair 0 property)
	list(GET property_pair 1 value)
	
	message(STATUS "set_target_properties: ${property} = ${value}")
	set_target_properties(${PROJECT_NAME} PROPERTIES ${property} ${value})
endforeach()
