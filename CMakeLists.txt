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
# Debug Options                                                               #
###############################################################################
option(DEBUG_VERBOSE_MAKEFILE "Debug CMake Build (Verbose Makefile)" OFF)
if(DEBUG_VERBOSE_MAKEFILE)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

set(DEBUG_VIEW "None" CACHE STRING "Debug Visualization")
set_property(CACHE DEBUG_VIEW PROPERTY STRINGS "None" "BaseColor" "Normals" "Metalness" "Roughness" "Emissive")
if(DEBUG_VIEW STREQUAL "BaseColor")
	add_definitions(-DDEBUG_SHOW_BASECOLOR)
elseif(DEBUG_VIEW STREQUAL "Normals")
	add_definitions(-DDEBUG_SHOW_NORMALS)
elseif(DEBUG_VIEW STREQUAL "Metalness")
	add_definitions(-DDEBUG_SHOW_METALNESS)
elseif(DEBUG_VIEW STREQUAL "Roughness")
	add_definitions(-DDEBUG_SHOW_ROUGHNESS)
elseif(DEBUG_VIEW STREQUAL "Emissive")
	add_definitions(-DDEBUG_SHOW_EMISSIVE)
endif()

set(TRACY_ACCELERATION_STRUCTURE "Kd-Tree" CACHE STRING "Use Acceleration Structure")
set_property(CACHE TRACY_ACCELERATION_STRUCTURE PROPERTY STRINGS "None" "Kd-Tree" "BVH")
if(TRACY_ACCELERATION_STRUCTURE STREQUAL "Kd-Tree")
	add_definitions(-DUSE_KDTREE)
elseif(TRACY_ACCELERATION_STRUCTURE STREQUAL "BVH")
	add_definitions(-DUSE_BVH)
endif()

option(DEBUG_KDTREE_DISABLE_SAH "Disable SAH (Surface Area Heuristic) when building Kd-Tree" ON)
if(TRACY_ACCELERATION_STRUCTURE STREQUAL "Kd-Tree" AND DEBUG_KDTREE_DISABLE_SAH)
	add_definitions(-DDISABLE_SAH)
endif()

option(DEBUG_ACCELERATION_STRUCTURE_SHOW "Draw Acceleration Structure only (no geometry)" OFF)
if(NOT TRACY_ACCELERATION_STRUCTURE STREQUAL "None" AND DEBUG_ACCELERATION_STRUCTURE_SHOW)
	add_definitions(-DDEBUG_SHOW_TREE)
endif()

option(DEBUG_FORCE_CONSOLE "Force launch with debug console" OFF)
if(DEBUG_FORCE_CONSOLE)
	add_definitions(-DFORCE_CONSOLE)
endif()



###############################################################################
# Math library selection                                                      #
###############################################################################
set(TRACY_MATH_LIBRARY "cclib" CACHE STRING "Math Library")
set_property(CACHE TRACY_MATH_LIBRARY PROPERTY STRINGS cclib glm)

if(TRACY_MATH_LIBRARY STREQUAL "glm")
	add_definitions(-DUSE_GLM)
else()
	add_definitions(-DUSE_CCLIB)
	set(cclib_includes "./ext/cclib/cclib.h")
	source_group("ext\\cclib\\" FILES ${cclib_includes})
	list(APPEND tracy_includes ${cclib_includes})
endif()



###############################################################################
# Misc options                                                                #
###############################################################################
set(TRACY_TONEMAPPING "sRGB" CACHE STRING "Tonemapper Function")
set_property(CACHE TRACY_TONEMAPPING PROPERTY STRINGS None sRGB Aces Reinhard)
if(TRACY_TONEMAPPING STREQUAL "Aces")
	add_definitions(-DUSE_TONEMAP_ACES)
elseif(TRACY_TONEMAPPING STREQUAL "Reinhard")
	add_definitions(-DUSE_TONEMAP_REINHARD)
elseif(TRACY_TONEMAPPING STREQUAL "sRGB")
	add_definitions(-DUSE_TONEMAP_SRGB)
endif()

set(TRACY_EXPOSURE "1" CACHE STRING "Exposure")
add_definitions(-DTRACY_EXPOSURE=${TRACY_EXPOSURE})

option(TRACY_ACCUMULATE_SAMPLES "Accumulate samples over time" ON)
if(TRACY_ACCUMULATE_SAMPLES)
	add_definitions(-DACCUMULATE_SAMPLES)
endif()

set(TRACY_MAX_BOUNCES "5" CACHE STRING "Max Ray Bounces") 
add_definitions(-DTRACY_MAX_BOUNCES=${TRACY_MAX_BOUNCES})

option(TRACY_RUSSIAN_ROULETTE "Use Russian roulette for early ray termination" ON)
if(TRACY_RUSSIAN_ROULETTE)
	add_definitions(-DUSE_RUSSIAN_ROULETTE)
endif()


###############################################################################
# Common Sources                                                              #
###############################################################################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(. ./src ./ext)

set(ext_includes "./ext/stb_image.h"
				 "./ext/tiny_obj_loader.h")
source_group("ext\\" FILES ${ext_includes})
list(APPEND tracy_includes ${ext_includes})

set(main_sources "./src/win_raytracer.cpp"
                 "./src/scene.cpp"
				 "./src/bitmap.cpp")

source_group("src\\" FILES ${main_sources})

list(APPEND tracy_sources ${main_sources})


set(main_includes "./src/mesh.h"
                  "./src/scene.h"
                  "./src/material.h"
				  "./src/texture.h"
                  "./src/common.h"
				  "./src/module.h"
                  "./src/ray.h"
                  "./src/camera.h"
                  "./src/timer.h"
                  "./src/random.h"
                  "./src/vertex.h"
				  "./src/container.h"
				  "./src/aabb.h"
				  "./src/collision.h"
				  "./src/kdtree.h"
				  "./src/bvh.h"
				  "./src/log.h"
				  "./src/input.h"
				  "./src/bitmap.h")

source_group("include\\" FILES ${main_includes})

list(APPEND tracy_includes ${main_includes})



###############################################################################
# Kernel selection                                                            #
###############################################################################

set(CPURTX_ON True) # no dependencies - enable cpu rtx
set(CPU_ON True) # no dependencies - enable cpu raster
set(CUDA_ON False)
set(OPENGL_ON False)

find_package(OpenGL)
find_package(CUDA)

if(OpenGL_FOUND)
	set(OPENGL_ON True)
	
	if(CUDA_FOUND)
		set(CUDA_ON True)
	endif()

endif()

if(CPURTX_ON)

	message(STATUS "Enabling CPU Raytracing")
	add_definitions(-DCPU_KERNEL=1)

	include_directories(./src/kernels/raytracing/software/)

	set(CPU_RAND_ALGORITHM "PCG" CACHE STRING "Random Algorithm to use")
	set_property(CACHE CPU_RAND_ALGORITHM PROPERTY STRINGS "rand()" "XorShift" "LCG" "PCG")
	
	option(CPU_NO_OPENMP "Disable OpenMP in CPU Trace loop" OFF)

	option(CPU_TILED_RENDER "Split screen in tiles and render indipendently" OFF)

	option(CPU_USE_SIMD_INTRINSICS "Enable hand-optimized hot path code with SIMD intrinsics" OFF)
	
	set(cpu_includes "./src/kernels/raytracing/software/cpu_trace.h"
					 "./src/kernels/raytracing/software/cpu_details.h")

	source_group("kernel\\raytracing\\cpu\\include" FILES ${cpu_includes})
	list(APPEND tracy_includes ${cpu_includes})

	set(cpu_sources "./src/kernels/raytracing/software/cpu_trace.cpp"
	                "./src/kernels/raytracing/software/cpu_details.cpp")
	source_group("kernel\\raytracing\\cpu\\src" FILES ${cpu_sources})
	list(APPEND tracy_sources ${cpu_sources})
	
	if(CPU_RAND_ALGORITHM STREQUAL "XorShift")
		add_definitions(-DRANDOM_XORSHIFT=1)
	elseif(CPU_RAND_ALGORITHM STREQUAL "LCG")
		add_definitions(-DRANDOM_LCG=1)
	elseif(CPU_RAND_ALGORITHM STREQUAL "PCG")
		add_definitions(-DRANDOM_PCG=1)
	endif()

	if (CPU_TILED_RENDER)
		add_definitions(-DTILED_RENDERING=1)
	endif()

	if (CPU_USE_SIMD_INTRINSICS)
		add_definitions(-DUSE_INTRINSICS=1)
	endif()
	
endif()
if(CUDA_ON)

	message(STATUS "Enabling CUDA Raytracing")
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

	# deprecated but currently the only way to detect cuda arch
	# see https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures
	include(FindCUDA/select_compute_arch)
	CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
	string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
	string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
	string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

	set(CMAKE_CUDA_FLAGS "-D_FORCE_INLINES --expt-relaxed-constexpr")
	set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -D_DEBUG")
	set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG -lineinfo")

	set(cuda_sources "./src/kernels/raytracing/cuda/cuda_trace.cpp"
	                 "./src/kernels/raytracing/cuda/cuda_details.cpp"
	                 "./src/kernels/raytracing/cuda/cuda_trace.cu")
	source_group("kernel\\raytracing\\cuda\\src" FILES ${cuda_sources})
	list(APPEND tracy_sources ${cuda_sources})

	set(cuda_includes "./src/kernels/raytracing/cuda/cuda_trace.h"
	                  "./src/kernels/raytracing/cuda/cuda_details.h"
	                  "./src/kernels/raytracing/cuda/cuda_log.h"
					  "./src/kernels/raytracing/cuda/cuda_trace.cuh")
	source_group("kernel\\raytracing\\cuda\\include" FILES ${cuda_includes})
	list(APPEND tracy_includes ${cuda_includes})

endif()
if(OPENGL_ON)

	message(STATUS "Enabling OpenGL")
	add_definitions(-DOPENGL_KERNEL=1)

	find_package(OpenGL REQUIRED)
	list(APPEND tracy_libraries ${OPENGL_LIBRARIES})

	enable_language(C)
	add_definitions(-DGLEW_STATIC)
	include_directories(./ext/glew/include)
	set(glew_source "./ext/glew/glew.c")
	source_group("ext\\glew" FILES ${glew_source})
	list(APPEND tracy_sources ${glew_source})
	
	include_directories(./src/kernels/raster/opengl/)

	set(opengl_sources "./src/kernels/raster/opengl/opengl_render.cpp")
	source_group("kernel\\raster\\opengl\\src" FILES ${opengl_sources})
	list(APPEND tracy_sources ${opengl_sources})

	set(opengl_includes "./src/kernels/raster/opengl/opengl_render.h"
	                    "./src/kernels/raster/opengl/gl_mesh.h"
						"./src/kernels/raster/opengl/gl_material.h")
	source_group("kernel\\raster\\opengl\\include" FILES ${opengl_includes})
	list(APPEND tracy_includes ${opengl_includes})

endif()
if(CPU_ON)

	message(STATUS "Enabling CPU")
	add_definitions(-DCPU_RASTER_KERNEL=1)

	include_directories(./src/kernels/raster/cpu/)

	set(cpu_sources "./src/kernels/raster/cpu/cpu_render.cpp")
	source_group("kernel\\raster\\cpu\\src" FILES ${cpu_sources})
	list(APPEND tracy_sources ${cpu_sources})

	set(cpu_includes "./src/kernels/raster/cpu/cpu_render.h")
	source_group("kernel\\raster\\cpu\\include" FILES ${cpu_includes})
	list(APPEND tracy_includes ${cpu_includes})

endif()



###############################################################################
# Compiler options                                                            #
###############################################################################
message(STATUS "Found Compiler " ${CMAKE_CXX_COMPILER_ID} " " ${CMAKE_CXX_COMPILER_VERSION} " : " ${CMAKE_CXX_COMPILER})
if(MSVC)

	message(STATUS "MSVC Version " ${MSVC_VERSION})

    set(CMAKE_CXX_FLAGS "/nologo /W3 /D_CRT_SECURE_NO_WARNINGS /MP /Gm- /Zi")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /D_DEBUG /EHsc")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GR- /Gw /GL /O2 /Ob1 /Oi /Gv /fp:fast /Qpar /DNDEBUG /D_HAS_EXCEPTIONS=0")
	
	# enable auto vectorizer report
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qvec-report:2")

	# enable auto parallelizer report
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qpar-report:2")

	# TODO: MSVC lacks something like -march=native to enable best instruction set available on host compiler
	#set(MARCH_NATIVE "/ARCH:(SSE,SSE2,AVX,AVX2,AVX512,...)")
	#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${MARCH_NATIVE}")

	if(CPU_NO_OPENMP)
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp-")
	else()
		# choose "experimental" openmp implementation if available
		# (and suppress warning C4849 about 'collapse' clause ignored)
		#
		# TODO: test openmp:llvm (https://devblogs.microsoft.com/cppblog/improved-openmp-support-for-cpp-in-visual-studio/)
		#
		if(MSVC_VERSION GREATER_EQUAL 1920)
			set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp:experimental /wd4849")
		else()
			set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /openmp /wd4849")
		endif()
	endif()
    
	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /LTCG /INCREMENTAL:NO")

else()

    set(CMAKE_CXX_FLAGS "-Wall -g")
    set(CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Ofast -march=native -fno-rtti -fno-exceptions -funroll-loops")
	
	if(NOT CPU_NO_OPENMP)
		find_package(OpenMP REQUIRED)
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
	endif()
	
	if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsingle-precision-constant -flto")
	endif()

endif()



###############################################################################
# Build Rules                                                                 #
###############################################################################
if(WIN32 AND NOT DEBUG_FORCE_CONSOLE)
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
