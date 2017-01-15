project(Tracy)

cmake_minimum_required(VERSION 2.8)

include_directories(.)
include_directories(./src)

file(GLOB_RECURSE INC_LIST "./src/*.hpp" "./src/*.h")
aux_source_directory(./src SRC_LIST)

add_executable(${PROJECT_NAME} ${SRC_LIST} ${INC_LIST})
