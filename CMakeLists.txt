project(Tracy)

cmake_minimum_required(VERSION 2.8)

include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLFW_INCLUDE_DIRS})

aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES})
