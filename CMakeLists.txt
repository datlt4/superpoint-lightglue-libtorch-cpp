# CMakeLists.txt
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch-app)


# Pytorch
# Set the path for LibTorch
set(TORCH_CMAKE_PREFIX_PATH "/usr/local/libtorch/")
# Add -DCMAKE_PREFIX_PATH when calling find_package or configuring
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(LIB ${LIB} "${TORCH_LIBRARIES}")
set(INC ${INC} ${TORCH_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
set(LIB ${LIB} ${OpenCV_LIBRARIES})
set(INC ${INC} ${OpenCV_INCLUDE_DIRS})

# Superpoint Lightglue
set(INC ${INC} "${PROJECT_SOURCE_DIR}/SuperpointLightglue/")
set(SRC ${SRC} "${PROJECT_SOURCE_DIR}/SuperpointLightglue/superpoint.cpp")
set(SRC ${SRC} "${PROJECT_SOURCE_DIR}/SuperpointLightglue/lightglue.cpp")
set(SRC ${SRC} "${PROJECT_SOURCE_DIR}/SuperpointLightglue/utils.cpp")
set(SRC ${SRC} "${PROJECT_SOURCE_DIR}/SuperpointLightglue/feature_matching.cpp")

# Main
set(SRC ${SRC} "${PROJECT_SOURCE_DIR}/main.cpp")

# Executable target
add_executable(${PROJECT_NAME} main.cpp ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC ${INC})
target_link_libraries(${PROJECT_NAME} ${LIB})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET torch-app
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:torch-app>)
endif (MSVC)

# export LIB_TORCH=/usr/local/libtorch/
# cmake -DCMAKE_PREFIX_PATH=$LIB_TORCH ..
