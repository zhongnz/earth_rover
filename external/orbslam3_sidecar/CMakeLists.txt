cmake_minimum_required(VERSION 3.16)
project(orbslam3_sidecar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ORB_SLAM3_ROOT "${CMAKE_SOURCE_DIR}/../ORB_SLAM3" CACHE PATH "Path to ORB-SLAM3 checkout")
set(ORB_SLAM3_INCLUDE_HINTS
    "${ORB_SLAM3_ROOT}/include"
    "${ORB_SLAM3_ROOT}"
)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

find_path(ORB_SLAM3_INCLUDE_DIR
    NAMES System.h
    PATHS ${ORB_SLAM3_INCLUDE_HINTS}
    NO_DEFAULT_PATH
)

if(NOT ORB_SLAM3_INCLUDE_DIR)
    message(FATAL_ERROR "ORB-SLAM3 headers not found. Expected checkout under ${ORB_SLAM3_ROOT}. Run scripts/setup_orbslam3.sh first.")
endif()

add_executable(orbslam3_sidecar
    src/main.cpp
)

target_include_directories(orbslam3_sidecar PRIVATE
    ${ORB_SLAM3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(orbslam3_sidecar PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)
