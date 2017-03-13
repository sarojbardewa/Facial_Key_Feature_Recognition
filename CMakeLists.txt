# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(faceDetector)

# Configure the build to print all details.
SET(CMAKE_VERBOSE_MAKEFILE ON)

# Configure the build type. This can be passed as parameter too.
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

# Find CUDA, you may need to set CUDA_DIR variable
# to the absolute path to the directory containing CUDAConfig.cmake file
# via the command line or GUI
find_package(CUDA REQUIRED)
# /usr/share/cmake/Modules/FindCUDA.cmake
INCLUDE(FindCUDA)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the CUDAConfig.cmake file.
# Print some message showing some of them
message(STATUS "CUDA library status:")
message(STATUS "    toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#Bring the headers, *.h into the project
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

#Bring the sources, *.cpp into the project
#set(SOURCES ./src/main.cu)
file(GLOB FILESOURCES 
	"./src/*.cu"
	"./src/*.cpp"
	"./src/*.h"
	".src/kernels/*.cu"
)


# Declare the executable target built from your sources
cuda_add_executable(faceDetector ${FILESOURCES})
set_target_properties(faceDetector PROPERTIES LINKER_LANGUAGE CXX)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 ")#-ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

# Link your application with CUDA and OpenCV libraries
target_link_libraries(faceDetector ${CUDA_LIBRARIES} ${OpenCV_LIBS})


