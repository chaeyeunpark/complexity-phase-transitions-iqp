cmake_minimum_required(VERSION 3.17)

project(iqp_dist_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(kokkos
                    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
                    GIT_TAG        4.1.00
                    GIT_SUBMODULES "" # Avoid recursively cloning all submodules
)

FetchContent_MakeAvailable(kokkos)

find_package(MPI REQUIRED)

get_target_property(kokkos_INC_DIR kokkos INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(kokkos PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Kokkos_INC_DIR}")

add_library(iqp_dist_cpp_external_libs INTERFACE)
target_include_directories(iqp_dist_cpp_external_libs INTERFACE ./external)

add_executable(iqp_dist_scaled_mpi src/iqp_dist_scaled_mpi.cpp)
target_link_libraries(iqp_dist_scaled_mpi PRIVATE kokkos iqp_dist_cpp_external_libs MPI::MPI_CXX)
