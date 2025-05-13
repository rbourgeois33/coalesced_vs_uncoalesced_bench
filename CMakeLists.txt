cmake_minimum_required(VERSION 3.15)
project(main CXX)

# Use FetchContent to download Kokkos
include(FetchContent)

FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
)

FetchContent_MakeAvailable(kokkos)

add_executable(main main.cpp)

# Link Kokkos (handles flags for you)
target_link_libraries(main Kokkos::kokkos)