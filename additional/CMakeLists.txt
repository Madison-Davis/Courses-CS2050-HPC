cmake_minimum_required(VERSION 3.16)
project(kokkos_benchmark CXX)

# Point to your installed Kokkos
find_package(Kokkos REQUIRED)

add_executable(run_kokkos run_kokkos.cpp)

# Link against Kokkos — this automatically pulls in all flags, includes, and libs
target_link_libraries(run_kokkos Kokkos::kokkos)

# Include your additional config headers
target_include_directories(run_kokkos PRIVATE ../additional)