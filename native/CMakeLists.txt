cmake_minimum_required(VERSION 3.15)
project(stabdisc_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

add_executable(e8_sign_m2 src/e8_sign_t2_exact.cpp)
target_compile_options(e8_sign_m2 PRIVATE -O3)
target_link_libraries(e8_sign_m2 PRIVATE Threads::Threads)

add_executable(e8_sign_m3 src/e8_sign_t3_raw_parallel.cpp)
target_compile_options(e8_sign_m3 PRIVATE -O3)
target_link_libraries(e8_sign_m3 PRIVATE Threads::Threads)

foreach(copies RANGE 1 4)
    add_executable(e8_xor_k${copies} src/e8_xor_t2_raw_parallel.cpp)
    target_compile_options(e8_xor_k${copies} PRIVATE -O3)
    target_link_libraries(e8_xor_k${copies} PRIVATE Threads::Threads)
    target_compile_definitions(e8_xor_k${copies} PRIVATE XOR_COPIES=${copies})
endforeach()

add_executable(twoq_xor_k2 src/twoqubit_xor_raw_parallel.cpp)
target_compile_options(twoq_xor_k2 PRIVATE -O3)
target_link_libraries(twoq_xor_k2 PRIVATE Threads::Threads)
target_compile_definitions(twoq_xor_k2 PRIVATE XOR_COPIES=2)

add_executable(twoq_xor_k3 src/twoqubit_xor_raw_parallel.cpp)
target_compile_options(twoq_xor_k3 PRIVATE -O3)
target_link_libraries(twoq_xor_k3 PRIVATE Threads::Threads)
target_compile_definitions(twoq_xor_k3 PRIVATE XOR_COPIES=3)

add_executable(twoq_t_m0 src/twoqubit_tstate_discrimination.cpp)
target_compile_options(twoq_t_m0 PRIVATE -O3)
target_link_libraries(twoq_t_m0 PRIVATE Threads::Threads)
target_compile_definitions(twoq_t_m0 PRIVATE TSTATES=0)

add_executable(twoq_t_m1 src/twoqubit_tstate_discrimination.cpp)
target_compile_options(twoq_t_m1 PRIVATE -O3)
target_link_libraries(twoq_t_m1 PRIVATE Threads::Threads)
target_compile_definitions(twoq_t_m1 PRIVATE TSTATES=1)

add_executable(twoq_t_m2 src/twoqubit_tstate_discrimination.cpp)
target_compile_options(twoq_t_m2 PRIVATE -O3)
target_link_libraries(twoq_t_m2 PRIVATE Threads::Threads)
target_compile_definitions(twoq_t_m2 PRIVATE TSTATES=2)

add_executable(twoq_t_m3 src/twoqubit_tstate_discrimination.cpp)
target_compile_options(twoq_t_m3 PRIVATE -O3)
target_link_libraries(twoq_t_m3 PRIVATE Threads::Threads)
target_compile_definitions(twoq_t_m3 PRIVATE TSTATES=3)

add_executable(twoq_t_m4 src/twoqubit_tstate_discrimination.cpp)
target_compile_options(twoq_t_m4 PRIVATE -O3)
target_link_libraries(twoq_t_m4 PRIVATE Threads::Threads)
target_compile_definitions(twoq_t_m4 PRIVATE TSTATES=4)

add_executable(e6_xor_k2 src/e6_sign_xor_t2_raw_parallel.cpp)
target_compile_options(e6_xor_k2 PRIVATE -O3)
target_link_libraries(e6_xor_k2 PRIVATE Threads::Threads)

add_executable(e6_t_m0 src/e6_tstate_discrimination.cpp)
target_compile_options(e6_t_m0 PRIVATE -O3)
target_link_libraries(e6_t_m0 PRIVATE Threads::Threads)
target_compile_definitions(e6_t_m0 PRIVATE TSTATES=0)

add_executable(e6_t_m1 src/e6_tstate_discrimination.cpp)
target_compile_options(e6_t_m1 PRIVATE -O3)
target_link_libraries(e6_t_m1 PRIVATE Threads::Threads)
target_compile_definitions(e6_t_m1 PRIVATE TSTATES=1)

add_executable(e6_t_m2 src/e6_tstate_discrimination.cpp)
target_compile_options(e6_t_m2 PRIVATE -O3)
target_link_libraries(e6_t_m2 PRIVATE Threads::Threads)
target_compile_definitions(e6_t_m2 PRIVATE TSTATES=2)

add_executable(e6_t_m3 src/e6_tstate_discrimination.cpp)
target_compile_options(e6_t_m3 PRIVATE -O3)
target_link_libraries(e6_t_m3 PRIVATE Threads::Threads)
target_compile_definitions(e6_t_m3 PRIVATE TSTATES=3)

add_executable(e6_full_t3 src/e6_full_t3_raw_parallel.cpp)
target_compile_options(e6_full_t3 PRIVATE -O3)
target_link_libraries(e6_full_t3 PRIVATE Threads::Threads)

add_executable(four_state_m0 src/four_state_tstate_discrimination.cpp)
target_compile_options(four_state_m0 PRIVATE -O3)
target_link_libraries(four_state_m0 PRIVATE Threads::Threads)
target_compile_definitions(four_state_m0 PRIVATE TSTATES=0)

add_executable(four_state_m1 src/four_state_tstate_discrimination.cpp)
target_compile_options(four_state_m1 PRIVATE -O3)
target_link_libraries(four_state_m1 PRIVATE Threads::Threads)
target_compile_definitions(four_state_m1 PRIVATE TSTATES=1)

add_executable(four_state_m2 src/four_state_tstate_discrimination.cpp)
target_compile_options(four_state_m2 PRIVATE -O3)
target_link_libraries(four_state_m2 PRIVATE Threads::Threads)
target_compile_definitions(four_state_m2 PRIVATE TSTATES=2)

add_executable(four_state_m3 src/four_state_tstate_discrimination.cpp)
target_compile_options(four_state_m3 PRIVATE -O3)
target_link_libraries(four_state_m3 PRIVATE Threads::Threads)
target_compile_definitions(four_state_m3 PRIVATE TSTATES=3)

