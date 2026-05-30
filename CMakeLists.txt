cmake_minimum_required(VERSION 3.14)
project(MLPCpp_AD_Test)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SU2_ROOT /root/SU2)

add_executable(mlpcpp_ad_test
    TestCase/test_codi.cpp
)

target_include_directories(mlpcpp_ad_test PRIVATE
    ${SU2_ROOT}/Common/include
    ${SU2_ROOT}/externals/codi/include
    ${SU2_ROOT}/externals/medi/include
    ${SU2_ROOT}/externals/opdi/include
)

target_compile_definitions(mlpcpp_ad_test PRIVATE
    CODI_REVERSE_TYPE
    CODI_JACOBIAN_LINEAR_TAPE
    HAVE_CODI
)

target_link_libraries(mlpcpp_ad_test PRIVATE
    /root/SU2/build/Common/src/libSU2CommonAD.a
)