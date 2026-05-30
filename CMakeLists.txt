cmake_minimum_required(VERSION 3.10)
project(MLPCpp_AD_Test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- SU2 root (for CoDiPack) ----
set(SU2_ROOT /root/SU2)

include_directories(
    ${SU2_ROOT}
    ${SU2_ROOT}/Common/include
    ${SU2_ROOT}/externals/codi/include
)

add_executable(mlpcpp_ad_test
    TestCase/test_codi.cpp
)

target_compile_definitions(mlpcpp_ad_test PRIVATE
    CODI_REVERSE_TYPE
    ENABLE_SU2_AD
    HAVE_CODI
)