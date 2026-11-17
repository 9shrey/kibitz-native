include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # match MSVC's default runtime
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

set(data_defs KIBITZ_MODELS_DIR="${KIBITZ_MODELS_DIR}" KIBITZ_TEST_DATA_DIR="${KIBITZ_TEST_DATA_DIR}")

# Unit tests of internals: compiled straight in, since the shared library hides them.
add_executable(kibitz_unit_tests test_features.cpp ${PROJECT_SOURCE_DIR}/src/features.cpp)
target_include_directories(kibitz_unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(kibitz_unit_tests PRIVATE ${data_defs})
target_link_libraries(kibitz_unit_tests PRIVATE GTest::gtest_main kibitz_warnings)
gtest_discover_tests(kibitz_unit_tests NO_PRETTY_VALUES DISCOVERY_MODE PRE_TEST)

# Tests of the public C API, linked against the shared library like any other user.
add_executable(kibitz_api_tests test_api.cpp)
target_compile_definitions(kibitz_api_tests PRIVATE ${data_defs})
target_link_libraries(kibitz_api_tests PRIVATE kibitz::kibitz GTest::gtest_main kibitz_warnings)
gtest_discover_tests(kibitz_api_tests NO_PRETTY_VALUES DISCOVERY_MODE PRE_TEST
  PROPERTIES ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan.supp")

# The same library loaded from Python through ctypes.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME python_ctypes
    COMMAND ${Python3_EXECUTABLE} -m unittest -v test_ctypes
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(python_ctypes PROPERTIES ENVIRONMENT
    "KIBITZ_LIBRARY=$<TARGET_FILE:kibitz>;KIBITZ_MODELS_DIR=${KIBITZ_MODELS_DIR};KIBITZ_TEST_DATA_DIR=${KIBITZ_TEST_DATA_DIR};PYTHONPATH=${PROJECT_SOURCE_DIR}/python")
  if(KIBITZ_SANITIZE)
    # An ASan-instrumented library needs the ASan runtime preloaded into the Python process.
    set_tests_properties(python_ctypes PROPERTIES DISABLED TRUE)
  endif()
endif()
