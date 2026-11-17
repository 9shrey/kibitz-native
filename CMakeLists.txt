cmake_minimum_required(VERSION 3.24)
project(kibitz_native VERSION 0.1.0 DESCRIPTION "Kibitz chess evaluation network as a C/C++ shared library" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KIBITZ_WITH_ONNXRUNTIME "Build the ONNX Runtime backend" ON)
option(KIBITZ_BUILD_TESTS "Build the tests" ON)
option(KIBITZ_BUILD_BENCH "Build the benchmark" ON)
option(KIBITZ_WERROR "Treat warnings as errors" OFF)
option(KIBITZ_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Put every .so/.dll and executable in one folder so Windows finds DLLs next to the .exe files.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_library(kibitz_warnings INTERFACE)
if(MSVC)
  target_compile_options(kibitz_warnings INTERFACE /W4 /permissive- $<$<BOOL:${KIBITZ_WERROR}>:/WX>)
else()
  target_compile_options(kibitz_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion
    -Wno-sign-conversion $<$<BOOL:${KIBITZ_WERROR}>:-Werror>)
endif()
if(KIBITZ_SANITIZE AND NOT MSVC)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

# --- the library ------------------------------------------------------------------------------

add_library(kibitz SHARED
  src/kibitz.cpp
  src/features.cpp
  src/native_backend.cpp)
add_library(kibitz::kibitz ALIAS kibitz)
target_include_directories(kibitz PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_definitions(kibitz PRIVATE KIBITZ_BUILDING KIBITZ_VERSION="${PROJECT_VERSION}")
target_link_libraries(kibitz PRIVATE kibitz_warnings)
# Export only the functions marked KIBITZ_API; everything else stays internal to the .so/.dll.
set_target_properties(kibitz PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

if(KIBITZ_WITH_ONNXRUNTIME)
  include(cmake/onnxruntime.cmake)
  target_sources(kibitz PRIVATE src/onnxruntime_backend.cpp)
  target_compile_definitions(kibitz PRIVATE KIBITZ_WITH_ONNXRUNTIME)
  target_link_libraries(kibitz PRIVATE onnxruntime::onnxruntime)
  # Find libonnxruntime next to libkibitz at run time, wherever the folder is moved to.
  if(APPLE)
    set_target_properties(kibitz PROPERTIES BUILD_RPATH "@loader_path;${ONNXRUNTIME_ROOT}/lib" INSTALL_RPATH "@loader_path")
  elseif(UNIX)
    set_target_properties(kibitz PROPERTIES BUILD_RPATH "$ORIGIN;${ONNXRUNTIME_ROOT}/lib" INSTALL_RPATH "$ORIGIN")
  endif()
  if(WIN32)
    add_custom_command(TARGET kibitz POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime::onnxruntime> $<TARGET_FILE_DIR:kibitz>)
  endif()
endif()

# --- command-line tool ------------------------------------------------------------------------

add_executable(kibitz-eval tools/kibitz_eval.c)
target_link_libraries(kibitz-eval PRIVATE kibitz::kibitz)
# The tool is C, but libkibitz is C++: link with the C++ driver so the C++ runtime (and, in
# sanitizer builds, UBSan's C++ runtime) comes along. Linking with the C driver fails under Clang.
set_target_properties(kibitz-eval PROPERTIES LINKER_LANGUAGE CXX)

# --- tests and benchmark ----------------------------------------------------------------------

set(KIBITZ_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/models)
set(KIBITZ_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)

if(KIBITZ_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(KIBITZ_BUILD_BENCH)
  add_executable(kibitz-bench bench/bench.cpp)
  target_link_libraries(kibitz-bench PRIVATE kibitz::kibitz kibitz_warnings)
  target_compile_definitions(kibitz-bench PRIVATE
    KIBITZ_MODELS_DIR="${KIBITZ_MODELS_DIR}" KIBITZ_TEST_DATA_DIR="${KIBITZ_TEST_DATA_DIR}")
endif()

# --- install ----------------------------------------------------------------------------------

include(GNUInstallDirs)
install(TARGETS kibitz kibitz-eval)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES models/kibitz.onnx models/kibitz.weights DESTINATION ${CMAKE_INSTALL_DATADIR}/kibitz)
if(KIBITZ_WITH_ONNXRUNTIME)
  install(FILES $<TARGET_FILE:onnxruntime::onnxruntime> DESTINATION
    $<IF:$<BOOL:${WIN32}>,${CMAKE_INSTALL_BINDIR},${CMAKE_INSTALL_LIBDIR}>)
endif()
