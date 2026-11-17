# Provides the imported target onnxruntime::onnxruntime.
#
# Uses an existing install if ONNXRUNTIME_ROOT is set (a folder with include/ and lib/), and
# otherwise downloads Microsoft's prebuilt release for this platform, checked against its SHA-256.

set(KIBITZ_ORT_VERSION 1.30.1)

if(NOT ONNXRUNTIME_ROOT)
  if(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    set(_ort_pkg onnxruntime-win-x64-${KIBITZ_ORT_VERSION}.zip)
    set(_ort_sha 9e70d829a7c2014c6f6d2889cd3578733c2e5bbe919e186b5f2038d59176ab13)
  elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_ort_pkg onnxruntime-osx-arm64-${KIBITZ_ORT_VERSION}.tgz)
    set(_ort_sha b9b1f72babd1b121b2323ac3069939623ce1d75db693b282bf5c2e593d340c79)
  elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_ort_pkg onnxruntime-linux-x64-${KIBITZ_ORT_VERSION}.tgz)
    set(_ort_sha 35a95c9c2b298bbd7dd2d63c73b333adb039b7a0faa04790fa69243dfddefdd3)
  else()
    message(FATAL_ERROR
      "No prebuilt ONNX Runtime for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. "
      "Set ONNXRUNTIME_ROOT, or configure with -DKIBITZ_WITH_ONNXRUNTIME=OFF.")
  endif()

  include(FetchContent)
  FetchContent_Declare(onnxruntime_prebuilt
    URL https://github.com/microsoft/onnxruntime/releases/download/v${KIBITZ_ORT_VERSION}/${_ort_pkg}
    URL_HASH SHA256=${_ort_sha}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(onnxruntime_prebuilt)
  set(ONNXRUNTIME_ROOT ${onnxruntime_prebuilt_SOURCE_DIR})
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
target_include_directories(onnxruntime::onnxruntime INTERFACE ${ONNXRUNTIME_ROOT}/include)
if(WIN32)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_IMPLIB ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib
    IMPORTED_LOCATION ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll)
elseif(APPLE)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
else()
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()
