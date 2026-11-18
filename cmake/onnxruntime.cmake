# Provides the imported target onnxruntime::onnxruntime.
#
# Uses an existing install if ONNXRUNTIME_ROOT is set (a folder with include/ and lib/), and
# otherwise downloads Microsoft's Microsoft.ML.OnnxRuntime package from nuget.org (one package
# with headers and prebuilt libraries for every platform), checked against its SHA-256, and
# extracts only this platform's files.

set(KIBITZ_ORT_VERSION 1.31.1)
set(KIBITZ_ORT_SHA256 6871ccbc97f3d5bdb4095a1dcec33aa78d56f6ed8de5980ef5fa1d9f2e7c679a)

if(ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
  set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)
  if(WIN32)
    set(_ort_lib onnxruntime.dll)
  elseif(APPLE)
    set(_ort_lib libonnxruntime.dylib)
  else()
    set(_ort_lib libonnxruntime.so)
  endif()
else()
  # The file names libkibitz will look for at run time are the libraries' own SONAME / install
  # name. The package stores them under their unversioned names, so they are renamed on extract.
  if(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    set(_ort_rid win-x64)
    set(_ort_lib onnxruntime.dll)
  elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_ort_rid osx-arm64)
    set(_ort_file libonnxruntime.dylib)
    set(_ort_lib libonnxruntime.1.dylib)
  elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_ort_rid linux-x64)
    set(_ort_file libonnxruntime.so)
    set(_ort_lib libonnxruntime.so.1)
  else()
    message(FATAL_ERROR
      "No prebuilt ONNX Runtime for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. "
      "Set ONNXRUNTIME_ROOT, or configure with -DKIBITZ_WITH_ONNXRUNTIME=OFF.")
  endif()

  set(_ort_dir ${CMAKE_BINARY_DIR}/_deps/onnxruntime-${KIBITZ_ORT_VERSION})
  set(_ort_pkg ${CMAKE_BINARY_DIR}/_deps/microsoft.ml.onnxruntime.${KIBITZ_ORT_VERSION}.nupkg)
  set(ONNXRUNTIME_INCLUDE_DIR ${_ort_dir}/include)
  set(ONNXRUNTIME_LIB_DIR ${_ort_dir}/lib)

  if(NOT EXISTS ${ONNXRUNTIME_LIB_DIR}/${_ort_lib})
    message(STATUS "Downloading ONNX Runtime ${KIBITZ_ORT_VERSION} from nuget.org")
    # Skipped if the file is already there with the right hash.
    file(DOWNLOAD
      https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${KIBITZ_ORT_VERSION}/microsoft.ml.onnxruntime.${KIBITZ_ORT_VERSION}.nupkg
      ${_ort_pkg}
      EXPECTED_HASH SHA256=${KIBITZ_ORT_SHA256}
      SHOW_PROGRESS)
    set(_ort_tmp ${_ort_dir}-extract)
    file(REMOVE_RECURSE ${_ort_tmp})
    # A .nupkg is a zip file; take the headers and this platform's libraries only.
    file(ARCHIVE_EXTRACT INPUT ${_ort_pkg} DESTINATION ${_ort_tmp}
      PATTERNS build/native/include runtimes/${_ort_rid}/native)
    file(REMOVE_RECURSE ${_ort_dir})
    file(MAKE_DIRECTORY ${ONNXRUNTIME_LIB_DIR})
    file(RENAME ${_ort_tmp}/build/native/include ${ONNXRUNTIME_INCLUDE_DIR})
    set(_ort_src ${_ort_tmp}/runtimes/${_ort_rid}/native)
    if(WIN32)
      foreach(f onnxruntime.dll onnxruntime.lib)
        file(RENAME ${_ort_src}/${f} ${ONNXRUNTIME_LIB_DIR}/${f})
      endforeach()
    else()
      # libonnxruntime.so -> libonnxruntime.so.1, libonnxruntime.dylib -> libonnxruntime.1.dylib
      file(RENAME ${_ort_src}/${_ort_file} ${ONNXRUNTIME_LIB_DIR}/${_ort_lib})
    endif()
    file(REMOVE_RECURSE ${_ort_tmp})
  endif()
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
target_include_directories(onnxruntime::onnxruntime INTERFACE ${ONNXRUNTIME_INCLUDE_DIR})
set_target_properties(onnxruntime::onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRUNTIME_LIB_DIR}/${_ort_lib})
if(WIN32)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES IMPORTED_IMPLIB ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib)
endif()
