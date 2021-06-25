cmake_minimum_required(VERSION 3.10)
project(libmut)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# libprim
find_library(LIBPRIM_A NAMES libprim.a PATHS ${CMAKE_INSTALL_RPATH})
if(LIBPRIM_A MATCHES "NOTFOUND")
  message(FATAL_ERROR "libprim.a not found")
endif()
string(REPLACE lib/libprim.a include/ LIBPRIM_INC ${LIBPRIM_A})
if(NOT EXISTS ${LIBPRIM_INC})
  message(FATAL_ERROR "libprim include dir doesn't exist")
endif()

add_library(
  mut
  STATIC
  ${PROJECT_SOURCE_DIR}/src/mut/mut.cc
  ${PROJECT_SOURCE_DIR}/src/mut/mut.h
  ${PROJECT_SOURCE_DIR}/src/mut/mut.tcc
  )

set_target_properties(
  mut
  PROPERTIES
  LINKER_LANGUAGE
  CXX
  )

target_include_directories(
  mut
  PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${LIBPRIM_INC}
  )

target_link_libraries(
  mut
  ${LIBPRIM_A}
  )

include(GNUInstallDirs)

install(
  FILES
  ${PROJECT_SOURCE_DIR}/src/mut/mut.h
  DESTINATION
  ${CMAKE_INSTALL_INCLUDEDIR}/mut/
  )

install(
  FILES
  ${PROJECT_SOURCE_DIR}/src/mut/mut.tcc
  DESTINATION
  ${CMAKE_INSTALL_INCLUDEDIR}/mut/
  )

install(
  TARGETS
  mut
  )

