cmake_minimum_required(VERSION 3.20)
project(libmut)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FindPkgConfig)

# libprim
pkg_check_modules(libprim REQUIRED IMPORTED_TARGET libprim)
  get_target_property(
  LIBPRIM_INC
  PkgConfig::libprim
  INTERFACE_INCLUDE_DIRECTORIES
)

add_library(
  mut
  SHARED
  ${PROJECT_SOURCE_DIR}/src/mut/mut.cc
  ${PROJECT_SOURCE_DIR}/src/mut/mut.h
  ${PROJECT_SOURCE_DIR}/src/mut/mut.tcc
  )

set_target_properties(
  mut
  PROPERTIES
  LINKER_LANGUAGE
  CXX
  POSITION_INDEPENDENT_CODE
  ON
  )

target_include_directories(
  mut
  PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${LIBPRIM_INC}
  )

target_link_libraries(
  mut
  PkgConfig::libprim
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

configure_file(
  ${PROJECT_SOURCE_DIR}/libmut.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libmut.pc
  @ONLY
  )
install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libmut.pc
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/pkgconfig/
  )
