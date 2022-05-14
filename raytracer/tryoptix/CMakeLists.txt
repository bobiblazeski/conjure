#cmake_minimum_required(VERSION 2.8.9)
#project (hello)

include_directories(${PROJECT_SOURCE_DIR}/rayoptix/)

add_executable(tryoptixexe tryoptix/tryoptix.cpp)

target_link_libraries(tryoptixexe
  rayoptix  
)

