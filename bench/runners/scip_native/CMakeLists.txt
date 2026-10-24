cmake_minimum_required(VERSION 3.18)
project(qopt_scip_diagnostic C)

find_package(SCIP CONFIG REQUIRED)

add_executable(qopt_scip_diagnostic main.c)
target_link_libraries(qopt_scip_diagnostic PRIVATE libscip)
target_include_directories(qopt_scip_diagnostic PRIVATE ${SCIP_INCLUDE_DIRS})
set_target_properties(qopt_scip_diagnostic PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:libscip>"
)
