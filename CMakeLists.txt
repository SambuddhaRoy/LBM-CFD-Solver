cmake_minimum_required(VERSION 3.24)
project(VirtualWindTunnel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive- /wd4100 /wd4201)
    add_link_options(/LARGEADDRESSAWARE)
    # Suppress console window on Windows release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_link_options(/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
    endif()
elseif(LINUX)
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Vulkan 1.3 REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_path(XLSXWRITER_INCLUDE_DIR NAMES xlsxwriter.h)
find_library(XLSXWRITER_LIBRARY NAMES xlsxwriter)

set(SOURCES
    src/main.cpp
    src/mesh_loader.cpp
    src/fluid_solver.cpp
    src/renderer.cpp
    src/vk_engine.cpp
    src/benchmark/auto_benchmark.cpp
)

set(HEADERS
    src/mesh_loader.h
    src/fluid_solver.h
    src/renderer.h
    src/vk_engine.h
    src/vk_types.h
    src/environment.h
    src/sim_scaler.h
    src/logger.h
    src/benchmark/auto_benchmark.h
    src/benchmark/benchmark_config.h
    src/benchmark/model_selector.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    imgui::imgui
    assimp::assimp
    glm::glm
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    ${XLSXWRITER_LIBRARY}
)

if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE Comdlg32)
endif()

if(LINUX)
    # Vulkan_LIBRARIES is set by find_package(Vulkan); fall back to Vulkan_LIBRARY
    set(_VK_LIB "${Vulkan_LIBRARIES}")
    if(NOT _VK_LIB)
        set(_VK_LIB "${Vulkan_LIBRARY}")
    endif()
    if(_VK_LIB)
        get_filename_component(VULKAN_LIB_DIR "${_VK_LIB}" DIRECTORY)
        target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-rpath,${VULKAN_LIB_DIR}")
    endif()
endif()

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${XLSXWRITER_INCLUDE_DIR}
)

# ─── Compile shaders ────────────────────────────────────────────────────────
if(Vulkan_GLSLC_EXECUTABLE)
    set(GLSLC ${Vulkan_GLSLC_EXECUTABLE})
else()
    find_program(GLSLC glslc HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
    )
endif()

if(NOT GLSLC)
    message(WARNING "glslc not found — shaders will not be recompiled.")
endif()
message(STATUS "glslc: ${GLSLC}")

set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SPIRV_DIR  ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPIRV_DIR})

file(GLOB SHADER_SOURCES
    "${SHADER_DIR}/*.comp"
    "${SHADER_DIR}/*.vert"
    "${SHADER_DIR}/*.frag"
)

foreach(SRC ${SHADER_SOURCES})
    get_filename_component(NAME ${SRC} NAME)
    set(SPV "${SPIRV_DIR}/${NAME}.spv")
    if(GLSLC)
        add_custom_command(
            OUTPUT  ${SPV}
            COMMAND ${GLSLC} -O ${SRC} -o ${SPV}
            DEPENDS ${SRC}
            COMMENT "Compiling shader: ${NAME}"
        )
    endif()
    list(APPEND SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_OUTPUTS})
add_dependencies(${PROJECT_NAME} shaders)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SPIRV_DIR}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
)
