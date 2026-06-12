cmake_minimum_required(VERSION 3.24)
project(VirtualWindTunnel VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
    add_link_options(/LARGEADDRESSAWARE)
elseif(LINUX)
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ─── Dependencies (vcpkg) ───────────────────────────────────────────────────
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

# ─── Target ─────────────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/gpu.cpp
    src/sim.cpp
    src/mesh.cpp
    src/viz.cpp
    src/ui.cpp
    src/app.cpp
    src/validate.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    imgui::imgui
    assimp::assimp
    glm::glm
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
)

if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE Comdlg32)
endif()

if(LINUX)
    set(_VK_LIB "${Vulkan_LIBRARIES}")
    if(NOT _VK_LIB)
        set(_VK_LIB "${Vulkan_LIBRARY}")
    endif()
    if(_VK_LIB)
        get_filename_component(VULKAN_LIB_DIR "${_VK_LIB}" DIRECTORY)
        target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-rpath,${VULKAN_LIB_DIR}")
    endif()
endif()

# ─── Shader compilation (GLSL → SPIR-V) ─────────────────────────────────────
if(Vulkan_GLSLC_EXECUTABLE)
    set(GLSLC ${Vulkan_GLSLC_EXECUTABLE})
else()
    find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
endif()
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found — install the Vulkan SDK or vcpkg shaderc.")
endif()
message(STATUS "glslc: ${GLSLC}")

set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SPIRV_DIR  ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPIRV_DIR})

file(GLOB SHADER_SOURCES "${SHADER_DIR}/*.comp")

foreach(SRC ${SHADER_SOURCES})
    get_filename_component(NAME ${SRC} NAME)
    set(SPV "${SPIRV_DIR}/${NAME}.spv")
    add_custom_command(
        OUTPUT  ${SPV}
        COMMAND ${GLSLC} -O ${SRC} -o ${SPV}
        DEPENDS ${SRC}
        COMMENT "Compiling shader: ${NAME}"
    )
    list(APPEND SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_OUTPUTS})
add_dependencies(${PROJECT_NAME} shaders)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SPIRV_DIR}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
)
