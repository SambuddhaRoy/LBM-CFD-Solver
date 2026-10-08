cmake_minimum_required(VERSION 3.24)
project(WindTunnel VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8 /wd4324)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)   # std::getenv is standard C++
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(Stb REQUIRED)

add_executable(WindTunnel
    src/main.cpp
    src/vk.cpp
    src/solver.cpp
    src/geometry.cpp
    src/render.cpp
    src/app.cpp
    src/tests.cpp
    src/validate.cpp
    src/fluid.cpp
)
target_link_libraries(WindTunnel PRIVATE
    Vulkan::Vulkan glfw imgui::imgui assimp::assimp glm::glm
    vk-bootstrap::vk-bootstrap GPUOpen::VulkanMemoryAllocator)
target_include_directories(WindTunnel PRIVATE ${Stb_INCLUDE_DIR})
if(WIN32)
    target_link_libraries(WindTunnel PRIVATE Comdlg32)
endif()

# ─── Shaders ────────────────────────────────────────────────────────────────
# Every kernel that touches the distribution functions is built once per
# storage precision (0 = FP32, 1 = FP16S, 2 = FP16C); the step kernel is also
# built with and without the in-kernel momentum-exchange force reduction.
if(Vulkan_GLSLC_EXECUTABLE)
    set(GLSLC ${Vulkan_GLSLC_EXECUTABLE})
else()
    find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
endif()
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found: install the Vulkan SDK or vcpkg shaderc")
endif()

set(SHADER_SRC ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUT ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT})
file(GLOB SHADER_DEPS ${SHADER_SRC}/*.glsl)

function(add_shader SRC OUT)
    add_custom_command(
        OUTPUT  ${SHADER_OUT}/${OUT}.spv
        COMMAND ${GLSLC} --target-env=vulkan1.3 -O ${ARGN}
                ${SHADER_SRC}/${SRC} -o ${SHADER_OUT}/${OUT}.spv
        DEPENDS ${SHADER_SRC}/${SRC} ${SHADER_DEPS}
        COMMENT "glslc ${OUT}")
    set_property(GLOBAL APPEND PROPERTY SPV_LIST ${SHADER_OUT}/${OUT}.spv)
endfunction()

foreach(P 0 1 2)
    add_shader(step.comp     step_p${P}       -DSTORE=${P} -DFORCES=0)
    add_shader(step.comp     step_force_p${P} -DSTORE=${P} -DFORCES=1)
    add_shader(init.comp     init_p${P}       -DSTORE=${P})
    add_shader(classify.comp classify_p${P}   -DSTORE=${P})
    add_shader(probe.comp    probe_p${P}      -DSTORE=${P})
endforeach()
add_shader(wallmask.comp   wallmask)
add_shader(reduce.comp     reduce)
add_shader(stats.comp      stats)
add_shader(sdf_shape.comp  sdf_shape)
add_shader(frontal.comp    frontal)
add_shader(sdf_mesh.comp   sdf_mesh)
add_shader(sdf_sign.comp   sdf_sign)
add_shader(render.comp     render)

get_property(SPV GLOBAL PROPERTY SPV_LIST)
add_custom_target(shaders ALL DEPENDS ${SPV})
add_dependencies(WindTunnel shaders)
add_custom_command(TARGET WindTunnel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADER_OUT}
            $<TARGET_FILE_DIR:WindTunnel>/shaders)
