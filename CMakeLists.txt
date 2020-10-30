cmake_minimum_required(VERSION 3.10)

project(RayTracingInVulkan)
# add_link_options("-no-demangle")

include_directories(src src/imgui)

SET(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS exception program_options) 
find_package(glfw3 REQUIRED)
find_package(glm CONFIG REQUIRED)
# find_package(imgui CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(Vulkan REQUIRED)

set(SOURCE_FILES
  src/main.cpp
  src/ModelViewController.cpp
  src/Options.cpp
  src/RayTracer.cpp
  src/SceneList.cpp
  src/UserInterface.cpp

  src/Assets/CornellBox.cpp
  src/Assets/Model.cpp
  src/Assets/Scene.cpp
  src/Assets/Texture.cpp
  src/Assets/TextureImage.cpp
  src/Assets/UniformBuffer.cpp

  src/Utilities/Console.cpp
  src/Utilities/StbImage.cpp

  src/Vulkan/Application.cpp
  src/Vulkan/Buffer.cpp
  src/Vulkan/CommandBuffers.cpp
  src/Vulkan/CommandPool.cpp
  src/Vulkan/DebugUtilsMessenger.cpp
  src/Vulkan/DepthBuffer.cpp
  src/Vulkan/DescriptorPool.cpp
  src/Vulkan/DescriptorSetLayout.cpp
  src/Vulkan/DescriptorSetManager.cpp
  src/Vulkan/DescriptorSets.cpp
  src/Vulkan/Device.cpp
  src/Vulkan/DeviceMemory.cpp
  src/Vulkan/Fence.cpp
  src/Vulkan/FrameBuffer.cpp
  src/Vulkan/GraphicsPipeline.cpp
  src/Vulkan/Image.cpp
  src/Vulkan/ImageView.cpp
  src/Vulkan/Instance.cpp
  src/Vulkan/PipelineLayout.cpp
  src/Vulkan/RenderPass.cpp
  src/Vulkan/Sampler.cpp
  src/Vulkan/Semaphore.cpp
  src/Vulkan/ShaderModule.cpp
  src/Vulkan/Shaders.cpp
  src/Vulkan/Strings.cpp
  src/Vulkan/Surface.cpp
  src/Vulkan/SwapChain.cpp
  src/Vulkan/Vulkan.cpp
  src/Vulkan/Window.cpp

  src/Vulkan/RayTracing/AccelerationStructure.cpp
  src/Vulkan/RayTracing/Application.cpp
  src/Vulkan/RayTracing/BottomLevelAccelerationStructure.cpp
  src/Vulkan/RayTracing/DeviceProcedures.cpp
  src/Vulkan/RayTracing/RayTracingPipeline.cpp
  src/Vulkan/RayTracing/RayTracingProperties.cpp
  src/Vulkan/RayTracing/ShaderBindingTable.cpp
  src/Vulkan/RayTracing/Shaders.cpp
  src/Vulkan/RayTracing/TopLevelAccelerationStructure.cpp

  src/imgui/imgui.cpp
  src/imgui/imgui_draw.cpp
  src/imgui/imgui_widgets.cpp
  src/imgui/backends/imgui_impl_glfw.cpp
  src/imgui/backends/imgui_impl_vulkan.cpp

)

set_source_files_properties(
  src/Vulkan/Shaders.cpp 
  src/Vulkan/RayTracing/Shaders.cpp 
  PROPERTIES COMPILE_FLAGS -shader
)

add_executable(raytracer ${SOURCE_FILES})

target_link_libraries(raytracer 
  dl
  glfw
  vulkan
  boost_program_options
  tinyobjloader
  stdc++fs
)