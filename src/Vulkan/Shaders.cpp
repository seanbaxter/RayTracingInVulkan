#include "GraphicsPipeline.hpp"
#include <cmath>

// binding 3
struct UniformBufferObject {
  mat4 ModelView;
  mat4 Projection;
  mat4 ModelViewInverse;
  mat4 ProjectionInverse;
  float Aperture;
  float FocusDistance;
  uint TotalNumberOfSamples;
  uint NumberOfSamples;
  uint NumberOfBounces;
  uint RandomSeed;
  uint GammaCorrection;
  uint HasSky;
};

enum class material_model_t : uint {
  lambertian,
  metallic,
  dielectric,
  isotropic,
  diffuse
};

struct material_t {
  vec4 Diffuse;
  int DiffuseTextureId;
  float Fuzziness;
  float RefractionIndex;
  material_model_t material_model;
};

// Common utilities.
template<auto index, typename type_t = @enum_type(index)> 
[[spirv::in((size_t)index)]]
type_t shader_in;

template<auto index, typename type_t = @enum_type(index)> 
[[spirv::out((size_t)index)]]
type_t shader_out;

// Binding slots
enum binding_t {
  binding_ubo,
  binding_materials,
  binding_samplers,
};

// binding 0
[[spirv::uniform(binding_ubo)]]
UniformBufferObject ubo;

// binding 1
[[using spirv: buffer(binding_materials), readonly]]
material_t Materials[];

// binding 2
[[using spirv: uniform(binding_samplers)]]
sampler2D TextureSamplers[];

// Declare the name, location and type for each shader attribute.
enum typename vattrib_t {
  vattrib_position       = vec3,
  vattrib_normal         = vec3,
  vattrib_texcoord       = vec2,
  vattrib_material       = int,
  vattrib_diffuse        = vec3
};

[[spirv::vert]]
void vert_main() {
  vec3 pos = shader_in<vattrib_position>;
  vec3 normal = shader_in<vattrib_normal>;

  // Transform the position.
  glvert_Output.Position = ubo.Projection * ubo.ModelView * vec4(pos, 1);

  // Load the material.
  material_t material = Materials[shader_in<vattrib_material>];

  // Pass through the diffuse color.
  shader_out<vattrib_diffuse> = material.Diffuse.rgb;

  // Compute normal vector.
  shader_out<vattrib_normal> = (ubo.ModelView * vec4(normal, 0)).xyz;

  // Pass the texcoord through.
  shader_out<vattrib_texcoord> = shader_in<vattrib_texcoord>;

  // Pass the material index through.
  shader_out<vattrib_material> = shader_in<vattrib_material>;
}

[[spirv::frag]]
void frag_main() {
  int texid = Materials[shader_in<vattrib_material>].DiffuseTextureId;
  vec3 light_vector = normalize(vec3(5, 4, 3));

  float d = max(dot(light_vector, normalize(shader_in<vattrib_normal>)), .2f);
  vec3 c = shader_in<vattrib_diffuse> * d;

  if(texid >= 0)
    c *= texture(TextureSamplers[texid], shader_in<vattrib_texcoord>).rgb;

  shader_out<0, vec4> = vec4(c, 1);
}

namespace Vulkan {
  ShadersBinary GraphicsPipeline::GetShaders() const {
    ShadersBinary shaders { };
    shaders.module_data = __spirv_data;
    shaders.module_size = __spirv_size;
    shaders.vert = @spirv(vert_main);
    shaders.frag = @spirv(frag_main);
    return shaders;
  }
}
