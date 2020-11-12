#pragma once
#include <cmath>
#include "RayTracingPipeline.hpp"

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


// Generates a seed for a random number generator from 2 inputs plus a backoff
// https://github.com/nvpro-samples/optix_prime_baking/blob/master/random.h
// https://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
inline uint InitRandomSeed(uint val0, uint val1) {
  uint v0 = val0, v1 = val1, s0 = 0;

  @meta for(int n = 0; n < 16; ++n) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

inline uint InitRandomSeed(uvec3 v) {
  return InitRandomSeed(InitRandomSeed(v.x, v.y), v.z);
}

inline uint RandomInt(uint& seed) {
  // LCG values from Numerical Recipes
  return (seed = 1664525 * seed + 1013904223); 
}

inline float RandomFloat(uint& seed) {
  // Float version using bitmask from Numerical Recipes
  const uint one = 0x3f800000;
  const uint msk = 0x007fffff;
  uint x = one | (msk & (RandomInt(seed) >> 9));
  return *(const float*)&x - 1;
}

inline vec3 RandomOnUnitSphere(uint& seed) {
  float theta = 2 * M_PIf32 * RandomFloat(seed);
  float y = 2 * RandomFloat(seed) - 1;
  float zx = sqrt(1 - y * y);
  return vec3(zx * cos(theta), y, zx * sin(theta));
}

inline vec3 RandomInUnitSphere(uint& seed) {
  float r = sqrt(RandomFloat(seed));
  return r * RandomOnUnitSphere(seed);
}


inline vec2 RandomInUnitDisk(uint& seed) {
  // Get a random angle and magnitude.
  float a = 2 * M_PIf32 * RandomFloat(seed);
  float r = sqrt(RandomFloat(seed));

  return vec2(r * cos(a), r * sin(a));
}

enum binding_t {
  binding_acceleration = 0,
  binding_accumulation = 1,
  binding_output       = 2,
  binding_ubo          = 3,
  binding_vertices     = 4,
  binding_indices      = 5,
  binding_materials    = 6,
  binding_offsets      = 7,
  binding_samplers     = 8,
  binding_spheres      = 9,
};

// binding 0
[[using spirv: uniform, binding(binding_acceleration)]]
accelerationStructure AS;

// binding 1
[[using spirv: uniform, binding(binding_accumulation), format(rgba32f)]]
image2D AccumulationImage;

// binding 2
[[using spirv: uniform, binding(binding_output), format(rgba8)]]
image2D OutputImage;

[[using spirv: uniform, binding(binding_ubo)]]
UniformBufferObject ubo;

struct vertex_t {
  vec3 pos;
  vec3 normal;
  float u, v;
  int material;
};

// binding 4
[[using spirv: buffer, readonly, binding(binding_vertices)]]
vertex_t Vertices[];

// binding 5
[[using spirv: buffer, readonly, binding(binding_indices)]]
int Indices[];

// binding 6
[[using spirv: buffer, readonly, binding(binding_materials)]]
material_t Materials[];

// binding 7
[[using spirv: buffer, readonly, binding(binding_offsets)]]
uvec2 Offsets[];
 
// binding 8
[[using spirv: uniform, binding(binding_samplers)]]
sampler2D TextureSamplers[];

// binding 9
[[using spirv: buffer, readonly, binding(binding_spheres)]]
vec4 Spheres[];


////////////////////////////////////////////////////////////////////////////////

struct RayPayload {
  vec4 ColorAndDistance; // rgb + t
  vec4 ScatterDirection; // xyz + w (is scatter needed)
  uint RandomSeed;
};

[[using spirv: rayPayload, location(0)]]
RayPayload rayPayload;

[[using spirv: rayPayloadIn, location(0)]]
RayPayload rayPayloadIn;

[[spirv::rgen]]
void rgen_main() {
  rayPayload.RandomSeed = InitRandomSeed(
    uvec3(glray_LaunchID.xy, ubo.TotalNumberOfSamples)
  );

  vec3 pixel_color { };

  uint pixelRandomSeed = ubo.RandomSeed;
  for(int s = 0; s < ubo.NumberOfSamples; ++s) {
    vec2 pixel(
      glray_LaunchID.x + RandomFloat(pixelRandomSeed),
      glray_LaunchID.y + RandomFloat(pixelRandomSeed)
    );
    vec2 uv = 2 * (pixel / (vec2)glray_LaunchSize.xy) - 1;

    vec2 offset = ubo.Aperture / 2 * RandomInUnitDisk(rayPayload.RandomSeed);
    vec4 origin = ubo.ModelViewInverse * vec4(offset, 0, 1);
    vec4 target = ubo.ProjectionInverse * vec4(uv, 1, 1);
    vec4 dir = ubo.ModelViewInverse * 
      vec4(normalize(target.xyz * ubo.FocusDistance - vec3(offset, 0)), 0);

    vec3 ray_color = 1;

    for(int b = 0; b <= ubo.NumberOfBounces; ++b) {
      const float tMin = .001;
      const float tMax = 10000;

      if(b == ubo.NumberOfBounces) {
        ray_color = 0;
        break;
      }

      // Intersect the ray against the acceleration structure.
      glray_Trace(AS, glray_FlagsOpaque, 0xff, 0, 0, 0, origin.xyz, tMin, 
        dir.xyz, tMax, 0);

      vec3 hit_color = rayPayload.ColorAndDistance.rgb;
      float t = rayPayload.ColorAndDistance.w;
      bool is_scattered = rayPayload.ScatterDirection.w > 0;

      ray_color *= hit_color;

      if(t < 0 || !is_scattered)
        break;

      // Trace hit.
      origin += t * dir;
      dir.xyz = rayPayload.ScatterDirection.xyz;
    }

    pixel_color += ray_color;
  }

  vec3 accumulated = pixel_color;
  if(ubo.NumberOfSamples != ubo.TotalNumberOfSamples)
    accumulated += imageLoad(AccumulationImage, ivec2(glray_LaunchID.xy)).rgb;

  pixel_color = accumulated / ubo.TotalNumberOfSamples;

  if(ubo.GammaCorrection)
    pixel_color = sqrt(pixel_color);

  imageStore(AccumulationImage, ivec2(glray_LaunchID.xy), vec4(accumulated, 0));
  imageStore(OutputImage, ivec2(glray_LaunchID.xy), vec4(pixel_color, 0));
}

////////////////////////////////////////////////////////////////////////////////
// Miss shader.

[[spirv::rmiss]]
void rmiss_main() {
   if(ubo.HasSky) {
    // Sky color.

    // NOTE: Do we have to normalize this? 
    float t = .5f * (normalize(glray_WorldRayDirection).y + 1);
    vec3 sky_color = mix(vec3(1), vec3(.5, .7, 1), t);
    rayPayloadIn.ColorAndDistance = vec4(sky_color, -1);

   } else {
    rayPayloadIn.ColorAndDistance = vec4(0, 0, 0, -1);
   }
}

////////////////////////////////////////////////////////////////////////////////
// Scattering for closest hit shaders.

float Schlick(float cosine, float refraction) {
  float r0 = (1 - refraction) / (1 + refraction);
  r0 *= r0;
  return r0 + (1 - r0) * pow(1 - cosine, 5.f);
}

// Functions accessing SPIRV declarations must be marked inline.
inline RayPayload ScatterLambertian(material_t m, vec3 dir, vec3 normal, 
  vec2 texcoord, float t, uint& seed) {

  bool is_scattered = dot(dir, normal) < 0;

  vec4 texColor = m.DiffuseTextureId >= 0 ? 
    textureLod(TextureSamplers[m.DiffuseTextureId], texcoord, 0) :
    1;

  vec4 colorAndDistance = vec4(m.Diffuse.rgb * texColor.rgb, t);
  vec4 scatter = vec4(normal + RandomInUnitSphere(seed), is_scattered);

  return RayPayload {
    colorAndDistance,
    scatter,
    seed
  };
}

inline RayPayload ScatterMetallic(material_t m, vec3 dir, vec3 normal,
  vec2 texcoord, float t, uint& seed) {

  vec3 reflected = reflect(dir, normal);
  bool is_scattered = dot(reflected, normal) > 0;

  vec3 texColor = m.DiffuseTextureId >= 0 ? 
    textureLod(TextureSamplers[m.DiffuseTextureId], texcoord, 0).rgb :
    1;

  vec4 colorAndDistance = is_scattered ? 
    vec4(m.Diffuse.rgb * texColor, t) :
    vec4(1, 1, 1, -1);

  vec4 scatter = vec4(
    reflected + m.Fuzziness * RandomInUnitSphere(seed), 
    is_scattered
  );

  return RayPayload {
    colorAndDistance,
    scatter,
    seed
  };
}

inline RayPayload ScatterDielectric(material_t m, vec3 dir, vec3 normal,
  vec2 texcoord, float t, uint& seed) {

  float d = dot(dir, normal); 
  vec3 outward_normal = d > 0 ? -normal : normal;
  float niOverNt = d > 0 ? m.RefractionIndex : 1 / m.RefractionIndex;
  float cosine = d > 0 ? m.RefractionIndex * d : -d;

  vec3 refracted = refract(dir, outward_normal, niOverNt);
  float prob = (refracted.x || refracted.y || refracted.z) ? 
    Schlick(cosine, m.RefractionIndex) : 1;

  vec3 tex_color = m.DiffuseTextureId >= 0 ? 
    textureLod(TextureSamplers[m.DiffuseTextureId], texcoord, 0).rgb :
    1;

  vec3 scatter = RandomFloat(seed) < prob ? reflect(dir, normal) : refracted;

  return RayPayload {
    vec4(tex_color, t),
    vec4(scatter, 1),
    seed
  };
}

inline RayPayload ScatterDiffuse(material_t m, float t, uint& seed) {
  vec4 colorAndDistance = vec4(m.Diffuse.rgb, t);
  vec4 scatter = vec4(1, 0, 0, 0);

  return RayPayload { colorAndDistance, scatter, seed };
}

inline RayPayload Scatter(material_t m, vec3 dir, vec3 normal, vec2 texcoord,
  float t, uint& seed) {

  RayPayload rayPayload { };
  switch(m.material_model) {
    case material_model_t::lambertian:
      rayPayload = ScatterLambertian(m, dir, normal, texcoord, t, seed);
      break;

    case material_model_t::metallic:
      rayPayload = ScatterMetallic(m, dir, normal, texcoord, t, seed);
      break;

    case material_model_t::dielectric:
      rayPayload = ScatterDielectric(m, dir, normal, texcoord, t, seed);
      break;

    case material_model_t::isotropic:
      // Not implemented. Use Diffuse scattering.

    case material_model_t::diffuse:
      rayPayload = ScatterDiffuse(m, t, seed);
      break;

    default:
      break;
  }

  return rayPayload;
}


////////////////////////////////////////////////////////////////////////////////
// Process closest hit for a triangle. The any-hit functionality is provided
// by hardware.

[[spirv::hitAttribute]]
vec2 TriangleHit;

[[spirv::rchit]]
void rchit_triangle() {
  // Vertex data is stored in consecutive locatoins.
  uvec2 offset = Offsets[glray_InstanceCustomIndex];
  uint index = offset.x + 3 * glray_PrimitiveID;
  vertex_t v0 = Vertices[Indices[index + 0] + offset.y];
  vertex_t v1 = Vertices[Indices[index + 1] + offset.y];
  vertex_t v2 = Vertices[Indices[index + 2] + offset.y];

  material_t material = Materials[v0.material];

  vec3 barycentric(1 - TriangleHit.x - TriangleHit.y, TriangleHit);
  vec3 normal = normalize(mat3(v0.normal, v1.normal, v2.normal) * barycentric);
  vec2 texcoord = mat3x2(v0.u, v0.v, v1.u, v1.v, v2.u, v2.v) * barycentric;

  rayPayloadIn = Scatter(material, glray_WorldRayDirection, normal, 
    texcoord, glray_HitT, rayPayloadIn.RandomSeed);
}

////////////////////////////////////////////////////////////////////////////////
// Closest hit and any hit for spheres.

vec2 GetSphereCoord(vec3 point) {
  float phi = atan2(point.x, point.z);
  float theta = asin(point.y);

  const float pi = M_PIf32;

  return vec2(
    (phi + pi) / (2 * pi),
    1 - (theta + pi / 2) / pi
  );
}

[[spirv::rchit]]
void rchit_sphere() {
  // Find vertex data for this procedural item.
  uvec2 offset = Offsets[glray_InstanceCustomIndex];
  int index = Indices[offset.x] + offset.y;
  vertex_t v0 = Vertices[index];
  material_t material = Materials[v0.material];

  vec4 sphere = Spheres[glray_InstanceCustomIndex];
  vec3 center = sphere.xyz;
  float radius = sphere.w;

  // Get the point of intersection.
  vec3 point = glray_WorldRayOrigin + glray_HitT * glray_WorldRayDirection;
  vec3 normal = (point - center) / radius;
  vec2 texcoord = GetSphereCoord(normal);

  rayPayloadIn = Scatter(material, normalize(glray_WorldRayDirection), normal, 
    texcoord, glray_HitT, rayPayloadIn.RandomSeed);
}

[[spirv::hitAttribute]]
vec4 SphereHit;

[[spirv::rint]]
void rint_sphere() {
  vec4 sphere = Spheres[glray_InstanceCustomIndex];
  vec3 center = sphere.xyz;
  float radius = sphere.w;

  vec3 origin = glray_WorldRayOrigin;
  vec3 dir = glray_WorldRayDirection;
  float tMin = glray_Tmin;
  float tMax = glray_Tmax;

  vec3 oc = origin - center;
  float a = dot(dir, dir);
  float b = dot(oc, dir);
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - a * c;

  float t1 = (-b - sqrt(discriminant)) / a;
  float t2 = (-b + sqrt(discriminant)) / a;

  bool b1 = tMin <= t1 && t1 < tMax;
  bool b2 = tMin <= t2 && t2 < tMax;
  if(discriminant >= 0 && (b1 || b2)) {
    SphereHit = sphere;
    glray_reportIntersection(b1 ? t1 : t2, 0);
  }
}

////////////////////////////////////////////////////////////////////////////////

namespace Vulkan::RayTracing {
  ShadersBinary RayTracingPipeline::GetShaders() const {
    ShadersBinary shaders { };
    shaders.module_data = __spirv_data;
    shaders.module_size = __spirv_size;

    shaders.rgen           = @spirv(rgen_main);
    shaders.rmiss          = @spirv(rmiss_main);
    shaders.rchit_triangle = @spirv(rchit_triangle);
    shaders.rchit_sphere   = @spirv(rchit_sphere);
    shaders.rint_sphere    = @spirv(rint_sphere);

    return shaders;
  }
}
