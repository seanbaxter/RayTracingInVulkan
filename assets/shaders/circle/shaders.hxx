#pragma once
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
