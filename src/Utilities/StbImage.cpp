#define STB_IMAGE_IMPLEMENTATION

// Disable SSE2 instructions because Circle doesn't support them yet.
#define STBI_NO_SIMD

#include "StbImage.hpp"
