#ifndef AIRUNNER_EXECUTION_PROVIDER_CONFIG_H
#define AIRUNNER_EXECUTION_PROVIDER_CONFIG_H

#include <onnxruntime_cxx_api.h>

// Append the best available platform-specific execution provider to the given
// session options. Returns true on success, false on failure or if no provider
// is available for this platform. The caller should continue with CPU as a
// safe fallback when this returns false.
//
// Behavior:
//   - macOS : appends CoreML EP (flags = 0 — let CoreML decide ANE/GPU/CPU)
//   - Win32 : appends DirectML EP (device 0)
//   - other : no-op, returns false
bool TryAppendAcceleratedEP(Ort::SessionOptions& opts);

#endif // AIRUNNER_EXECUTION_PROVIDER_CONFIG_H
