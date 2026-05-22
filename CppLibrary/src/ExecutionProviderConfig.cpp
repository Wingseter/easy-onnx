#include "../Utils/pch.h"
#include "../include/ExecutionProviderConfig.h"
#include "../include/Logger.h"

#include <onnxruntime_c_api.h>

#if defined(__APPLE__)
    #include <coreml_provider_factory.h>
#elif defined(_WIN32)
    #include <dml_provider_factory.h>
#endif

bool TryAppendAcceleratedEP(Ort::SessionOptions& opts, int device_id) {
#if defined(__APPLE__)
    (void)device_id; // CoreML has no multi-device selection.
    try {
        OrtStatus* status =
            OrtSessionOptionsAppendExecutionProvider_CoreML(opts, /*flags=*/0);
        if (status != nullptr) {
            const char* msg = Ort::GetApi().GetErrorMessage(status);
            std::string err = msg ? msg : "(unknown)";
            Ort::GetApi().ReleaseStatus(status);
            Logger::getInstance().error(
                std::string("CoreML EP append failed, falling back to CPU: ") + err);
            return false;
        }
        Logger::getInstance().info("EP appended: CoreML");
        return true;
    } catch (const Ort::Exception& e) {
        Logger::getInstance().error(
            std::string("CoreML EP exception, falling back to CPU: ") + e.what());
        return false;
    }
#elif defined(_WIN32)
    try {
        OrtStatus* status =
            OrtSessionOptionsAppendExecutionProvider_DML(opts, device_id);
        if (status != nullptr) {
            const char* msg = Ort::GetApi().GetErrorMessage(status);
            std::string err = msg ? msg : "(unknown)";
            Ort::GetApi().ReleaseStatus(status);
            Logger::getInstance().error(
                std::string("DirectML EP append failed, falling back to CPU: ") + err);
            return false;
        }
        Logger::getInstance().info(
            std::string("EP appended: DirectML (device_id=") + std::to_string(device_id) + ")");
        return true;
    } catch (const Ort::Exception& e) {
        Logger::getInstance().error(
            std::string("DirectML EP exception, falling back to CPU: ") + e.what());
        return false;
    }
#else
    (void)opts;
    (void)device_id;
    Logger::getInstance().info("No accelerated EP available on this platform; using CPU.");
    return false;
#endif
}
