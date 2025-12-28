//
// Created by 권용훈 on 8/2/24.
//

#ifndef AIRUNNER_UTILS_H
#define AIRUNNER_UTILS_H

#include <string>
#include <onnxruntime_cxx_api.h>

class Utils {
public:
    static std::string onnxTypeToString(ONNXTensorElementDataType type);
};

#endif //AIRUNNER_UTILS_H
