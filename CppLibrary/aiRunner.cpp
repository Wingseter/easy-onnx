//
// Created by 권용훈 on 7/10/24.
//

#include "Utils/pch.h"
#include "aiRunner.h"
#include <string.h>

extern "C" const char* GetResponse(const char* input) {
    if (strcmp(input, "hello") == 0) {
        return "world";
    }
    return "unknown";
}