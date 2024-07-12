//
// Created by 권용훈 on 7/11/24.
//

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <thread>
using namespace std;

#ifdef PLATFORM_WINDOWS
#pragma comment(lib, "onnxruntime.lib")
#endif

enum class DataType {
    FLOAT,
    DOUBLE,
    INT32,
    INT64
};

union DataPointer {
    float* float_ptr;
    double* double_ptr;
    int32_t* int32_ptr;
    int64_t* int64_ptr;
};


#include <onnxruntime_cxx_api.h>
#include <coreml_provider_factory.h>

#define DECLARE_SINGLE(type)		\
private:							\
	type() {}						\
	~type() {}						\
public:								\
	static type* GetInstance()		\
	{								\
		static type instance;		\
		return &instance;			\
	}								\

#define GET_SINGLE(type)	type::GetInstance()



