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



