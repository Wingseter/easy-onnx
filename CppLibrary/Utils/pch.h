//
// Created by 권용훈 on 7/11/24.
//

#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <onnxruntime_cxx_api.h>

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



