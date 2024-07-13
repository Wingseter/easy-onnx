//
// Created by 권용훈 on 7/10/24.
//
#pragma once

extern "C" const char* GetResponse(const char* input);

extern "C" const bool InitModel(const char * modelPath, bool cpu_use);