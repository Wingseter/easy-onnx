#ifndef AIRUNNER_WORKFLOW_H
#define AIRUNNER_WORKFLOW_H

class Model;
class DataLoader;
#include <memory>

class Workflow {
public:
    // 생성자에서 ONNX Runtime 환경 및 세션 초기화
    Workflow(const char* modelPath, bool cpu_use)
    {
        // 세션을 생성하고 모델을 로드
        Ort::SessionOptions session_options{nullptr};
        auto envLocal = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
        env = std::move(envLocal);
        auto sessionLocal = std::make_unique<Ort::Session>(*env, modelPath, session_options);
        session = std::move(sessionLocal);
        std::cout << "what?" << std::endl;

    }

    void init_model(const char* modelPath, bool cpu_use);
    void run_model(float* data, int num_elements);
    std::vector<float> getFlattenedOutput() const;
    std::vector<int64_t> getOriginalShape() const;
    void run_test(const char* modelPath, bool cpu_use, float* data, int num_elements);

private:
    void run_inference(float* data, int num_elements);

    // ONNX Runtime 환경 및 세션 관련 멤버 변수
    std::unique_ptr<Ort::Session> session= nullptr;
    std::unique_ptr < Ort::Env> env = nullptr;
};

#endif //AIRUNNER_WORKFLOW_H
