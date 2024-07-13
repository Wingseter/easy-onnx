//
// Created by 권용훈 on 7/10/24.
//

#ifndef AIRUNNER_WORKFLOW_H
#define AIRUNNER_WORKFLOW_H

class Model;
class DataLoader;

class Workflow {
public:
    void init_model(const char* modelPath, bool cpu_use);
    void run_model(int* data, size_t num_elements);
    void run_model(float* data, size_t num_elements);
    void run_model(double* data, size_t num_elements);

private:
    template <typename T>
    void run_inference(T* data, size_t num_elements);

    std::shared_ptr<Model> model_;
    std::shared_ptr<DataLoader> data_loader_;
};

#endif //AIRUNNER_WORKFLOW_H
