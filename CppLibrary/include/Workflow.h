#ifndef AIRUNNER_WORKFLOW_H
#define AIRUNNER_WORKFLOW_H

class Model;
class DataLoader;

class Workflow {
public:
    void init_model(const char* modelPath, bool cpu_use);

    void run_model(int* data, int num_elements);
    void run_model(float* data, int num_elements);
    void run_model(double* data, int num_elements);
    std::vector<float> getFlattenedOutput() const;
    std::vector<int64_t> getOriginalShape() const;
    void run_test(const char* modelPath, bool cpu_use, float* data, int num_elements);

private:
    template <typename T>
    void run_inference(T* data, int num_elements);

    std::string previous = "";
    std::shared_ptr<Model> model_ = nullptr;
    std::shared_ptr<DataLoader> data_loader_ = nullptr;
};

#endif //AIRUNNER_WORKFLOW_H
