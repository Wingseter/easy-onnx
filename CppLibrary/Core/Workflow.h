//
// Created by 권용훈 on 7/10/24.
//

#ifndef AIRUNNER_WORKFLOW_H
#define AIRUNNER_WORKFLOW_H

class Model;
class DataLoader;

class Workflow {
public:
    void set_model(const char* modelPath);

private:
    shared_ptr<Model> _model;
    shared_ptr<DataLoader> _dataLoader;

};


#endif //AIRUNNER_WORKFLOW_H
