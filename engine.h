#pragma once
#include <string>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime.h>

using namespace std;
using namespace nvinfer1;

namespace dawnbench {

class Engine {
public:
    // Create engine from engine path
    Engine(const string &engine_path, bool verbose=false);

    ~Engine();

    // Save model to path
    void save(const string &path);

    // Infer using pre-allocated GPU buffers {data, scores, boxes, classes}
    float* infer();

    // Get (h, w) size of the fixed input
    vector<int> getInputSize();

    // Get max allowed batch size
    int getMaxBatchSize();

    // Get size of output
    vector<int> getOutputSize();

    // Get stride
    int getStride();

    int getBindingIndex(const char* name);

    const char* getBindingName(int index);

    void loadImage(vector<float> image);

private:
    IRuntime *_runtime = nullptr;
    ICudaEngine *_engine = nullptr;
    IExecutionContext *_context = nullptr;
    cudaStream_t _stream = nullptr;
    void *_input_d = nullptr;
    void *_output_d = nullptr;
    float *_output = nullptr;
    size_t _output_size = 0;
    vector<void *> _buffers; 
    void _load(const string &path);
    void _prepare();

};

}

