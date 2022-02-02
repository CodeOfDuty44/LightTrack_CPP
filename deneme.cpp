#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include "NvOnnxParser.h"
#include "utils.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace nvonnxparser;

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        //suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;


int main(){

    //load image, create a new image by multiplying 2 and viusalize
    //A: brighter, new image
    std::string image_path = "lena2.png";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::imshow("Deneme image", img);
    cv::Mat other = img * 2;
    cv::imshow("*2", other);
    int k = cv::waitKey(100000);

    // print image and size
    std::cout << "Mat is: " << std::endl << img << std::endl;
    std::cout << "Size is " << img.size[0] << std::endl;

    //create Mat object by giving size and default value
    img = cv::Mat(300,500, CV_8UC3, cv::Scalar(23,44,97));

    //cannot print InputOutputArray
    // cv::InputOutputArray ioa = cv::InputOutputArray(300,500, CV_8UC3, cv::Scalar(23,44,97));
    // std::cout << ioa << std::endl;

    //cv::InputOutputArray can be constructed from cv::Mat object
    cv::InputOutputArray ioa = cv::InputOutputArray(img);

    //crop image and check if shallow copy
    // A: yes it does shallow copy
    cv::Mat img_cropped = img(cv::Range(0,3), cv::Range(0,4));
    img_cropped.at<cv::Vec3b>(1,0) = cv::Vec3b(255,255,255);
    std::cout << "img_cropped: " << img_cropped << std::endl;
    std::cout << "img pixel : " << img.at<cv::Vec3b>(1,0) << std::endl;

    //crop image using deep copy and change pixel value and confirm orig image is not modified
    //A: clone() does deep copy
    img = cv::imread(image_path, cv::IMREAD_COLOR); //read again as we changed it before
    img_cropped = img(cv::Range(0,3), cv::Range(0,4)).clone();
    img_cropped.at<cv::Vec3b>(1,0) = cv::Vec3b(255,255,255);
    std::cout <<  "orig image pixel is (shouldnt change since deep copy)" << img.at<cv::Vec3b>(1,0) << std::endl;
    std::cout <<  "cropped image pixel is)" << img_cropped.at<cv::Vec3b>(1,0) << std::endl;


    //Build TensorRt engine from onnx
    IBuilder* builder = createInferBuilder(logger);

    uint32_t flag = 1U <<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flag);

    IParser* parser = createParser(*network, logger);
    const char* onnx_path = "LightTrack_Backbone_z.onnx";
    parser->parseFromFile(onnx_path, 0);
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
        std::cout << "ahmet :)";
    }

    IBuilderConfig* config = builder->createBuilderConfig();
    //IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config); //attribute error, can be bcz of tensporrt version 7 vs 8
    //ICudaEngine* serkanengine = builder->buildCudaEngine(*network); //works fine 
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "All fine" << std::endl;

    //Save the built engine 
    IHostMemory *serializedModel = engine->serialize();
    std::ofstream p("LightTrack_Backbone_z.trt", std::ios::binary);
    p.write((const char*)serializedModel->data(),serializedModel->size());
    p.close();
    std::cout << "seems okay" << std::endl; 

    //Load engine  
    IRuntime* runtime = createInferRuntime(logger);
    std::string buffer = readEngine("LightTrack_Backbone_z.trt");
    ICudaEngine* loadedengine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);

    int32_t templateIndex = loadedengine->getBindingIndex("z");
    int32_t templateFeatureIndex = loadedengine->getBindingIndex("zf");
    
    std::cout << "templateIndex: " << templateIndex << std::endl;
    std::cout << "templateFeatureIndex: " << templateFeatureIndex << std::endl; 

    //Run engine
    IExecutionContext *context = engine->createExecutionContext();
    void* buffers[2];
    const int batchSize = 1;
    const int height = 127;
    const int width = 127;
    const int channels = 3;
    img = cv::Mat(127,127, CV_8UC3, cv::Scalar(24,44,97));

    cudaMalloc( &buffers[templateIndex], batchSize*channels*height*width*sizeof(int)); // ? size of float or int
    cudaMalloc(&buffers[templateFeatureIndex], batchSize*96*8*8*sizeof(float));

    float* input = new float[height*width*channels*batchSize];
    float* output = new float[96*8*8];
    uint8_t * rawpgm = new uint8_t[height*width*channels];
    uint8_t * rawpgm_chw = new uint8_t[height*width*channels];
    std::memcpy(rawpgm, img.data, height*width*channels);
    HWC_to_CHW( rawpgm, rawpgm_chw, height, width, channels );
    std::cout << "Converted HWC TO CHW!\n";
        for( int i=0; i<height*width*channels; i++ )
        {
            input[height*width*channels*0 + i] = float(rawpgm_chw[i]) ; 
        }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::cout << "cudaMemcpyHostToDevice ASYNC\n";
    cudaMemcpyAsync( buffers[templateIndex], input, batchSize*height*width*channels*sizeof(float), cudaMemcpyHostToDevice, stream );
    context->enqueue(batchSize, buffers, stream, nullptr);
    std::cout << "engine run successful" << std::endl; 
    cudaMemcpy( output, buffers[templateFeatureIndex], batchSize*96*8*8*sizeof(float), cudaMemcpyDeviceToHost );

    for(int i = 0; i < 96*8*8; i++)
    {
        std::cout << output[i] << std::endl;
    }
    

    return 0;
}
