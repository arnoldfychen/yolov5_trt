#include <iostream>
#include <experimental/filesystem>
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "NvInferPlugin.h"
//#include "NvInfer.h"
// #include "trt_utils.h"
#include "nvdsinfer_context.h"
#include "nvdsinfer_custom_impl.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id

#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
static int CLASS_NUM = Yolo::CLASS_NUM;


// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
// we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1; 
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8) {
    //return math.ceil(x / divisor) * divisor
    if (int(x * gw) % divisor == 0) {
        return int(x * gw);
    }
    return (int(x * gw / divisor) + 1) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) {
        return 1;
    } else {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}

#if 0
ICudaEngine* createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,
      unsigned int numDetectedClasses, std::string& wts_name ) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 128, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");

    // yolov5 head
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 256 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(256);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), 512, 256, 1, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 128, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 128 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(128);

    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), 256, 128, 1, false, 1, 0.5, "model.17");
    //yolo P3
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (numDetectedClasses + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
     auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), 256, 256, 1, false, 1, 0.5, "model.20");
    //yolo P4
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (numDetectedClasses + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 256, 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.23");
    //yolo P5
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (numDetectedClasses + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2, numDetectedClasses);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();
    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));

    }
    return engine;
}
#endif

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, 
        unsigned int numDetectedClasses, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * get_width(512, gw) * 2 * 2));
    for (int i = 0; i < get_width(512, gw) * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, get_width(512, gw) * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), get_width(512, gw), DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(get_width(512, gw));
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, get_width(256, gw) * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), get_width(256, gw), DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(get_width(256, gw));
    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (numDetectedClasses + 5), 
           DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (numDetectedClasses + 5),
         DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (numDetectedClasses + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2, numDetectedClasses);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

struct NetworkInfo
{
    std::string networkType;
    std::string configFilePath;
    std::string wtsFilePath;
    std::string deviceType;
    std::string inputBlobName;
};

bool fileExists(const std::string fileName, bool verbose = true);
bool fileExists(const std::string fileName, bool verbose)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
}

static bool getYoloNetworkInfo (NetworkInfo &networkInfo, const NvDsInferContextInitParams* initParams)
{
    std::string yoloCfg = initParams->customNetworkConfigFilePath;
    std::string yoloType;

    std::transform (yoloCfg.begin(), yoloCfg.end(), yoloCfg.begin(), [] (uint8_t c) {
        return std::tolower (c);});

    if (yoloCfg.find("yolov2") != std::string::npos) {
        if (yoloCfg.find("yolov2-tiny") != std::string::npos)
            yoloType = "yolov2-tiny";
        else
            yoloType = "yolov2";
    } else if (yoloCfg.find("yolov3") != std::string::npos) {
        if (yoloCfg.find("yolov3-tiny") != std::string::npos)
            yoloType = "yolov3-tiny";
        else
            yoloType = "yolov3";
    } else if (yoloCfg.find("yolov5") != std::string::npos) {
        if (yoloCfg.find("yolov5s") != std::string::npos)
            yoloType = "yolov5s";
        else if (yoloCfg.find("yolov5m") != std::string::npos)
            yoloType = "yolov5m";
        else if (yoloCfg.find("yolov5l") != std::string::npos)
            yoloType = "yolov5l";
        else if (yoloCfg.find("yolov5x") != std::string::npos)
            yoloType = "yolov5x";
        else
            yoloType = "yolov5";
    } else {
        std::cerr << "Yolo type is not defined from config file name:"
                  << yoloCfg << std::endl;
        return false;
    }

    networkInfo.networkType     = yoloType;
    networkInfo.configFilePath  = initParams->customNetworkConfigFilePath;
    networkInfo.wtsFilePath     = initParams->modelFilePath;
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "data";

    if (networkInfo.configFilePath.empty() ||
        networkInfo.wtsFilePath.empty()) {
        std::cerr << "Yolo config file or weights file is NOT specified."
                  << std::endl;
        return false;
    }

    if (!fileExists(networkInfo.configFilePath) ||
        !fileExists(networkInfo.wtsFilePath)) {
        std::cerr << "Yolo config file or weights file is NOT exist."
                  << std::endl;
        return false;
    }

    return true;
}

bool NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine, float gd, float gw)
{
    NetworkInfo networkInfo;
    if (!getYoloNetworkInfo(networkInfo, initParams)) {
      return false;
    }

    IBuilderConfig* config = builder->createBuilderConfig();
    std::cout << "[NvDsInferYoloCudaEngineGet]the current class number is :" << initParams->numDetectedClasses << std::endl; 
    cudaEngine = build_engine(1, builder, config, DataType::kFLOAT, initParams->numDetectedClasses, gd, gw, networkInfo.wtsFilePath); 
    // cudaEngine = createEngine_s(1, builder, config, DataType::kFLOAT, initParams->numDetectedClasses,networkInfo.wtsFilePath);
    if (cudaEngine == nullptr)
    {
        std::cerr << "Failed to build cuda engine on "
                  << networkInfo.configFilePath << std::endl;
        return false;
    }
    config->destroy();
    return true;
}

extern "C"
bool NvDsInferYoloCudaEngineGetS(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool NvDsInferYoloCudaEngineGetS(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine)
{

    float gd = 0.33f, gw = 0.50f;
    return NvDsInferYoloCudaEngineGet(builder, initParams, dataType, cudaEngine, gd, gw);
}

extern "C"
bool NvDsInferYoloCudaEngineGetM(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool NvDsInferYoloCudaEngineGetM(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine)
{

    float gd = 0.67f, gw = 0.75f;
    return NvDsInferYoloCudaEngineGet(builder, initParams, dataType, cudaEngine, gd, gw);
}

extern "C"
bool NvDsInferYoloCudaEngineGetL(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool NvDsInferYoloCudaEngineGetL(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine)
{

    float gd = 1.0f, gw = 1.0f;
    return NvDsInferYoloCudaEngineGet(builder, initParams, dataType, cudaEngine, gd, gw);
}

extern "C"
bool NvDsInferYoloCudaEngineGetX(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool NvDsInferYoloCudaEngineGetX(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine)
{

    float gd = 1.33f, gw = 1.25f;
    return NvDsInferYoloCudaEngineGet(builder, initParams, dataType, cudaEngine, gd, gw);
}

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    
    // Get the actual number of classes
    // unsigned int numClasses = detectionParams.numClassesConfigured;
    float w,h,cy,cx,x1,x2,y1,y2;
    float * output = (float *)outputLayersInfo[0].buffer;
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    // for (int i = 0; i < (int)detectionParams.numClassesConfigured; i++)
    //    std::cout << "class id:"<< i << ", perClassPreclusterThreshold:" << detectionParams.perClassPreclusterThreshold[i] << std::endl;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        //if ( det.class_id != 11) continue;
        if (output[1 + det_size * i + 4] <= detectionParams.perClassPreclusterThreshold[det.class_id]) continue;
        
        cx = det.bbox[0];
        cy = det.bbox[1];
        w =  det.bbox[2];
        h =  det.bbox[3];
        x1 = cx - w/2.0;
        y1 = cy - h/2.0;
        //verification
        x2 = cx + w/2.0;
        y2 = cy + h/2.0;
        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x2 > networkInfo.width -1) x2 = networkInfo.width -1;
        if (y2 > networkInfo.height -1 ) y2 = networkInfo.height -1;
        // if (det.class_id >10)
        //   std::cout << "[NvDsInferParseCustomYoloV5]det.class_id: "<<det.class_id <<",conf:" << det.conf << std::endl;
        objectList.push_back({ (unsigned int)det.class_id, x1,y1,
                          (x2 - x1),(y2 - y1), det.conf});
                          // w,h, det.conf});
    }

    return true;
}

#if 1
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT,CLASS_NUM, gd, gw, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& file_path) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net == "s") {
            gd = 0.33;
            gw = 0.50;
        } else if (net == "m") {
            gd = 0.67;
            gw = 0.75;
        } else if (net == "l") {
            gd = 1.0;
            gw = 1.0;
        } else if (net == "x") {
            gd = 1.33;
            gw = 1.25;
        } else if (net == "c" && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
        } else {
            return false;
        }
    } else if ((std::string(argv[1]) == "-d" || std::string(argv[1]) == "-v" ) && argc == 4) {
        engine = std::string(argv[2]);
        file_path = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string wts_name = "";
    std::string engine_name = "";
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;
    if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s [.wts] [.engine] [s/m/l/x or c gd gw]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -d [.engine] ../samples  // deserialize plan file and run inference to recognize images" << std::endl;
        std::cerr << "./yolov5 -v [.engine] video_file  // deserialize plan file and run inference to recognize video" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    if (std::string(argv[1]) == "-s") {//!wts_name.empty()) {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, gd, gw, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    std::vector<std::string> file_names;
    if(std::string(argv[1]) == "-d") {
       if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
          std::cerr << "read_files_in_dir failed." << std::endl;
          return -1;
       }
    }
   
    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    std::map<int,std::string> class_dict;
    class_dict[0] = "person";
    class_dict[1] = "head";
    class_dict[2] = "wm_g_open";
    class_dict[3] = "wm_g_close";
    class_dict[4] = "wm_g_close_idle";
    class_dict[5] = "wm_g_close_work";
    class_dict[6] = "wm_g_close_unknown";
    class_dict[7] = "wm_s_open";
    class_dict[8] = "wm_s_close";
    class_dict[9] = "wm_s_close_idle";
    class_dict[10] = "wm_s_close_work";
    class_dict[11] = "wm_s_close_unknown";
    class_dict[12] = "dr_open";
    class_dict[13] = "dr_close";
    class_dict[14] = "dr_close_idle";
    class_dict[15] = "dr_close_work";
    class_dict[16] = "dr_close_unknown";
    class_dict[17] = "operation";
    class_dict[18] = "operation_in";
    class_dict[19] = "operation_out";

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if(std::string(argv[1]) == "-d") {
      int fcount = 0;
      for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            printf("processing file %s, W %d, H %d\n",file_names[f - fcount + 1 + b].c_str(), INPUT_W, INPUT_H);
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
            cv::imwrite("resized_" + file_names[f - fcount + 1 + b],pr_img);
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            //std::cout << res.size() << std::endl;
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, class_dict[(int)res[j].class_id]+" "+std::to_string(res[j].conf), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
     }
   }else if(std::string(argv[1]) == "-v") {
     cv::VideoCapture cap(img_dir);
     std::string outFile = img_dir + "-det.mp4";
     int fourcc = cv::VideoWriter::fourcc('m','p','4','v'); //('D', 'I', 'V', 'X');
     double fps = cap.get(cv::CAP_PROP_FPS);
     cv::Size size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
		(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
     if (fps<= 0)
       fps = 15;
     cv::VideoWriter  writer(outFile, fourcc, fps,size);
     
     cv::Mat frame;
     while (cap.read(frame)) { 
         cv::Mat pr_img = preprocess_img(frame, INPUT_W, INPUT_H); // letterbox BGR to RGB
         int i = 0;
         for (int row = 0; row < INPUT_H; ++row) {
             uchar* uc_pixel = pr_img.data + row * pr_img.step;
             for (int col = 0; col < INPUT_W; ++col) {
                 data[i] = (float)uc_pixel[2] / 255.0;
                 data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                 data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                 uc_pixel += 3;
                 ++i;
             }
         }   
       
         doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
         std::vector<Yolo::Detection> res;
         nms(res, &prob[0], CONF_THRESH, NMS_THRESH);
         for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(frame, res[j].bbox);
            std::string label = class_dict[(int)res[j].class_id];
            int baseline = 0, fontscale = 1.2, thickness = 2;
            cv::Size textSize = getTextSize(label, cv::FONT_HERSHEY_PLAIN, fontscale, thickness, &baseline);
            baseline += thickness;
            cv::Point textOrg(r.x, r.y < 2 ? r.y + 10: r.y);
            //cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::rectangle(frame, r, cv::Scalar(0, 0, 0xFF), 2);
            cv::rectangle(frame,textOrg - cv::Point(0, textSize.height),textOrg + cv::Point(textSize.width, textSize.height), cv::Scalar(0,0,0), -1);
            //cv::putText(frame, class_dict[(int)res[j].class_id]+" "+std::to_string(res[j].conf), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            cv::putText(frame, label, cv::Point(r.x, r.y < 2 ? r.y + 10: r.y), cv::FONT_HERSHEY_PLAIN, fontscale, cv::Scalar(0xFF, 0xFF, 0xFF), thickness);
           
            #if 1
            if ((int)res[j].class_id == 1 ) { // head
               int x = r.x;
               int y = r.y;
               int w = r.width;
               int h = r.height;
               int pixel = 30;
               // printf("x %d, y %d, w %d, h %d\n",x,y,w,h);
               cv::Mat obj_mat = (frame)(cv::Rect(x, y, w, h));
               for (int i = 0; i < w; i += pixel) {
                  for (int j = 0; j < h; j += pixel) {
                     for (int k = i; k < i + pixel && k < w; k++) {
                        for (int m = j; m < j + pixel && m < h; m++) {
                           obj_mat.at<cv::Vec3b>(m, k)[0] = obj_mat.at<cv::Vec3b>(j, i)[0];
                           obj_mat.at<cv::Vec3b>(m, k)[1] = obj_mat.at<cv::Vec3b>(j, i)[1];
                           obj_mat.at<cv::Vec3b>(m, k)[2] = obj_mat.at<cv::Vec3b>(j, i)[2] > 200 ? 50 : obj_mat.at<cv::Vec3b>(j, i)[2];
                        }
                     }
                  }
               } 
          
            }
            #endif
         }

         writer << frame;
     }
     cap.release();
     writer.release();
   }
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
#endif
