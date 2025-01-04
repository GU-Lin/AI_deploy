#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#pragma once
struct PredBox
{
    float cx;
    float cy;
    float width;
    float height;
    float score;
    int label;
};

const std::vector<std::string> coconame = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

class objectDetection
{
    public:
        objectDetection();
        ~objectDetection();
        void preprocess(cv::Mat input, cv::Mat &output);
        void HWC2NormalCHW(cv::Mat input, std::vector<float> &data);
        float iou(PredBox box1, PredBox box2);
        float areaBox(PredBox &box);
        void postprocess(cv::Mat &input);
        void NMS(std::vector<PredBox> &pred);
        void draw(cv::Mat &input, cv::Mat &output, std::vector<PredBox> &pred);
        void run(std::vector<float> &input);
        std::vector<PredBox> m_PredBox_vector;
    private:
        // inference engine;
        int i_height;
        int i_width;
        int i_channels;
        float m_conf_thres = 0.30;
        float m_iou_thres = 0.45;
        int m_class_number = 0;
        bool autoFlag = true;
        bool scaleFillFlag = false;
        int stride = 32;
};
