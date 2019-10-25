#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <chrono>
#include <typeinfo>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>
#include <cuda_runtime.h>
#include <math.h>
#include "engine.h"

using namespace std;
using namespace cv;

vector<size_t> argSort(const float *v, const int Len){

    vector<size_t> idx(Len);
    iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2){return v[i1] > v[i2];});
    return idx;
}

int getFileList(string dirent, vector<string> &fileList){

    DIR *p_dir;
    struct dirent *p_dirent;

    if((p_dir = opendir((dirent).c_str())) == NULL){
        cout << "check pir path:" << (dirent).c_str() << "failed" <<endl;
        return -1;
    }
    while((p_dirent=readdir(p_dir)))
    {
        string s(p_dirent->d_name);
        if(s != "." && s != "..")
            fileList.push_back(s);
    }
    closedir(p_dir);
    return fileList.size();
}

int doPreprocess(vector<vector<float>> &dstVec, vector<string> fileList, string filePath, vector<int> inputSize, int batchSize=1, int channels=3)
{
    vector<float> mean {0.485, 0.456, 0.406};
    vector<float> std {0.229, 0.224, 0.225};
    vector<float> src;
    vector<float> dst (batchSize * channels * inputSize[0] * inputSize[1]);
    string imgPath;

    for (auto imageName: fileList)
    {
        imgPath = filePath + imageName;
        auto image = imread(imgPath, IMREAD_COLOR);
        auto srcHeight = image.size().height;
        auto srcWidth = image.size().width;
	int dstWidth;
	int dstHeight;
        if (srcHeight > srcWidth) {
	    dstWidth = 256;
	    dstHeight = floor(256 * srcHeight / srcWidth);
	}
        else {
	    dstHeight = 256;
	    dstWidth = floor(256 * srcWidth / srcHeight);
	}
	cv::resize(image, image, Size(dstWidth, dstHeight), 0, 0, cv::INTER_AREA);

	const int offset_H = (dstHeight - inputSize[0])/2;
	const int offset_W = (dstWidth - inputSize[1])/2;
	cv::Rect myROI(offset_W, offset_H, inputSize[0], inputSize[1]);
	image = image(myROI).clone();

	cv::Mat pixels;
        image.convertTo(pixels, CV_32FC3, 1.0/255, 0);

        if (pixels.isContinuous())
            src.assign((float*)pixels.datastart, (float*)pixels.dataend);
        else {
            cerr << "Error reading image " << imgPath << endl;
            return -1;
        }

        for (int bs = 0; bs < batchSize; bs++) {
            for (int c = 0; c < channels; c++) {
                for (int j = 0, hw = inputSize[0] * inputSize[1]; j < hw; j++) {
                    dst[bs * channels * hw + c * hw + j] = (src[channels * j + 2 - c] - mean[c]) / std[c];
                }
            }
        }

        dstVec.push_back(dst);
    }
    return dstVec.size();
}

int main(int argc, char *argv[]) {
	
    int batch_size = 1;
    if (argc != 4) {
            cerr << "Usage: " << argv[0] << " engine.plan imagePath labelFile" << endl;
            return -1;
    }

    cout << "Read file list..." << endl;
    string filePath = argv[2];
    vector<string> fileList;
    int ret = getFileList(filePath, fileList);
    if (ret < 0){
        return -1;
    }

    cout << "Make label map..." << endl;
    string labelFile = argv[3];
    ifstream iFile(labelFile);
    string s;
    string imageName;
    int label;
    string delimiter = " ";
    size_t pos = 0;
    unordered_map<string, int> labelMap;
    while (getline(iFile, s))
    {
        pos = s.find(delimiter);
        imageName = s.substr(0, pos);
        label = stoi(s.substr(pos+1, s.length()));
        labelMap.insert(make_pair(imageName, label));
    }

    cout << "Loading engine..." << endl;
    auto engine = dawnbench::Engine(argv[1]);
    auto inputSize = engine.getInputSize();
    auto outputSize = engine.getOutputSize();

    cout << "Do preprocessing..." << endl;
    vector<vector<float>> dstVec;
    if (doPreprocess(dstVec, fileList, filePath, inputSize, 1, 3) < 0)
    {
        return -1;
    }

    cout << "Do Inference..." << endl;
    int count = 0;
    float* output;
    vector<size_t> index;
    unordered_map<string, int>::iterator it;
    int trueLabel = 0;
    int trueCount = 0;
    double totaltime = 0;
    for (auto imgName: fileList){
        engine.loadImage(dstVec[count]);
        auto start = chrono::steady_clock::now();
        output = engine.infer();
        auto stop = chrono::steady_clock::now();
        auto latency = chrono::duration_cast<chrono::duration<double>>(stop - start).count();
        if (count > 9)
	    totaltime += latency;        

        index = argSort(output, batch_size * outputSize[0]);
        it = labelMap.find(imgName);
        trueLabel = it->second;
        for (int i = 0; i < 5; i++){
            if (index[i] == (trueLabel-1))
                trueCount++;
        }
        count++;
    }
   
    cout << "Top5 inference accuracy : " << double(trueCount)/count << endl;
    cout << "avg inference time : " << totaltime/(count-10)*1000.0 << "ms" << endl; 

}
