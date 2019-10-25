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

vector<size_t> argsort(const float *v, const int Len){

    vector<size_t> idx(Len);
    iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2){return v[i1] > v[i2];});
    return idx;
}

int getFileList(string dirent, vector<string> &FileList){

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
            FileList.push_back(s);
    }
    closedir(p_dir);
    return FileList.size();
}

int preprocess(vector<vector<float>> &dst_vec, vector<string> fileList, string filePath, vector<int> inputSize, int batch_size=1, int channels=3)
{
    vector<float> mean {0.485, 0.456, 0.406};
    vector<float> std {0.229, 0.224, 0.225};
    vector<float> src;
    vector<float> dst (batch_size * channels * inputSize[0] * inputSize[1]);
    string imgPath;

    for (auto imagename: fileList)
    {
        imgPath = filePath+imagename;
        auto image = imread(imgPath, IMREAD_COLOR);
        auto src_height = image.size().height;
        auto src_width = image.size().width;
	int dst_width;
	int dst_height;
        if (src_height > src_width) {
	    dst_width = 256;
	    dst_height = floor(256 * src_height / src_width);
	}
        else {
	    dst_height = 256;
	    dst_width = floor(256 * src_width / src_height);
	}
	cv::resize(image, image, Size(dst_width, dst_height), 0, 0, cv::INTER_AREA);

	const int offset_H = (dst_height - inputSize[0])/2;
	const int offset_W = (dst_width - inputSize[1])/2;
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

        for (int bs = 0; bs < batch_size; bs++) {
            for (int c = 0; c < channels; c++) {
                for (int j = 0, hw = inputSize[0] * inputSize[1]; j < hw; j++) {
                    dst[bs * channels * hw + c * hw + j] = (src[channels * j + 2 - c] - mean[c]) / std[c];
                }
            }
        }

        dst_vec.push_back(dst);
    }
    return dst_vec.size();
}

int main(int argc, char *argv[]) {
	
    int batch_size = 1;
    if (argc != 4) {
            cerr << "Usage: " << argv[0] << " engine.plan filePath LabelFile" << endl;
            return 1;
    }

    cout << "Read file list..." << endl;
    string filePath = argv[2];
    vector<string> fileList;
    int ret = getFileList(filePath, fileList);
    if (ret < 0){
        return -1;
    }

    cout << "Make label map..." << endl;
    string LabelFile = argv[3];
    ifstream iFile(LabelFile);
    string s;
    string imagename;
    int label;
    string delimiter = " ";
    size_t pos = 0;
    unordered_map<string, int> label_map;
    while (getline(iFile, s))
    {
        pos = s.find(delimiter);
        imagename = s.substr(0, pos);
        label = stoi(s.substr(pos+1, s.length()));
        label_map.insert(make_pair(imagename, label));
    }

    cout << "Loading engine..." << endl;
    auto engine = dawnbench::Engine(argv[1]);
    auto inputSize = engine.getInputSize();
    auto outputSize = engine.getOutputSize();

    cout << "Do preprocessing..." << endl;
    vector<vector<float>> dst_vec;
    if (preprocess(dst_vec, fileList, filePath, inputSize, 1, 3) < 0)
    {
        return -1;
    }

    cout << "Do Inference..." << endl;
    int count = 0;
    float* output;
    vector<size_t> index;
    unordered_map<string, int>::iterator it;
    int true_index = 0;
    int true_count = 0;
    double totaltime = 0;
    for (auto imagename: fileList){
        engine.load_image(dst_vec[count]);
        auto start = chrono::steady_clock::now();
        output = engine.infer();
        auto stop = chrono::steady_clock::now();
        auto timing = chrono::duration_cast<chrono::duration<double>>(stop - start).count();
        if (count > 9)
	    totaltime += timing;        

        index = argsort(output, batch_size * outputSize[0]);
        it = label_map.find(imagename);
        true_index = it->second;
        for (int i = 0; i < 5; i++){
            if (index[i] == (true_index-1))
                true_count++;
        }
        count++;
    }
   
    cout << "Top5 inference accuracy : " << double(true_count)/count << endl;
    cout << "avg inference time : " << totaltime/(count-10)*1000.0 << "ms" << endl; 

}
