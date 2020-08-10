//
// Created by 张哲华 on 19/09/2017.
//

#ifndef VIDEOSTABLE_THREADCOMPENSATION_H
#define VIDEOSTABLE_THREADCOMPENSATION_H

#include <thread>
#include <cmath>
#include "ThreadContext.h"
#include "Filter.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include <queue>
#include "ThreadRollingShutter.h"
#include "HomoExtractor.h"
#include <string>
#include "AutoFilter.h"
#include <fstream>

namespace threads {
    class ThreadCompensation {
    public:
        Size videoSize, frameSize;
        bool cropControlFlag = true;//裁剪控制
        bool drawFlag = true;
        float cropRation;//裁剪率
        bool shakeDetect;//抖动检测
        bool is_write_to_file_ = false;
    private:
        const char* TAG = "CompensationThread";
        thread worker_thread_;
        int cm_las_index_ = 0;
        int cm_cur_index_ = 1;
        int ex_index_ = 0;
        int out_index_ = 0;
        bool with_roll = false;
        Filter filter;
        Mat curGray, lastGray;
        Mat H_scale;
        std::vector<Point2f> lastFeatures,curFeatures;
        std::vector<Point2f> lastFeaturesTmp,curFeaturesTmp;
        std::vector<uchar> status,status_choose;
        Vec<double, 3> lastRot;

        bool is_first_use_rtheta = true;
        bool is_first_cal_features = true;
        cv::Mat last_homography_;
        AutoFilter filter1;

//        HomoExtractor homoExtractor;

        void worker();
        bool stable_count(double e);
        Mat computeAffine();
        void frameCompensate();
        double computeMaxDegree( vector<Point2f> img_line , vector<Point2f> crop_line , double degree , Point2f center );
        void WriteToFile(FILE* old_file, cv::Mat mat);
        cv::Mat inmat=(cv::Mat_<double>(3, 3)<<1430.2,0.0,505.7, 0.0,1422.9,922.1,0.0,0.0,1.0);//OnePlus 6T
//        cv::Mat inmat=(cv::Mat_<double>(3, 3)<<1492.89950430177,0.0,940.850079740057, 0.0,1496.13805384036,552.228021875255,0.0,0.0,1.0);//demo board

        bool is_stable_;

        cv::Vec2f CalTranslationByR(cv::Mat r);
        Mat RR2stableVec = (cv::Mat_<double>(3, 3)<<0.0, 1.0, 0.0, -1.0, 0.0, 1080.0, 0.0, 0.0, 1.0);
        Mat stableVec2RR = (cv::Mat_<double>(3, 3)<<0.0, -1.0, 1080.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        cv::Mat crop_vertex;
        void decomposeHomo(cv::Mat h, Point2f cen, cv::Mat &perp, cv::Mat &sca, cv::Mat &shear, cv::Mat &rot, cv::Mat &trans);
    public:
        void start();

        ~ThreadCompensation();
    };
}


#endif //VIDEOSTABLE_THREADCOMPENSATION_H
