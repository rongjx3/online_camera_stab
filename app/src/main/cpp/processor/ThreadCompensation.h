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
//#include "AutoFilter_2.h"
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
        int homo_type = 1;
    private:
        const char* TAG = "CompensationThread";
        thread worker_thread_;
        int cm_las_index_ = 0;
        int cm_cur_index_ = 1;
        int out_index_ = 0;
        cv::Point2f center;
        Filter filter;
        Vec<double, 3> lastRot;
        cv::Mat lastf;
        std::vector<double> error_q, error_q_x, error_q_y, error_q_z;

        bool is_first_use_rtheta = true;
        bool is_first_frame = true;
        AutoFilter filter1;
        cv::Mat cum_H = cv::Mat::eye(3, 3, CV_64F);

        HomoExtractor homoExtractor;

        void worker();
        bool stable_count(double e);
        Mat computeAffine(cv::Mat &R);
        void frameCompensate();
        cv::Mat limit_Mat(cv::Mat homo);
        void decomposeHomo(cv::Mat h, Point2f cen, cv::Mat &perp, cv::Mat &sca, cv::Mat &shear, cv::Mat &rot, cv::Mat &trans);

        double computeMaxDegree( vector<Point2f> img_line , vector<Point2f> crop_line , double degree , Point2f center );
        void WriteToFile(FILE* old_file, cv::Mat mat);
        //cv::Mat inmat=(cv::Mat_<double>(3, 3)<<1430.2,0.0,505.7, 0.0,1422.9,922.1,0.0,0.0,1.0);//OnePlus 6T
        cv::Mat inmat=(cv::Mat_<double>(3, 3)<<1490.1,0.0,533.1, 0.0,1481.7,974.5,0.0,0.0,1.0);//MI8
        //cv::Mat inmat=(cv::Mat_<double>(3, 3)<<1694.8,0.0,517.8, 0.0,1687.7,961.7,0.0,0.0,1.0);//HWp40pro
//        cv::Mat inmat=(cv::Mat_<double>(3, 3)<<1492.89950430177,0.0,940.850079740057, 0.0,1496.13805384036,552.228021875255,0.0,0.0,1.0);//demo board

        bool is_stable_;

        cv::Vec2f CalTranslationByR(cv::Mat r);
        Mat RR2stableVec = (cv::Mat_<double>(3, 3)<<0.0, 1.0, 0.0, -1.0, 0.0, 1080.0, 0.0, 0.0, 1.0);
        Mat stableVec2RR = (cv::Mat_<double>(3, 3)<<0.0, -1.0, 1080.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    public:
        void start();

        ~ThreadCompensation();
    };
}


#endif //VIDEOSTABLE_THREADCOMPENSATION_H
