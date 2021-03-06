//
// Created by ShiJJ on 2020/6/1.
//

#ifndef STABLE_CAMERA_HOMOEXTRACTOR_H
#define STABLE_CAMERA_HOMOEXTRACTOR_H


#include <opencv2/opencv.hpp>
#include "ThreadContext.h"
#include <android/log.h>
#include <mutex>
#include <thread>
#include <condition_variable>
using namespace threads;
#define PI 3.1415926535
class HomoExtractor {
private:

    std::vector<cv::Point2f> curFeaturesTmp, lastFeaturesTmp;
    cv::Mat lastGray,curGray;
    int ex_index_ = 0;
    int point_num[16];
    int statussize;
    cv::Mat H, second_H;
    cv::Mat last_perp=cv::Mat::eye(3, 3, CV_64F), last_shear=cv::Mat::eye(3, 3, CV_64F);
    bool stable_move=true, stable_move2=false;
    std::vector<int> block_index_;
    int move_status = 1;

    std::mutex mutex_, detect_mutex_, track_mutex_;
    cv::Mat last_lu_, last_ru_, last_ld_, last_rd_;
    cv::Mat cur_lu_, cur_ru_, cur_ld_, cur_rd_;
    cv::Mat gyro_r;
    std::thread detect_thread0_, detect_thread1_, detect_thread2_, detect_thread3_;
    std::thread track_thread0_, track_thread1_, track_thread2_, track_thread3_;
    std::condition_variable is_wait_over;
    int detect_thread_over_num_ = 4;
    int track_thread_over_num_ = 4;
    std::vector<cv::Point2f> last_features_[4], cur_features_[4];


    void detectFeature(const cv::Mat& img, int pic_index);//左上0，右上1， 左下2， 右下3
    void trackFeature(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& last_rect, const cv::Mat& cur_rect, int pic_index);
    std::vector<cv::Point2f> detectCertainFeature(const std::vector<int>& block_index);
    void trackCertainFeature( std::vector<cv::Point2f>& last_add_feature);
    double point_distance(cv::Point2f p1,cv::Point2f p2);
    bool outOfImg(const cv::Point2f &point, const cv::Size &size);
    int in_area(cv::Point2f p1, int w, int h);
    bool judge_area();
    void calcul_Homo(std::vector<char> &ifselect, int niter, int type);
    void calcul_Aff(std::vector<char> &ifselect);
    bool judge_recal_simple(cv::Mat img1, std::vector<char> ifselect);
    double calcul_H_error(int c_h, int c_w);
    cv::Point2f goround(cv::Point2f p1, cv::Point2f p0, double degree);
    cv::Point2f goscale(cv::Point2f p1,cv::Point2f p0, double scale);
    double vec_cos(cv::Point2f s, cv::Point2f e1, cv::Point2f e2);
    void decomposeHomo(cv::Mat h, Point2f cen, cv::Mat &perp, cv::Mat &sca, cv::Mat &shear, cv::Mat &rot, cv::Mat &trans);
    double cal_degree(cv::Point2f vec1, cv::Point2f vec2);

public:
    bool draw_information = false;
    cv::Mat extractHomo( cv::Mat& img1, cv::Mat& img2, cv::Mat R_old);
    void setDrawStatus(bool is_draw);
    void set_move_status(int m);
};


#endif //STABLE_CAMERA_HOMOEXTRACTOR_H
