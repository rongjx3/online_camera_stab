//
// Created by ShiJJ on 2020/6/15.
//

#ifndef VIDEO_STAB_RT_AUTOFILTER_H
#define VIDEO_STAB_RT_AUTOFILTER_H

#include <deque>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "BSpline.h"
#include <android/log.h>

#define PI (2 * acos(0))
#define filterWeight(k,theta) exp(-(k) * (k) / 2.0 / (theta) / (theta)) / sqrt(2 * PI * (theta))
typedef struct limit{
    double xmin, xmax, ymin, ymax;
}limit;
class AutoFilter {
private:
    std::deque<cv::Mat> input_buffer_;
    std::deque<cv::Mat> trans_mat_buffer_;
    std::deque<cv::Mat> input_trans_buffer_;
    std::queue<cv::Mat> output_buffer_;
    std::vector<cv::Mat> window_;
    double crop_rate_ = 0.71;
    int max_size_;
    double sigma_;
    int delay_num_ = 10;
    std::vector<double> weight_vec_;
    cv::Size size_;
    cv::Mat cropvertex_;
    cv::Mat vertex_;
    cv::Point2f center;
    std::deque<cv::Mat> global_trans_;
    //new idea
    static int predict_num_;
    double que_x_[5], que_y_[5];
    int num_que_ = 0;
    double f_num_[5];
    int ex_count = 0;
    double cur_x = 0, cur_y = 0;
    bool need_fit_x_ = false;
    bool need_fit_y_ = false;
    std::queue<cv::Mat> s_mat_;
    std::vector<double> gauss_;
    double index_ = 0;
    std::queue<limit> limit_que_;

    cv::Mat cum_H,trans_for_cumh;
    std::deque<cv::Point2f> trans_buffer_, sca_buffer_;
    int keep_num,keep;
    int mode,move_status;
    double keep_rate,keep_change_rate;
    bool first_in;

    void queue_in(double q[], int m, double x);
    void polyfit(double arrX[], double arrY[], int num, int n, double* result);
    double calError(double* ori, double* aft, int n);

    bool putIntoWindow(int target, int offset = 0);
    bool isInside(cv::Mat cropvertex ,cv::Mat newvertex);
    void processCrop(const cv::Mat& comp, const cv::Size& size);
    void decomposeHomo(cv::Mat h, cv::Point2f cen, cv::Mat &perp, cv::Mat &sca, cv::Mat &shear, cv::Mat &rot, cv::Mat &trans);
    void analyzeTrans(cv::Mat& comp);
public:
    bool write_status_ = false;

    explicit AutoFilter(int max_size = 30, double sigma = 40);
    void set_move_status(int m);
    bool push(cv::Mat goodar);
    cv::Mat pop();
};


#endif //VIDEO_STAB_RT_AUTOFILTER_H
