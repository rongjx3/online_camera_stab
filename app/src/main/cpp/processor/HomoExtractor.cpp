//
// Created by ShiJJ on 2020/6/1.
//

#include "HomoExtractor.h"
#define LOG_TAG    "c_HomoExtractor"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

double HomoExtractor::point_distance(cv::Point2f p1, cv::Point2f p2) {
    cv::Point2f d = p1 - p2;
    double d_mu;
    d_mu = sqrt(d.x * d.x + d.y * d.y);
    return d_mu;
}
bool HomoExtractor::outOfImg(const cv::Point2f &point, const cv::Size &size) {
    return (point.x <= 0 || point.y <= 0 || point.x >= size.width - 1 || point.y >= size.height - 1 );
}

int HomoExtractor::in_area(cv::Point2f p1, int w, int h) {
    int res1,res2;
    if(p1.x<w)
    {
        res1 = 0;
    }
    else if(p1.x>=w && p1.x<2*w)
    {
        res1 = 1;
    }
    else if(p1.x>=2*w && p1.x<3*w)
    {
        res1 = 2;
    }
    else
    {
        res1 = 3;
    }

    if(p1.y<h)
    {
        res2 = 0;
    }
    else if(p1.y>=h && p1.y<2*h)
    {
        res2 = 1;
    }
    else if(p1.y>=2*h && p1.y<3*h)
    {
        res2 = 2;
    }
    else
    {
        res2 = 3;
    }

    return res1 * 4 + res2;
}

bool HomoExtractor::judge_area() {
    block_index_.clear();
    bool redetect = false;
    int num[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    for (int i = 0; i < 16; i++) {
        if (point_num[i] < num[i]) {
            redetect = true;
            block_index_.push_back(i);
        }
    }

    return redetect;
}

bool HomoExtractor::judge_recal_simple(cv::Mat img1, std::vector<char> ifselect) {
    int half_w=img1.cols/2 , half_h=img1.rows/2;
    double inli[4]={0,0,0,0}, p_num[4]={0,0,0,0};
    for(int i=0; i<curFeaturesTmp.size(); i++)
    {
        if(i>=ifselect.size())
        {
            break;
        }
        int i1,i2;
        if(curFeaturesTmp[i].x<half_w)
        {
            i1=0;
        }
        else{
            i1=1;
        }
        if(curFeaturesTmp[i].y<half_h)
        {
            i2=2;
        }
        else{
            i2=3;
        }
        p_num[i1]++;
        p_num[i2]++;

        if(ifselect[i]==1)
        {
            inli[i1]++;
            inli[i2]++;
        }
    }

    double rate[4];
    for(int i=0;i<4;i++)
    {
        rate[i]=double(inli[i])/double(p_num[i]);
    }
    bool re=false;
    double limit=0.25;
    for(int i=0;i<4;i++)
    {
        if((rate[i]<limit || inli[i]<2) && p_num[i]>1)
        {
            re=true;
        }
    }

    return re;
}

void HomoExtractor::detectFeature(const cv::Mat& img, int pic_index) {
    //LOGI("step2_1");
    double quality_level = 0.1;
    double min_distance = 8;
    int max_corners = 8;

    std::vector<cv::Point2f> lastFeatures_a[16], startp;
//    lastFeatures.clear();
    last_features_[pic_index].clear();

    int half_w=img.cols/2 , half_h=img.rows/2;
    startp.push_back(cv::Point2f(0, 0));
    startp.push_back(cv::Point2f(0, half_h));
//    startp.push_back(cv::Point2f(0, 2*half_h));
//    startp.push_back(cv::Point2f(0, 3*half_h));

    startp.push_back(cv::Point2f(half_w, 0));
    startp.push_back(cv::Point2f(half_w, half_h));
//    startp.push_back(cv::Point2f(half_w, 2*half_h));
//    startp.push_back(cv::Point2f(half_w, 3*half_h));

//    startp.push_back(cv::Point2f(2*half_w, 0));
//    startp.push_back(cv::Point2f(2*half_w, half_h));
//    startp.push_back(cv::Point2f(2*half_w, 2*half_h));
//    startp.push_back(cv::Point2f(2*half_w, 3*half_h));
//
//    startp.push_back(cv::Point2f(3*half_w, 0));
//    startp.push_back(cv::Point2f(3*half_w, half_h));
//    startp.push_back(cv::Point2f(3*half_w, 2*half_h));
//    startp.push_back(cv::Point2f(3*half_w, 3*half_h));

    //LOGI("step2_2");
    cv::Rect rect[16];
    cv::Mat lastGray_a[16];


    for(int i=0;i<4;i++)
    {
        rect[i]=cv::Rect(startp[i].x, startp[i].y, half_w, half_h);//左上角坐标以及矩形宽和高
        lastGray_a[i] = img(rect[i]);
        goodFeaturesToTrack(lastGray_a[i], lastFeatures_a[i], max_corners, quality_level, min_distance);//检测特征点

        for(int j=0; j<lastFeatures_a[i].size(); j++)
        {
            cv::Point2f pt=lastFeatures_a[i][j]+startp[i];//转化成全局坐标
            last_features_[pic_index].push_back(pt);
        }
    }
    //LOGI("step2_3");

    std::unique_lock<std::mutex> detect_lock(detect_mutex_);

    detect_thread_over_num_--;
    if(!detect_thread_over_num_){
        is_wait_over.notify_one();

    }
    detect_lock.unlock();

    //LOGI("step2_4");
}

std::vector<cv::Point2f> HomoExtractor::detectCertainFeature(const std::vector<int>& block_index) {
    std::vector<cv::Point2f> ret_feature;
    double quality_level = 0.1;
    double min_distance = 8;
    int max_corners = 4;
    std::vector<cv::Point2f> lastFeatures_a;
    int half_w=lastGray.cols/4 , half_h=lastGray.rows/4;
    cv::Mat lastGray_a;
    for(auto i : block_index){
        int x_i = i % 4;
        int y_i = i / 4;
        cv::Rect rect(y_i*half_w, x_i*half_h, half_w, half_h);
        lastGray_a = lastGray(rect);
        goodFeaturesToTrack(lastGray_a, lastFeatures_a,  max_corners, quality_level, min_distance);
        cv::Point2f temp(y_i*half_w, x_i*half_h);
        for(int j=0; j<lastFeatures_a.size(); j++)
        {
            cv::Point2f pt=lastFeatures_a[j]+temp;//转化成全局坐标
            ret_feature.push_back(pt);
        }
    }
    return ret_feature;
}

void HomoExtractor::trackCertainFeature( std::vector<cv::Point2f> &last_add_feature) {
    double rate = 1.4;
    double min_rate = 0.8;
    double max_rate = 1.4;
    std::vector<cv::Point2f> cur_add_feature;
    std::vector<uchar> add_status;
    std::vector<float> err;
    calcOpticalFlowPyrLK( lastGray , curGray , last_add_feature , cur_add_feature , add_status , err);
    int max = last_add_feature.size() < cur_add_feature.size() ? last_add_feature.size() : cur_add_feature.size();
    double dis_sum=0;
    double gradient = 0;
    for(int i=0;i<max;i++)
    {
        dis_sum += point_distance(last_add_feature[i],cur_add_feature[i]);
//        gradient += (cur_add_feature[i].y - last_add_feature[i].y)
//                        /point_distance(last_add_feature[i],cur_add_feature[i]);
    }
    double dis_avg=0;
    dis_avg=dis_sum/max;
    double avg_gradient = gradient / max;
    max = max < add_status.size() ? max : add_status.size();
    for(int i=0;i<max;i++)
    {
        if(point_distance(last_add_feature[i],cur_add_feature[i]) > dis_avg * rate)
        {
            add_status[i] = 0;
        }
//        gradient += (cur_add_feature[i].y - last_add_feature[i].y)
//                        /point_distance(last_add_feature[i],cur_add_feature[i]);
        if(gradient < avg_gradient){
            add_status[i] = 0;
        }
    }


    //计算平均角度，先确定合理的起始向量，再计算间的平均角度
    cv::Point2f start_vec;
    double deg_avg=0;
    for(int i=0;i<max;i++)
    {
        start_vec = cur_add_feature[i] - last_add_feature[i];
        double sum_deg = 0,sum_count = 0;
        for(int j=0;j<max;j++)
        {
            cv::Point2f vec = cur_add_feature[j] - last_add_feature[j];
            double deg = cal_degree(start_vec, vec);
            //LOGI("single_deg: %f", deg);
            if(deg != 2*PI)
            {
                sum_count++;
                sum_deg += deg;
            }
        }

        if(sum_count != 0)
        {
            deg_avg=sum_deg/sum_count;
        }
        //LOGI("deg_avg: %f, sum & max: %f/%f", deg_avg,sum_deg,max);
        if( deg_avg < PI/2 && deg_avg > -PI/2 && sum_count != 0 )
        {
            break;
        }
    }
    //LOGI("deg_avg find end: %f", deg_avg);
    double range = PI/4;
    if( deg_avg < PI/2 && deg_avg > -PI/2 ) {
        for (int i = 0; i < max; i++) {
            cv::Point2f vec = cur_add_feature[i] - last_add_feature[i];
            if (!(cal_degree(start_vec, vec) > deg_avg - range && cal_degree(start_vec, vec) < deg_avg + range)) {
                add_status[i] = 0;
            }
        }
    }


    cv::Mat m_Fundamental;
    std::vector<uchar> m_RANSACStatus;
    cv::Mat p1(last_add_feature);
    cv::Mat p2(cur_add_feature);
    double outliner=0;
    m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, cv::FM_RANSAC, 3, 0.99);
    for (int j = 0 ; j < add_status.size() ; j++ )
    {
        if (m_RANSACStatus.size() > j && m_RANSACStatus[j] == 0) // 状态为0表示野点(误匹配)
        {
            add_status[j] = 0;
        }
        if(add_status[j]==0)
        {
            outliner++;
        }
    }

    int new_point_num[16] = {0};
    for(auto pt : cur_add_feature){
        int a = in_area(pt, lastGray.cols/4, lastGray.rows/4);
        new_point_num[a]++;
    }
    LOGI("new_point_num :new[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]",
         new_point_num[0], new_point_num[1],  new_point_num[2], new_point_num[3],
         new_point_num[4],  new_point_num[5],  new_point_num[6], new_point_num[7],
         new_point_num[8],  new_point_num[9],  new_point_num[10], new_point_num[11],
         new_point_num[12], new_point_num[13], new_point_num[14], new_point_num[15]);


    statussize += max;
    int idx = 0;
    for (auto itC = cur_add_feature.begin(), itP = last_add_feature.begin(); itC != cur_add_feature.end(); itC ++, itP ++, idx ++) {
        LOGI("see err in point: %f",  err[idx]);
        if (add_status[idx] == 0 || err[idx] > 20 || outOfImg(*itC, cv::Size(lastGray.cols, lastGray.rows))) {

        } else {
            cv::Point2f cfp=*itC * ThreadContext::DOWNSAMPLE_SCALE;
            cv::Point2f lfp=*itP * ThreadContext::DOWNSAMPLE_SCALE;
            curFeaturesTmp.push_back(cfp);
            lastFeaturesTmp.push_back(lfp);
        }
    }
}

void HomoExtractor::trackFeature(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat& last_rect, const cv::Mat& cur_rect, int pic_index) {
    double rate = 1.4;
    double min_rate = 0.6;
    double max_rate = 1.4;
    std::vector<uchar> status;
    status.clear();
    std::vector<float> err;
//    curFeatures.clear();
    cur_features_[pic_index].clear();
    //LOGI("step4_1");
    //LOGI("OFP points size: %d",last_features_[pic_index].size());
    if(last_features_[pic_index].size() == 0) {
        LOGI("OFP size is zero!");
        track_thread_over_num_--;
        if (!track_thread_over_num_) {
            is_wait_over.notify_one();
        }
        //track_lock.unlock();

        return ;
    }
    calcOpticalFlowPyrLK( last_rect , cur_rect , last_features_[pic_index] , cur_features_[pic_index] , status , err);//根据已检测到的前一帧特征点在后一帧查找匹配的特征点
    //如果没找到匹配点会将前一帧特征点位置复制到curFeatures中，并在status中标记为0
//    status_choose.clear();
//    status_choose.assign(status.begin(), status.end());//将status复制到status_choose中

    //LOGI("step4_2");
    int max = last_features_[pic_index].size() < cur_features_[pic_index].size() ? last_features_[pic_index].size() : cur_features_[pic_index].size();
    double dis_sum=0;
    //如果是没有找到匹配点，两点之间的距离为0，不影响平均距离
    for(int i=0;i<max;i++)
    {
        dis_sum += point_distance(last_features_[pic_index][i],cur_features_[pic_index][i]);
    }
    double dis_avg=0;
    dis_avg=dis_sum/max;
    max = max < status.size() ? max : status.size();
    for(int i=0;i<max;i++)
    {
        if(point_distance(last_features_[pic_index][i],cur_features_[pic_index][i]) > dis_avg * rate)
        {
            status[i] = 0;
        }
    }//如果大于特征点之间的距离大于平均距离的1.4倍，则舍弃

    //LOGI("step4_3");
    //计算平均角度，先确定合理的起始向量，再计算间的平均角度
    cv::Point2f start_vec;
    double deg_avg=0;
    for(int i=0;i<max;i++)
    {
        start_vec = cur_features_[pic_index][i] - last_features_[pic_index][i];
        double sum_deg = 0,sum_count = 0;
        for(int j=0;j<max;j++)
        {
            cv::Point2f vec = cur_features_[pic_index][j] - last_features_[pic_index][j];
            double deg = cal_degree(start_vec, vec);
            //LOGI("single_deg: %f", deg);
            if(deg != 2*PI)
            {
                sum_count++;
                sum_deg += deg;
            }
        }

        if(sum_count != 0)
        {
            deg_avg=sum_deg/sum_count;
        }
        //LOGI("deg_avg: %f, sum & max: %f/%f", deg_avg,sum_deg,max);
        if( deg_avg < PI/2 && deg_avg > -PI/2 && sum_count != 0 )
        {
            break;
        }
    }
    //LOGI("deg_avg find end: %f", deg_avg);
    double range = PI/4;
    if( deg_avg < PI/2 && deg_avg > -PI/2 ) {
        for (int i = 0; i < max; i++) {
            cv::Point2f vec = cur_features_[pic_index][i] - last_features_[pic_index][i];
            if (!(cal_degree(start_vec, vec) > deg_avg - range && cal_degree(start_vec, vec) < deg_avg + range)) {
                status[i] = 0;
            }
        }
    }

    //LOGI("step4_4");
    cv::Mat m_Fundamental;
    std::vector<uchar> m_RANSACStatus;
    cv::Mat p1(last_features_[pic_index]);
    cv::Mat p2(cur_features_[pic_index]);
    double outliner=0;
    m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, cv::FM_RANSAC, 3, 0.99);

    //LOGI("step4_5");
    for (int j = 0 ; j < status.size() ; j++ )
    {
        if (m_RANSACStatus.size() > j && m_RANSACStatus[j] == 0) // 状态为0表示野点(误匹配)
        {
            status[j] = 0;
        }
        if(status[j]==0)
        {
            outliner++;
        }
    }

    statussize+=max;
    int half_w=img1.cols/4 , half_h=img1.rows/4;
    cv::Point2f center(img1.cols / 2, img1.rows / 2);
    int idx = 0;
//    curFeaturesTmp.clear();
//    lastFeaturesTmp.clear();

    cv::Point2f convert((pic_index % 2) * last_rect.cols, (pic_index / 2) * last_rect.rows);
    //LOGI("step4_6");
    std::unique_lock<std::mutex> track_lock(track_mutex_);
    for (auto itC = cur_features_[pic_index].begin(), itP = last_features_[pic_index].begin(); itC != cur_features_[pic_index].end(); itC ++, itP ++, idx ++) {

        if (status[idx] == 0 || err[idx] > 20 || outOfImg(*itC + convert, cv::Size(lastGray.cols, lastGray.rows))) {
//            status_choose[idx]=0;
        } else {
            cv::Point2f cfp=(*itC + convert) * ThreadContext::DOWNSAMPLE_SCALE;
            cv::Point2f lfp=(*itP + convert) * ThreadContext::DOWNSAMPLE_SCALE;
            curFeaturesTmp.push_back(cfp);
            lastFeaturesTmp.push_back(lfp);
            int a = in_area(cfp, half_w, half_h);
            point_num[a]++;
        }
    }
    track_thread_over_num_--;
    if(!track_thread_over_num_){
        is_wait_over.notify_one();
    }
    track_lock.unlock();
}

void HomoExtractor::calcul_Homo(std::vector<char> &ifselect, int niter, int type) {
    H = cv::Mat();
    ifselect.clear();
    if(lastFeaturesTmp.size() < 6)
    {
        H = cv::Mat::eye(3, 3, CV_64F);
    }
    else if(niter<0)
    {
        H = findHomography(lastFeaturesTmp, curFeaturesTmp, 0);
    }
    else
    {
        if (!lastFeaturesTmp.empty() && !curFeaturesTmp.empty() && lastFeaturesTmp.size() > 3) {

            if(type==0)
            {
                H = findHomography(lastFeaturesTmp, curFeaturesTmp, cv::LMEDS, 3, ifselect, niter, 0.995);
            }
            else
            {
                H = findHomography(lastFeaturesTmp, curFeaturesTmp, cv::RHO, 4, ifselect, niter, 0.995);
            }

        }
        /*else if(!lastFeaturesTmp.empty() && !curFeaturesTmp.empty() && lastFeaturesTmp.size() > 1)
        {
            if(lastFeaturesTmp.size()==3)
            {
                cv::Mat AF=getAffineTransform(lastFeaturesTmp,curFeaturesTmp);
                H=cv::Mat::zeros(3,3,AF.type());

                H.at<double>(0,0)=AF.at<double>(0,0);
                H.at<double>(0,1)=AF.at<double>(0,1);
                H.at<double>(0,2)=AF.at<double>(0,2);
                H.at<double>(1,0)=AF.at<double>(1,0);
                H.at<double>(1,1)=AF.at<double>(1,1);
                H.at<double>(1,2)=AF.at<double>(1,2);
                H.at<double>(2,0)=0;
                H.at<double>(2,1)=0;
                H.at<double>(2,2)=1;

            }
            else if(lastFeaturesTmp.size()==2)
            {
                std::vector<cv::Point2f> curFeat_2, lastFeat_2;
                curFeat_2.push_back(curFeaturesTmp[0]);
                lastFeat_2.push_back(lastFeaturesTmp[0]);
                curFeat_2.push_back(curFeaturesTmp[1]);
                lastFeat_2.push_back(lastFeaturesTmp[1]);

                cv::Point2f cp(curFeaturesTmp[1].x-(curFeaturesTmp[1].y-curFeaturesTmp[0].y),curFeaturesTmp[1].y-(curFeaturesTmp[1].x-curFeaturesTmp[0].x));
                cv::Point2f lp(lastFeaturesTmp[1].x-(lastFeaturesTmp[1].y-lastFeaturesTmp[0].y),lastFeaturesTmp[1].y-(lastFeaturesTmp[1].x-lastFeaturesTmp[0].x));
                curFeat_2.push_back(cp);
                lastFeat_2.push_back(lp);

                cv::Mat AF = getAffineTransform(lastFeat_2, curFeat_2);
                H=cv::Mat::zeros(3,3,AF.type());

                H.at<double>(0, 0) = AF.at<double>(0, 0);
                H.at<double>(0, 1) = AF.at<double>(0, 1);
                H.at<double>(0, 2) = AF.at<double>(0, 2);
                H.at<double>(1, 0) = AF.at<double>(1, 0);
                H.at<double>(1, 1) = AF.at<double>(1, 1);
                H.at<double>(1, 2) = AF.at<double>(1, 2);
                H.at<double>(2, 0) = 0;
                H.at<double>(2, 1) = 0;
                H.at<double>(2, 2) = 1;
            }
        }*/
    }
    if(H.cols == 0 || H.rows == 0)
    {
        LOGI("An error H");
        H = cv::Mat::eye(3, 3, CV_64F);
    }

}

cv::Point2f HomoExtractor::goround(cv::Point2f p1, cv::Point2f p0, double degree) {
    double ro = PI / 180 * degree;
    double xr=(p1.x-p0.x)*cos(ro) - (p1.y-p0.y)*sin(ro) + p0.x;
    double yr=(p1.x-p0.x)*sin(ro) + (p1.y-p0.y)*cos(ro) + p0.y;
    cv::Point2f pr(xr,yr);
    return pr;
}

cv::Point2f HomoExtractor::goscale(cv::Point2f p1, cv::Point2f p0, double scale) {
    cv::Point2f pr=p0+(p1-p0)*scale;
    return pr;
}

double HomoExtractor::vec_cos(cv::Point2f s, cv::Point2f e1, cv::Point2f e2) {
    cv::Point2f v1=e1-s, v2=e2-s;
    cv::Point2f o(0,0);
    double l1=point_distance(o,v1), l2=point_distance(o,v2);
    double y = (v1.x * v2.x + v1.y * v2.y)/(l1 * l2);

    return y;
}

double HomoExtractor::calcul_H_error(int c_h, int c_w) {
    //LOGI("step7_1");
    int height=100;
    int width=100;
    double error_rate=1.1;
    bool aff_adj=false;

    cv::Mat  vertex=(cv::Mat_<double>(3, 4)<<c_w-width, c_w-width, c_w+width, c_w+width, c_h-height, c_h+height, c_h+height, c_h-height, 1.0, 1.0, 1.0, 1.0);

    //LOGI("step7_2");
    cv::Point2f cp1, cp2, cp3, cp4;
    std::vector<cv::Point2f> l, c;

    //LOGI("step7_3");
    /*LOGI("H in calcul_H_error:[%f, %f, %f; %f, %f, %f; %f, %f, %f]",
         H.at<double>(0,0), H.at<double>(0,1), H.at<double>(0,2),
         H.at<double>(1,0), H.at<double>(1,1), H.at<double>(1,2),
         H.at<double>(2,0), H.at<double>(2,1), H.at<double>(2,2));*/
    cv::Mat newvertex = H * vertex;
    //LOGI("step7_4");
    cp1.x = newvertex.at<double>(0, 0) / newvertex.at<double>(2, 0);
    cp1.y = newvertex.at<double>(1, 0) / newvertex.at<double>(2, 0);

    cp2.x = newvertex.at<double>(0, 1) / newvertex.at<double>(2, 1);
    cp2.y = newvertex.at<double>(1, 1) / newvertex.at<double>(2, 1);

    cp3.x = newvertex.at<double>(0, 2) / newvertex.at<double>(2, 2);
    cp3.y = newvertex.at<double>(1, 2) / newvertex.at<double>(2, 2);

    cp4.x = newvertex.at<double>(0, 3) / newvertex.at<double>(2, 3);
    cp4.y = newvertex.at<double>(1, 3) / newvertex.at<double>(2, 3);
    //LOGI("step7_5");

    double c_x=(cp1.x + cp2.x + cp3.x + cp4.x)/4;
    double c_y=(cp1.y + cp2.y + cp3.y + cp4.y)/4;
    cv::Point2f pc(c_x,c_y);

    cv::Point2f lp1(c_x-width, c_y-height),lp2(c_x-width, c_y+height),lp3(c_x+width, c_y+height),lp4(c_x+width, c_y-height);
    double mindis=9999;
    double mind=0;
    //LOGI("step7_6");

    for(double d=-10; d<=10; d++)
    {
        cv::Point2f rp1=goround(lp1,pc,d);
        cv::Point2f rp2=goround(lp2,pc,d);
        cv::Point2f rp3=goround(lp3,pc,d);
        cv::Point2f rp4=goround(lp4,pc,d);

        double l1 = point_distance(cp1,rp1);
        double l2 = point_distance(cp2,rp2);
        double l3 = point_distance(cp3,rp3);
        double l4 = point_distance(cp4,rp4);

        if(l1+l2+l3+l4<mindis)
        {
            mindis=l1+l2+l3+l4;
            mind=d;
        }
    }
    //LOGI("step7_7");

    double mindis2=9999;
    double mins=0;
    cv::Point2f rp1=goround(lp1,pc,mind);
    cv::Point2f rp2=goround(lp2,pc,mind);
    cv::Point2f rp3=goround(lp3,pc,mind);
    cv::Point2f rp4=goround(lp4,pc,mind);
    for(double s=0.9; s<=1.1; s=s+0.01)
    {
        cv::Point2f sp1=goscale(rp1,pc,s);
        cv::Point2f sp2=goscale(rp2,pc,s);
        cv::Point2f sp3=goscale(rp3,pc,s);
        cv::Point2f sp4=goscale(rp4,pc,s);

        double l1 = point_distance(cp1,sp1);
        double l2 = point_distance(cp2,sp2);
        double l3 = point_distance(cp3,sp3);
        double l4 = point_distance(cp4,sp4);

        if(l1+l2+l3+l4<mindis2)
        {
            mindis2=l1+l2+l3+l4;
            mins=s;
        }
    }
    //LOGI("step7_8");

    if(true)
    {
        double d1=vec_cos(cp1,cp2,cp4);
        double d2=vec_cos(cp2,cp1,cp3);
        double d3=vec_cos(cp3,cp2,cp4);
        double d4=vec_cos(cp4,cp1,cp3);

        stable_move=true;
        stable_move2=false;
        if((d1<0 && d2<0) || (d2<0 && d3<0) || (d3<0 && d4<0) || (d4<0 && d1<0))
        {
            stable_move=false;
        }
        double limit2=0.015;
        if((d1>limit2 && d3>limit2) || (d2>limit2 && d4>limit2))
        {
            stable_move2=true;
        }
    }
    //LOGI("step7_9");

    if(true)
    {
        cp1=goround(lp1,pc,mind);
        cp2=goround(lp2,pc,mind);
        cp3=goround(lp3,pc,mind);
        cp4=goround(lp4,pc,mind);

        cp1=goscale(cp1,pc,mins);
        cp2=goscale(cp2,pc,mins);
        cp3=goscale(cp3,pc,mins);
        cp4=goscale(cp4,pc,mins);

        lp1=cv::Point2f(c_w-width, c_h-height);
        lp2=cv::Point2f(c_w-width, c_h+height);
        lp3=cv::Point2f(c_w+width, c_h+height);
        lp4=cv::Point2f(c_w+width, c_h-height);

        std::vector<cv::Point2f> lpt,cpt;
        lpt.push_back(lp1);
        lpt.push_back(lp2);
        lpt.push_back(lp3);
        lpt.push_back(lp4);

        cpt.push_back(cp1);
        cpt.push_back(cp2);
        cpt.push_back(cp3);
        cpt.push_back(cp4);

        second_H = findHomography(lpt, cpt, 0);
    }
    //LOGI("step7_10");

    return mindis;

    /*int height=100;
    int width=100;
    double error_rate=1.1;
    bool aff_adj=false;

    cv::Mat  vertex=(cv::Mat_<double>(3, 4)<<c_w-width, c_w-width, c_w+width, c_w+width, c_h-height, c_h+height, c_h+height, c_h-height, 1.0, 1.0, 1.0, 1.0);

    cv::Mat perp, sca, shear, rot, trans;
    cv::Point2f center(c_w, c_h);
    std::cout<<"("<<c_w<<","<<c_h<<")"<<std::endl;
    decomposeHomo2(H, center, perp, sca, shear, rot, trans);
    cv::Mat simH = trans * rot * sca;

    cv::Point2f cp1, cp2, cp3, cp4;
    std::vector<cv::Point2f> l, c;

    cv::Mat newvertex = H * vertex;
    cp1.x = newvertex.at<double>(0, 0) / newvertex.at<double>(2, 0);
    cp1.y = newvertex.at<double>(1, 0) / newvertex.at<double>(2, 0);

    cp2.x = newvertex.at<double>(0, 1) / newvertex.at<double>(2, 1);
    cp2.y = newvertex.at<double>(1, 1) / newvertex.at<double>(2, 1);

    cp3.x = newvertex.at<double>(0, 2) / newvertex.at<double>(2, 2);
    cp3.y = newvertex.at<double>(1, 2) / newvertex.at<double>(2, 2);

    cp4.x = newvertex.at<double>(0, 3) / newvertex.at<double>(2, 3);
    cp4.y = newvertex.at<double>(1, 3) / newvertex.at<double>(2, 3);

    double c_x=(cp1.x + cp2.x + cp3.x + cp4.x)/4;
    double c_y=(cp1.y + cp2.y + cp3.y + cp4.y)/4;
    cv::Point2f pc(c_x,c_y);


    if(true)
    {
        double d1=vec_cos(cp1,cp2,cp4);
        double d2=vec_cos(cp2,cp1,cp3);
        double d3=vec_cos(cp3,cp2,cp4);
        double d4=vec_cos(cp4,cp1,cp3);

        stable_move=true;
        stable_move2=false;
        if((d1<0 && d2<0) || (d2<0 && d3<0) || (d3<0 && d4<0) || (d4<0 && d1<0))
        {
            stable_move=false;
        }
        double limit2=0.015;
        if((d1>limit2 && d3>limit2) || (d2>limit2 && d4>limit2))
        {
            stable_move2=true;
        }
    }

    if(true)
    {

        second_H = trans * rot * sca;
    }

    cv::Point2f sp1, sp2, sp3, sp4;

    newvertex = second_H * vertex;
    sp1.x = newvertex.at<double>(0, 0) / newvertex.at<double>(2, 0);
    sp1.y = newvertex.at<double>(1, 0) / newvertex.at<double>(2, 0);

    sp2.x = newvertex.at<double>(0, 1) / newvertex.at<double>(2, 1);
    sp2.y = newvertex.at<double>(1, 1) / newvertex.at<double>(2, 1);

    sp3.x = newvertex.at<double>(0, 2) / newvertex.at<double>(2, 2);
    sp3.y = newvertex.at<double>(1, 2) / newvertex.at<double>(2, 2);

    sp4.x = newvertex.at<double>(0, 3) / newvertex.at<double>(2, 3);
    sp4.y = newvertex.at<double>(1, 3) / newvertex.at<double>(2, 3);

    double l1 = point_distance(cp1,sp1);
    double l2 = point_distance(cp2,sp2);
    double l3 = point_distance(cp3,sp3);
    double l4 = point_distance(cp4,sp4);

    double mindis=l1+l2+l3+l4;

    return mindis;*/
}

cv::Mat HomoExtractor::extractHomo(cv::Mat &img1, cv::Mat &img2) {
    //LOGI("step0_1");
    std::unique_lock<std::mutex> main_lock(mutex_);
//    cv::Mat img2_r;
    curGray = img2.rowRange(0,img2.rows * 2 / 3);
    cv::resize(curGray, curGray, cv::Size(curGray.cols / ThreadContext::DOWNSAMPLE_SCALE, curGray.rows / ThreadContext::DOWNSAMPLE_SCALE));
//    cv::GaussianBlur(curGray, curGray, cv::Size(15, 15), 0, 0);

    cv::Rect rect_lu(0, 0, curGray.cols/2, curGray.rows/2);
    cv::Rect rect_ru(curGray.cols/2, 0, curGray.cols/2, curGray.rows/2);
    cv::Rect rect_ld(0, curGray.rows/2, curGray.cols/2, curGray.rows/2);
    cv::Rect rect_rd(curGray.cols/2, curGray.rows/2, curGray.cols/2, curGray.rows/2);
    //LOGI("step0_2");
    if(ex_index_ == 0){
//        cv::Mat img1_r;
        lastGray = img1.rowRange(0,img1.rows * 2 / 3);
        cv::resize(lastGray, lastGray, cv::Size(lastGray.cols / ThreadContext::DOWNSAMPLE_SCALE, lastGray.rows / ThreadContext::DOWNSAMPLE_SCALE));
//        cv::GaussianBlur(lastGray, lastGray, cv::Size(15, 15), 0, 0);

        last_lu_ = lastGray(rect_lu);
        last_ru_ = lastGray(rect_ru);
        last_ld_ = lastGray(rect_ld);
        last_rd_ = lastGray(rect_rd);
        detect_thread0_ = thread(&HomoExtractor::detectFeature, this, last_lu_, 0);
        detect_thread1_ = thread(&HomoExtractor::detectFeature, this, last_ru_, 1);
        detect_thread2_ = thread(&HomoExtractor::detectFeature, this, last_ld_, 2);
        detect_thread3_ = thread(&HomoExtractor::detectFeature, this, last_rd_, 3);
//        detect_thread0_.join();
//        detect_thread1_.join();
//        detect_thread2_.join();
//        detect_thread3_.join();
    }
    is_wait_over.wait(main_lock);
    //LOGI("step0_3");

    cur_lu_ = curGray(rect_lu);
    cur_ru_ = curGray(rect_ru);
    cur_ld_ = curGray(rect_ld);
    cur_rd_ = curGray(rect_rd);

    statussize = 0;
    curFeaturesTmp.clear();
    lastFeaturesTmp.clear();
    for(int j=0; j<16; j++)
    {
        point_num[j] = 0;
    }
    //LOGI("step0_4");
    track_thread0_ = thread(&HomoExtractor::trackFeature, this, img1.rowRange(0,img2.rows * 2 / 3),
            img2.rowRange(0,img2.rows * 2 / 3),last_lu_, cur_lu_, 0);
    track_thread1_ = thread(&HomoExtractor::trackFeature, this, img1.rowRange(0,img2.rows * 2 / 3),
                            img2.rowRange(0,img2.rows * 2 / 3),last_ru_, cur_ru_, 1);
    track_thread2_ = thread(&HomoExtractor::trackFeature, this, img1.rowRange(0,img2.rows * 2 / 3),
                            img2.rowRange(0,img2.rows * 2 / 3),last_ld_, cur_ld_, 2);
    track_thread3_ = thread(&HomoExtractor::trackFeature, this, img1.rowRange(0,img2.rows * 2 / 3),
                            img2.rowRange(0,img2.rows * 2 / 3),last_rd_, cur_rd_, 3);
//    track_thread0_.join();
//    track_thread1_.join();
//    track_thread2_.join();
//    track_thread3_.join();
    is_wait_over.wait(main_lock);
    //LOGI("step0_5");

    bool re = judge_area();
    if(re){
        auto last_add_feature = detectCertainFeature(block_index_);
        if(last_add_feature.size() != 0){
            trackCertainFeature(last_add_feature);
        }
    }
    LOGI("point_num:[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d], %d",
         point_num[0], point_num[1], point_num[2], point_num[3],
         point_num[4], point_num[5], point_num[6], point_num[7],
         point_num[8], point_num[9], point_num[10], point_num[11],
         point_num[12], point_num[13], point_num[14], point_num[15], re);

    //LOGI("step0_6");
    std::vector<char> ifselect;
    calcul_Homo(ifselect,2000,0);

    //LOGI("Point size : %d; H Matinthread: %f, %f, %f, %f, %f, %f, %f, %f, %f", lastFeaturesTmp.size(), H.at<double>(0,0), H.at<double>(0,1), H.at<double>(0,2), H.at<double>(1,0), H.at<double>(1,1), H.at<double>(1,2), H.at<double>(2,0), H.at<double>(2,1), H.at<double>(2,2));

    int height=img1.rows;
    int width=img1.cols;
    bool isValidH = !H.empty();
    int half_w=img1.cols/4 , half_h=img1.rows/4;
    int tmp_count[16]={0};
    double ifinliner = 0;

    //LOGI("step0_7");
    for(int i=0; i<ifselect.size(); i++)
    {
        if(ifselect[i] == 1)
        {
            ifinliner++;
        }
    }

    //LOGI("step0_7_1");
    double ifinrate1=ifinliner/statussize;
    double ifinrate2=ifinliner/ifselect.size();
    bool re2 = false;
    //LOGI("step0_7_2");
    re2= judge_recal_simple(img1, ifselect);
    //LOGI("step0_7_3");
    double h_err = calcul_H_error(height/2, width/2);
    //LOGI("step0_7_4");
    //LOGI("H Matinthread h_err:%f", h_err);

    //LOGI("step0_75");
    cv::Mat perp, sca, shear, rot, trans;
    cv::Point2f center(width/2, height/2);
    decomposeHomo(H, center, perp, sca, shear, rot, trans);
    cv::Mat M1 = (cv::Mat_<double>(3, 3) << 1, 0, -center.x, 0, 1, -center.y, 0, 0, 1);
    cv::Mat M2 = (cv::Mat_<double>(3, 3) << 1, 0, center.x, 0, 1, center.y, 0, 0, 1);

    if(h_err > 30 || re2) {
        H = trans * rot * last_shear * sca * last_perp;
    } else{
        last_perp = perp.clone();
        last_shear = shear.clone();
    }

    //LOGI("step0_8");
    if(draw_information){
        int tsize=curFeaturesTmp.size()>lastFeaturesTmp.size() ? lastFeaturesTmp.size() : curFeaturesTmp.size();
        for(int i = 0; i < tsize; i++){
            cv::Point2f p1 = lastFeaturesTmp[i];
            cv::Point2f p2 = curFeaturesTmp[i];
            cv::circle(img2,p1,10,cv::Scalar(255,255,0),2);
            cv::circle(img2,p2,10,cv::Scalar(40,0,0),4);
            cv::line(img2,p1,p2,cv::Scalar(0,0,255),8);
        }
        std::string num = "f_nums:";
        num  = num + std::to_string(curFeaturesTmp.size());
    }

    //LOGI("step0_9");
    threads::ThreadContext::feature_by_r_.push(curFeaturesTmp);
    LOGI("curFeaturesTmp size:%d", curFeaturesTmp.size());
    curGray.copyTo(lastGray);
    ex_index_++;
    if(ex_index_ == ThreadContext::SEGSIZE){
        ex_index_ = 0;
    }

    detect_thread0_.join();
    detect_thread1_.join();
    detect_thread2_.join();
    detect_thread3_.join();
    track_thread0_.join();
    track_thread1_.join();
    track_thread2_.join();
    track_thread3_.join();

    //LOGI("step0_10");
    return H;


}

void HomoExtractor::setDrawStatus(bool is_draw) {
    draw_information = is_draw;
}

void HomoExtractor::decomposeHomo(cv::Mat h, Point2f cen, cv::Mat &perp, cv::Mat &sca,
                                  cv::Mat &shear, cv::Mat &rot, cv::Mat &trans) {
    double hm[3][3];
    double fax, fay, tx, ty, sh, theta, lamx, lamy;
    double cx = cen.x;
    double cy = cen.y;
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            hm[i][j] = h.at<double>(i,j)/h.at<double>(2,2);
        }
    }

    fax = hm[2][0]/(cx*hm[2][0] + cy*hm[2][1] + 1);
    fay = hm[2][1]/(cx*hm[2][0] + cy*hm[2][1] + 1);
    cout << "fax: " << fax << " fay: " << fay << endl;

    double n = 1 - cx*fax - cy*fay;

    double ctx = n*hm[0][2] + cx*n*hm[0][0] + cy*n*hm[0][1];
    double cty = n*hm[1][2] + cx*n*hm[1][0] + cy*n*hm[1][1];
    cout << "n: " << n << " ctx: " << ctx << " cty: " << cty << endl;

    tx = ctx - cx;
    ty = cty - cy;

    double xc = n*hm[0][0] - fax*n*hm[0][2] - cx*fax*n*hm[0][0] - cy*fax*n*hm[0][1];
    double xs = n*hm[1][0] - fax*n*hm[1][2] - cx*fax*n*hm[1][0] - cy*fax*n*hm[1][1];
    double ysmc = - n*hm[0][1] + fay*n*hm[0][2] + cx*fay*n*hm[0][0] + cy*fay*n*hm[0][1];
    double ycps = n*hm[1][1] - fay*n*hm[1][2] - cx*fay*n*hm[1][0] - cy*fay*n*hm[1][1];
    cout << "xc: " << xc << " xs: " << xs << " ysmc: " << ysmc << " ycps: " << ycps << endl;
    if(xs == 0)
        xs=1e-8;

    sh = -(xc*ysmc - xs*ycps)/(xc*ycps + xs*ysmc);
    double z = xc*xc + xs*xs;
    double theta1 = -2*atan( (xc+sqrt(z)) / xs );
    double theta2 = -2*atan( (xc-sqrt(z)) / xs );
    if(xs == 0)
    {
        theta1 = 0;
        theta2 = 0;
    }
    if(abs(theta1)<abs(theta2))
    {
        theta = theta1;
        lamx = -sqrt(z);
        lamy = (xc*xc*ycps - xc*ycps*(xc+sqrt(z)) - xs*ysmc*(xc+sqrt(z)) + xc*xs*ysmc) / z;
    }
    else
    {
        theta = theta2;
        lamx = sqrt(z);
        lamy = (xc*xc*ycps - xc*ycps*(xc-sqrt(z)) - xs*ysmc*(xc-sqrt(z)) + xc*xs*ysmc) / z;
    }

    cout << "theta: " << theta << " lamx: " << lamx << " lamy: " << lamy << endl;

    Mat M1=(cv::Mat_<double>(3, 3) <<
                                   1.0, 0.0, -cx,
            0.0, 1.0, -cy,
            0.0, 0.0, 1.0);

    Mat M2=(cv::Mat_<double>(3, 3) <<
                                   1.0, 0.0, cx,
            0.0, 1.0, cy,
            0.0, 0.0, 1.0);

    Mat PE=(cv::Mat_<double>(3, 3) <<
                                   1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            fax, fay, 1.0);

    Mat SC=(cv::Mat_<double>(3, 3) <<
                                   lamx, 0.0, 0.0,
            0.0, lamy, 0.0,
            0.0, 0.0, 1.0);

    Mat SH=(cv::Mat_<double>(3, 3) <<
                                   1.0, sh, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0);

    Mat RO=(cv::Mat_<double>(3, 3) <<
                                   cos(theta), -sin(theta), 0.0,
            sin(theta), cos(theta), 0.0,
            0.0, 0.0, 1.0);

    Mat TR=(cv::Mat_<double>(3, 3) <<
                                   1.0, 0.0, tx,
            0.0, 1.0, ty,
            0.0, 0.0, 1.0);

    perp = M2 * PE * M1;
    perp = perp / perp.at<double>(2,2);
    sca = M2 * SC * M1;
    shear = M2 * SH * M1;
    rot = M2 * RO * M1;
    trans = TR.clone();
}

//计算两个向量之间的夹角，-pi到pi
double HomoExtractor::cal_degree(cv::Point2f vec1, cv::Point2f vec2)
{
    cv::Point2f o(0,0);
    double d,vd1,vd2,cross1,cross2;
    vd1 = point_distance(o,vec1);
    vd2 = point_distance(o,vec2);

    cross1 = vec1.x * vec2.x + vec1.y * vec2.y;
    cross2 = vec1.x * vec2.y - vec1.y * vec2.x;

    //LOGI("acos: %f", cross1 / (vd1 * vd2));
    //LOGI("vd1&2: %f", (vd1 * vd2));
    if((vd1 * vd2)==0)
    {
        return 2*PI;
    }

    double c = cross1 / (vd1 * vd2);
    if(c >= 1.0){
        d = 0;
    }else if (c <= -1.0){
        d = PI;
    }else {
        d = acos(cross1 / (vd1 * vd2));
    }
    if(cross2 < 0)
    {
        d = -d;
    }

    return d;
}