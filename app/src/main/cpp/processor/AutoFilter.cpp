//
// Created by ShiJJ on 2020/6/15.
//

#include "AutoFilter.h"
#include "BSpline.h"
#define LOG_TAG    "c_AutoFilter"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static std::ofstream file_w("/data/data/me.zhehua.gryostable/data/window.txt");
static int frame_count = 1;
static std::ofstream file_w1("/data/data/me.zhehua.gryostable/data/middle.txt");
static int frame_count_m = 1;
static bool is_first = true;
int AutoFilter::predict_num_ = 5;
AutoFilter::AutoFilter(int max_size, double sigma)
:max_size_(max_size), sigma_(sigma)
{
    std::cout << max_size_ <<" " <<sigma_ << std::endl;
    weight_vec_.clear();
    weight_vec_.resize(max_size_);
    for (int i = 0; i < max_size_; i ++) {
        double k=i-delay_num_;
        double weight = filterWeight(k, sigma);
        weight_vec_[i] = weight;
    }

    size_ = cv::Size(1920, 1080);
    center.x = size_.width / 2;
    center.y = size_.height / 2;
    keep_num = 0;
    keep_rate = 1.0;
    mode = 1;
    move_status = 1;
    first_in = true;
    cum_H = cv::Mat::eye(3, 3, CV_64F);
    vertex_ = (cv::Mat_<double>(3, 4)<<0.0,0.0,size_.width-1,size_.width-1,0.0,size_.height-1,size_.height-1,0.0,1.0,1.0,1.0,1.0);
    double cropw = crop_rate_ * size_.width;
    double croph = crop_rate_ * size_.height;
    double mw = (size_.width - cropw) / 2;
    double mh = (size_.height - croph) / 2;
    cropvertex_ = (cv::Mat_<double>(3, 4)<<mw,mw,cropw+mw,cropw+mw,mh,croph+mh,croph+mh,mh,1.0,1.0,1.0,1.0);

    for(int i=0; i<predict_num_; i++)
    {
        f_num_[i] = i+1;
    }
    num_que_ = 0;
}

void AutoFilter::set_move_status(int m){
    move_status = m;
}

bool AutoFilter::push(cv::Mat goodar) {
    if (goodar.empty()) {   //最后一帧，结束
        int target=max_size_/2;
        while (input_buffer_.size()>max_size_/2) {
            putIntoWindow(target);
            input_buffer_.pop_front();
            trans_buffer_.pop_front();
            sca_buffer_.pop_front();
        }
        return true;
    }
    //trans_pre = trans_now.clone();
    //trans_now = goodar.clone();
    input_buffer_.push_back(goodar);
    input_trans_buffer_.push_back(goodar);

    cv::Mat perp, sca, shear, rot, trans;
    decomposeHomo(goodar, center, perp, sca, shear, rot, trans);
    trans_mat_buffer_.push_back(trans);
    cv::Point2f t(trans.at<double>(0,2), trans.at<double>(1,2));
    trans_buffer_.push_back(t);
    cv::Mat M1 = (cv::Mat_<double>(3, 3) << 1, 0, - center.x, 0, 1, - center.y, 0, 0, 1);
    cv::Mat M2 = (cv::Mat_<double>(3, 3) << 1, 0, center.x, 0, 1, center.y, 0, 0, 1);
    sca = M2.inv() * sca * M1.inv();
    cv::Point2f s(sca.at<double>(0,0), sca.at<double>(1,1));
    sca_buffer_.push_back(s);

    if(input_buffer_.size() < max_size_){   //刚开始放入数据
        if(input_buffer_.size() >= delay_num_){
            int target = input_buffer_.size() - (delay_num_);
            int offset = max_size_ - input_buffer_.size();
            if(putIntoWindow(target, offset)){
                return true;
            } else {
                return false;
            }

        } else{
            return false;
        }
    } else {    //正常处理
        int target = input_buffer_.size() - delay_num_;
        if(putIntoWindow(target)){
            input_buffer_.pop_front();
            trans_buffer_.pop_front();
            sca_buffer_.pop_front();
            trans_mat_buffer_.pop_front();

            return true;
        } else {
            input_buffer_.pop_front();
            trans_buffer_.pop_front();
            sca_buffer_.pop_front();
            trans_mat_buffer_.pop_front();

            return false;
        }

    }

}

cv::Mat AutoFilter:: pop() {
    if(!output_buffer_.empty()){
        cv::Mat ret_mat = output_buffer_.front().clone();
        output_buffer_.pop();
        return ret_mat;
    }
    return cv::Mat();

}

bool AutoFilter::putIntoWindow(int target, int offset) {
    std::cout << "target:" << target << std::endl;
    window_.clear();
    window_.resize(input_buffer_.size());
    window_[target] = cv::Mat::eye(3, 3, CV_64F);
    for(int i = target - 1; i >= 0; i--){
        window_[i] = input_buffer_[i].inv() * window_[i + 1];
    }
    for(int i = target + 1; i < input_buffer_.size(); i++){
        window_[i] = input_buffer_[i - 1] * window_[i - 1];
    }
    double sum_weight = 0;
    cv::Mat ret_mat = cv::Mat::zeros(3, 3, CV_64F);
    for(int i = 0; i < window_.size(); i++) {
        ret_mat += (weight_vec_[i + offset] * window_[i]);
        sum_weight += weight_vec_[i + offset];
    }
    ret_mat /= sum_weight;

    window_.clear();
    window_.resize(trans_mat_buffer_.size());
    window_[target] = cv::Mat::eye(3, 3, CV_64F);
    for(int i = 0; i < trans_mat_buffer_.size(); i++){
        window_[i] = trans_mat_buffer_[i];
    }
    double t_sum_weight = 0;
    cv::Mat t_ret_mat = cv::Mat::zeros(3, 3, CV_64F);
    for(int i = 0; i < window_.size(); i++) {
        t_ret_mat += (weight_vec_[i + offset] * window_[i]);
        t_sum_weight += weight_vec_[i + offset];
    }
    t_ret_mat /= t_sum_weight;
    trans_for_cumh = t_ret_mat.clone();
    //trans_for_cumh.at<double>(0,2) = -trans_for_cumh.at<double>(0,2);
    //trans_for_cumh.at<double>(1,2) = -trans_for_cumh.at<double>(1,2);

    if(trans_buffer_.size() > max_size_/2 + 1)
    {
        analyzeTrans(ret_mat);
    }

    ex_count++;
    if(true)
    {
        processCrop(ret_mat, size_);
    } else
    {
        output_buffer_.push(ret_mat);
    }

    if(ex_count < predict_num_){
        return false;
    }
    return true;

}


bool AutoFilter::isInside(cv::Mat cropvertex, cv::Mat newvertex) {
    bool aInside = true;
    for( int i = 0 ; i < 4 ; i++ )
    {
        for( int j = 0 ; j < 4 ; j++ )
        {
            cv::Point2f vec1 , vec2;
            vec1.x=float(newvertex.at<double>(0,j)-cropvertex.at<double>(0,i));
            vec1.y=float(newvertex.at<double>(1,j)-cropvertex.at<double>(1,i));
            vec2.x=float(newvertex.at<double>(0,(j+1)%4)-newvertex.at<double>(0,j));
            vec2.y=float(newvertex.at<double>(1,(j+1)%4)-newvertex.at<double>(1,j));

            // vec1 = pt_transform[j] - pt_crop[i];
            // vec2 = pt_transform[(j+1)%4] - pt_transform[j];
            float cross_product = vec1.x * vec2.y - vec2.x * vec1.y;
            //  NSLog(@"%f",cross_product);
            if( cross_product > 0 )
            {
                aInside = false;
                break;
            }
        }
        if( !aInside )
        {
            break;
        }
    }
    return aInside;
}

void AutoFilter::processCrop(const cv::Mat &comp, const cv::Size &size) {
    s_mat_.push(comp.clone());
    cv::Point2d crop_window[4];
    cv::Mat stable_mat = comp.clone();
    cv::Mat box = stable_mat.inv() * cropvertex_;
    crop_window[0].x=box.at<double>(0,0)/box.at<double>(2,0);
    crop_window[0].y=box.at<double>(1,0)/box.at<double>(2,0);

    crop_window[1].x=box.at<double>(0,1)/box.at<double>(2,1);
    crop_window[1].y=box.at<double>(1,1)/box.at<double>(2,1);

    crop_window[2].x=box.at<double>(0,2)/box.at<double>(2,2);
    crop_window[2].y=box.at<double>(1,2)/box.at<double>(2,2);

    crop_window[3].x=box.at<double>(0,3)/box.at<double>(2,3);
    crop_window[3].y=box.at<double>(1,3)/box.at<double>(2,3);

    double xmin = 0 - crop_window[0].x ;
    double xmax = size.width - crop_window[0].x ;
    double ymin = 0 - crop_window[0].y ;
    double ymax = size.height - crop_window[0].y ;

    for(int j=1; j<4; j++)
    {
        if(xmin < 0 - crop_window[j].x)
            xmin = 0 - crop_window[j].x;
        if(xmax > size.width - crop_window[j].x)
            xmax = size.width - crop_window[j].x;
        if(ymin < 0 - crop_window[j].y)
            ymin = 0 - crop_window[j].y;
        if(ymax > size.height - crop_window[j].y)
            ymax = size.height - crop_window[j].y;
    }
    limit temp_limit;
    temp_limit.xmin = xmin;
    temp_limit.xmax = xmax;
    temp_limit.ymin = ymin;
    temp_limit.ymax = ymax;
    limit_que_.push(temp_limit);    //将上下限放入队列
    LOGI("temp_limit:%d, %f, %f, %f, %f", frame_count, temp_limit.xmin, temp_limit.xmax,
         temp_limit.ymin, temp_limit.ymax);

    double trans_x[predict_num_/2], trans_y[predict_num_/2];
    double x_m, y_m;

    if(num_que_ < predict_num_) {
        num_que_++;
    }
    x_m = cur_x;
    y_m = cur_y;
    double x_d = xmax - xmin;
    double y_d = ymax - ymin;
    if(ex_count % 30 == 0){
        cur_x = 0;
        cur_y = 0;
    }
    if(x_m < xmin){
        LOGI("touch border");
        if(mode == 2)
        {
            mode = 21;
        }
        x_m = xmin + x_d /4;
        cur_x = x_m;
    } else if(x_m > xmax){
        LOGI("touch border");
        if(mode == 2)
        {
            mode = 21;
        }
        x_m = xmax - x_d / 4;
        cur_x = x_m;
    }
    if(y_m < ymin){
        LOGI("touch border");
        if(mode == 2)
        {
            mode = 21;
        }
        y_m = ymin + y_d /4;
        cur_y = y_m;
    } else if(y_m > ymax){
        LOGI("touch border");
        if(mode == 2)
        {
            mode = 21;
        }
        y_m = ymax - y_d/4;
        cur_y = y_m;
    }   //边界检查

    if(write_status_){
        file_w << frame_count <<" " << xmin << " " << xmax << " " << ymin << " " << ymax << " "
               << x_m << " " << y_m <<std::endl;
        frame_count++;
    }

    queue_in(que_x_, num_que_ - 1, x_m);    //将xy插入队列
    queue_in(que_y_, num_que_ - 1, y_m);
    if(num_que_ < predict_num_){
        return ;
    } else {
        if(is_first){
            cv::Mat temp = s_mat_.front().inv();
            s_mat_.pop();
            output_buffer_.push(temp);
            limit_que_.pop();
            is_first = false;
        }
        double result_x_3[predict_num_/2], result_y_3[predict_num_/2];
        Bspline bspline_x, bspline_y;   //B样条曲线
        for(int i = 0; i < predict_num_; i+=2){
            bspline_x.push(cv::Point2d(index_, que_x_[i]));
            bspline_y.push(cv::Point2d(index_, que_y_[i])); //将xy放入B样条中
            index_ += 2;
        }
        bspline_x.calControlPoint();
        bspline_y.calControlPoint();    //计算控制点
        double ratio = 0.25;
        for(int i = 0; i < predict_num_/2; i++){
            result_x_3[i] = bspline_x.genInterpolationPoint(ratio).y;
            result_y_3[i] = bspline_y.genInterpolationPoint(ratio).y;   //计算插值
            ratio += 0.25;
        }
        for(int i = 0; i < predict_num_/2; i++){
            trans_x[i] = result_x_3[i];
            trans_y[i] = result_y_3[i];
        }
//        file_w << trans_x << " " << trans_y << std::endl;
        for(int i = 0; i < predict_num_/2; i++){
            cv::Mat comp2 = cv::Mat::eye(3, 3, CV_64F);

            limit li = limit_que_.front();
            limit_que_.pop();
            if(trans_x[i] < li.xmin){
                trans_x[i] = li.xmin;
            }else if(trans_x[i] > li.xmax){
                trans_x[i] = li.xmax;
            }
            if(trans_y[i] < li.ymin){
                trans_y[i] = li.ymin;
            }else if(trans_y[i] > li.ymax){
                trans_y[i] = li.ymax;
            }
            bool flag = false;
            if(li.xmin > li.xmax){
                LOGI("xmin is bigger than xmax!");
                flag = true;
            }
            if(li.ymin > li.ymax){
                LOGI("ymin is bigger than ymax!");
                flag = true;
            }

            comp2.at<double>(0, 2) = trans_x[i];
            comp2.at<double>(1, 2) = trans_y[i];
            cv::Mat temp = s_mat_.front().inv();
            s_mat_.pop();
            cv::Mat news = (comp2 * temp).inv();
            if(flag){
                cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
                cv::Mat stable_vec = temp.clone().inv();
                cv::Mat newvertex = stable_vec * vertex_;
                newvertex.at<double>(0, 0) = newvertex.at<double>(0, 0) / newvertex.at<double>(2, 0);
                newvertex.at<double>(1, 0) = newvertex.at<double>(1, 0) / newvertex.at<double>(2, 0);

                newvertex.at<double>(0, 1) = newvertex.at<double>(0, 1) / newvertex.at<double>(2, 1);
                newvertex.at<double>(1, 1) = newvertex.at<double>(1, 1) / newvertex.at<double>(2, 1);

                newvertex.at<double>(0, 2) = newvertex.at<double>(0, 2) / newvertex.at<double>(2, 2);
                newvertex.at<double>(1, 2) = newvertex.at<double>(1, 2) / newvertex.at<double>(2, 2);

                newvertex.at<double>(0, 3) = newvertex.at<double>(0, 3) / newvertex.at<double>(2, 3);
                newvertex.at<double>(1, 3) = newvertex.at<double>(1, 3) / newvertex.at<double>(2, 3);
                bool all_inside = false;
                double r = 1.0;
                cv::Mat result_vec;
                while ((!all_inside) && (r > 0.001)){
                    double transdet= cv::determinant(stable_vec);//求行列式
                    cv::Mat transtemp = stable_vec/pow(transdet, 1.0/3);
                    r = r - 0.05;
                    result_vec = I * (1-r) + transtemp * r;
                    cv::Mat test_res = (comp2 * result_vec.inv()).inv();
                    newvertex = test_res * vertex_;
                    newvertex.at<double>(0,0)=newvertex.at<double>(0,0)/newvertex.at<double>(2,0);
                    newvertex.at<double>(1,0)=newvertex.at<double>(1,0)/newvertex.at<double>(2,0);

                    newvertex.at<double>(0,1)=newvertex.at<double>(0,1)/newvertex.at<double>(2,1);
                    newvertex.at<double>(1,1)=newvertex.at<double>(1,1)/newvertex.at<double>(2,1);

                    newvertex.at<double>(0,2)=newvertex.at<double>(0,2)/newvertex.at<double>(2,2);
                    newvertex.at<double>(1,2)=newvertex.at<double>(1,2)/newvertex.at<double>(2,2);

                    newvertex.at<double>(0,3)=newvertex.at<double>(0,3)/newvertex.at<double>(2,3);
                    newvertex.at<double>(1,3)=newvertex.at<double>(1,3)/newvertex.at<double>(2,3);

                    all_inside = isInside(cropvertex_,newvertex);
                }
                LOGI("not compensation!  %f", r);
                news = (comp2 * result_vec.inv()).inv();
                if(r < 0.04){
                    news = I;
                }
                output_buffer_.push(news.clone());
            } else {
                cv::Mat I = cv::Mat::eye(3, 3, CV_64F);

                cv::Mat stable_vec = news.clone();
                cv::Mat newvertex = stable_vec * vertex_;
                newvertex.at<double>(0, 0) = newvertex.at<double>(0, 0) / newvertex.at<double>(2, 0);
                newvertex.at<double>(1, 0) = newvertex.at<double>(1, 0) / newvertex.at<double>(2, 0);

                newvertex.at<double>(0, 1) = newvertex.at<double>(0, 1) / newvertex.at<double>(2, 1);
                newvertex.at<double>(1, 1) = newvertex.at<double>(1, 1) / newvertex.at<double>(2, 1);

                newvertex.at<double>(0, 2) = newvertex.at<double>(0, 2) / newvertex.at<double>(2, 2);
                newvertex.at<double>(1, 2) = newvertex.at<double>(1, 2) / newvertex.at<double>(2, 2);

                newvertex.at<double>(0, 3) = newvertex.at<double>(0, 3) / newvertex.at<double>(2, 3);
                newvertex.at<double>(1, 3) = newvertex.at<double>(1, 3) / newvertex.at<double>(2, 3);
                if(!isInside(cropvertex_, newvertex)){
                    //LOGI("set it to I");
                    news = I;
                    cv::Mat result_vec;
                    bool all_inside = false;
                    double r = 1.0;
                    while ((!all_inside) && (r > 0.001)){
                        double transdet= cv::determinant(stable_vec);//求行列式
                        cv::Mat transtemp = stable_vec/pow(transdet, 1.0/3);
                        r = r - 0.01;
                        result_vec = I * (1-r) + transtemp * r;
                        //cv::Mat test_res = (comp2 * result_vec.inv()).inv();
                        newvertex = result_vec * vertex_;
                        newvertex.at<double>(0,0)=newvertex.at<double>(0,0)/newvertex.at<double>(2,0);
                        newvertex.at<double>(1,0)=newvertex.at<double>(1,0)/newvertex.at<double>(2,0);

                        newvertex.at<double>(0,1)=newvertex.at<double>(0,1)/newvertex.at<double>(2,1);
                        newvertex.at<double>(1,1)=newvertex.at<double>(1,1)/newvertex.at<double>(2,1);

                        newvertex.at<double>(0,2)=newvertex.at<double>(0,2)/newvertex.at<double>(2,2);
                        newvertex.at<double>(1,2)=newvertex.at<double>(1,2)/newvertex.at<double>(2,2);

                        newvertex.at<double>(0,3)=newvertex.at<double>(0,3)/newvertex.at<double>(2,3);
                        newvertex.at<double>(1,3)=newvertex.at<double>(1,3)/newvertex.at<double>(2,3);

                        all_inside = isInside(cropvertex_,newvertex);
                    }
                    LOGI("set it to I:  %f", r);
                    //news = (comp2 * result_vec.inv()).inv();
                    news = result_vec;
                    if(r < 0.04){
                        news = I;
                    }
                }


                output_buffer_.push(news.clone());
            }
            num_que_ = predict_num_/2 + 1;
            if(write_status_){
                file_w1 << frame_count_m << " " << trans_x[i] << " " << trans_y[i] << " "<< flag <<std::endl;
                frame_count_m++;
            }

        }

    }

}

double AutoFilter::calError(double *ori, double *aft, int n) {
    double err = 0;
    for(int i = 0; i < n; i++){
        err += abs(sqrt(abs(pow(ori[i], 2) - pow(aft[i], 2))));
    }
    return err;
}

void AutoFilter::queue_in(double *q, int m, double x) {
    for(int i=1; i<=m; i++)
    {
        q[i-1] = q[i];
    }
    q[m]=x;
}

void AutoFilter::polyfit(double *arrX, double *arrY, int num, int n, double* result) {
    int size = num;//vec里的个数
    int x_num = n + 1;//次数+1
    //构造矩阵U和Y
    cv::Mat mat_u(size, x_num, CV_64F);//num行，n+1列
    cv::Mat mat_y(size, 1, CV_64F);//num行，1列

    for (int i = 0; i < mat_u.rows; ++i)
        for (int j = 0; j < mat_u.cols; ++j)
        {
            mat_u.at<double>(i, j) = pow(arrX[i], j);//x^j次方
        }

    for (int i = 0; i < mat_y.rows; ++i)
    {
        mat_y.at<double>(i, 0) = arrY[i];
    }

    //矩阵运算，获得系数矩阵K
    cv::Mat mat_k(x_num, 1, CV_64F);
    mat_k = (mat_u.t()*mat_u).inv()*mat_u.t()*mat_y;
    //std::cout << mat_k << std::endl;
    //return mat_k;

    for(int i = 0; i < 3; i++){
        result[i] = 0;
    }
    for(int i = 0; i < 3; i++){
        for (int j = 0; j < n + 1; ++j)
        {
            result[i] += mat_k.at<double>(j, 0)*pow(i+1,j);
        }
    }

}

void AutoFilter::decomposeHomo(cv::Mat h, cv::Point2f cen, cv::Mat &perp, cv::Mat &sca,
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
    //cout << "fax: " << fax << " fay: " << fay << endl;

    double n = 1 - cx*fax - cy*fay;

    double ctx = n*hm[0][2] + cx*n*hm[0][0] + cy*n*hm[0][1];
    double cty = n*hm[1][2] + cx*n*hm[1][0] + cy*n*hm[1][1];
    //cout << "n: " << n << " ctx: " << ctx << " cty: " << cty << endl;

    tx = ctx - cx;
    ty = cty - cy;

    double xc = n*hm[0][0] - fax*n*hm[0][2] - cx*fax*n*hm[0][0] - cy*fax*n*hm[0][1];
    double xs = n*hm[1][0] - fax*n*hm[1][2] - cx*fax*n*hm[1][0] - cy*fax*n*hm[1][1];
    double ysmc = - n*hm[0][1] + fay*n*hm[0][2] + cx*fay*n*hm[0][0] + cy*fay*n*hm[0][1];
    double ycps = n*hm[1][1] - fay*n*hm[1][2] - cx*fay*n*hm[1][0] - cy*fay*n*hm[1][1];
    //cout << "xc: " << xc << " xs: " << xs << " ysmc: " << ysmc << " ycps: " << ycps << endl;
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

    //cout << "theta: " << theta << " lamx: " << lamx << " lamy: " << lamy << endl;

    cv::Mat M1=(cv::Mat_<double>(3, 3) <<
                                   1.0, 0.0, -cx,
            0.0, 1.0, -cy,
            0.0, 0.0, 1.0);

    cv::Mat M2=(cv::Mat_<double>(3, 3) <<
                                   1.0, 0.0, cx,
            0.0, 1.0, cy,
            0.0, 0.0, 1.0);

    cv::Mat PE=(cv::Mat_<double>(3, 3) <<
                                   1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            fax, fay, 1.0);

    cv::Mat SC=(cv::Mat_<double>(3, 3) <<
                                   lamx, 0.0, 0.0,
            0.0, lamy, 0.0,
            0.0, 0.0, 1.0);

    cv::Mat SH=(cv::Mat_<double>(3, 3) <<
                                   1.0, sh, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0);

    cv::Mat RO=(cv::Mat_<double>(3, 3) <<
                                   cos(theta), -sin(theta), 0.0,
            sin(theta), cos(theta), 0.0,
            0.0, 0.0, 1.0);

    cv::Mat TR=(cv::Mat_<double>(3, 3) <<
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

void AutoFilter::analyzeTrans(cv::Mat& comp) {
    double limit_x = center.x / 8;
    double limit_y = center.y / 8;
    bool all_small = true;
    double sum_x = 0, sum_y = 0;
    int num = 0;
    for (int i = max_size_ / 2; i < trans_buffer_.size(); i++) {
        sum_x += trans_buffer_[i].x;
        sum_y += trans_buffer_[i].y;
        num++;
        if (trans_buffer_[i].x > limit_x || trans_buffer_[i].y > limit_y) {
            all_small = false;
            break;
        }
    }
    sum_x /= num;
    sum_y /= num;
    if (sum_x > limit_x / 3 || sum_y > limit_y / 3) {
        all_small = false;
    }

    if(all_small)
    {
        double limit_up = 1.03, limit_up_avg = 1.005;
        double limit_down = 0.97, limit_down_avg = 0.995;
        sum_x = 0;
        sum_y = 0;
        num = 0;
        for (int i = max_size_ / 2; i < sca_buffer_.size(); i++) {
            sum_x += sca_buffer_[i].x;
            sum_y += sca_buffer_[i].y;
            num++;
            if (sca_buffer_[i].x > limit_up || sca_buffer_[i].y > limit_up || sca_buffer_[i].x < limit_down || sca_buffer_[i].y < limit_down) {
                all_small = false;
                break;
            }
        }
        sum_x /= num;
        sum_y /= num;
        LOGI("sca_avg para: %f, %f",sum_x, sum_y);
        if (sum_x > limit_up_avg || sum_y > limit_up_avg || sum_x < limit_down_avg || sum_y < limit_down_avg) {
            all_small = false;
        }
    }

    int t = input_buffer_.size();
    cv::Mat temp = input_trans_buffer_[1];
    input_trans_buffer_.pop_front();
    //LOGI("keep trans: %d / %d", keep_num, max_size_ / 2);
    /*if(all_small && keep < 20)
    {
        if(keep_num == 0)
        {
            keep = 0;
            cum_H = comp.clone();
        } else{
            \
            keep++;
            cum_H = cum_H * temp.inv();
        }
        keep_num = num;
        double per = (double)keep_num/(max_size_/2);
        LOGI("keep_per1: %f", per);
        if(per>1)
        {
            per = 1;
        }
        comp = cum_H.clone() * per + comp * (1-per);

        return;
    }
    else
    {
        if(keep_num > 0)
        {
            cum_H = cum_H * temp.inv();
            keep_num--;
            keep++;
            double per = (double)keep_num/(max_size_/2);
            LOGI("keep_per2: %f", per);
            if(per>1)
            {
                per = 1;
            }
            comp = cum_H.clone() * per + comp * (1-per);
        }
        if(keep_num == 0) {
            keep = 0;
        }
        return;
    }*/

    if(mode == 21)
    {

    }
    else if(mode == 12)
    {

    }
    else if(mode == 1 && all_small && keep_num <= 0)
    {
        mode = 12;
    }
    else if(mode == 2 && (!all_small || move_status == 3 || move_status==4))
    {
        mode = 21;
    }

    if (mode == 21)
    {
        LOGI("mode-21");
        first_in = true;
        //由辅助模式进入正常模式
        keep_rate -= 0.1;
        LOGI("keep_rate1: %f", keep_rate);
        if(keep_rate < 0)
        {
            keep_rate = 0;
            keep_num = 50;
            mode = 1;
        }
        cum_H = cum_H * temp.inv();
        comp = cum_H.clone() * keep_rate + comp * (1 - keep_rate);
    }
    else if (mode == 12)
    {
        LOGI("mode-12");
        //由正常模式进入辅助模式
        //cv::Mat I = trans_for_cumh.clone();
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        keep_rate += 0.1;

        if(keep_rate >= 1.0)
        {
            keep_rate = 1.0;
            cum_H = I.clone();
            mode = 2;
        }
        comp = comp * (1-keep_rate) + I * keep_rate;

    }
    else if (mode == 2) {
        LOGI("mode-2");
        /*if (first_in)
        {
            first_in = false;
            keep_rate = 1.0;    //由正常模式进入辅助模式（一步完成）
        }*/

        /*if (keep_rate == 1.0)
        {
            cum_H = comp.clone();
        } else{
            cum_H = cum_H * temp.inv();
        }

        if (keep_rate >= 1.0) {
            keep_rate = 1.0;
            keep_change_rate = -0.01;
        }
        if (keep_rate <= 0.0) {
            keep_rate = 1.0;
            //keep_change_rate = 0.01;
        }
        keep_rate += keep_change_rate;

        LOGI("keep_rate2: %f", keep_rate);

        comp = cum_H.clone() * keep_rate + comp * (1 - keep_rate);*/

        cum_H = cum_H * temp.inv();

        //cv::Mat temp = trans_for_cumh.clone();
        cv::Mat temp = cum_H.clone();
        LOGI("cum_h: %f, %f, %f, %f, %f, %f, %f, %f, %f", temp.at<double>(0,0), temp.at<double>(0,1), temp.at<double>(0,2), temp.at<double>(1,0), temp.at<double>(1,1), temp.at<double>(1,2), temp.at<double>(2,0), temp.at<double>(2,1), temp.at<double>(2,2));
        cv::Mat perp, sca, shear, rot, trans;
        decomposeHomo(temp, center, perp, sca, shear, rot, trans);

        cv::Mat M1=(cv::Mat_<double>(3, 3) <<
                                       1.0, 0.0, -center.x,
                0.0, 1.0, -center.y,
                0.0, 0.0, 1.0);
        cv::Mat M2=(cv::Mat_<double>(3, 3) <<
                                       1.0, 0.0, center.x,
                0.0, 1.0, center.y,
                0.0, 0.0, 1.0);

        perp = M2.inv() * perp * M1.inv();
        sca = M2.inv() * sca * M1.inv();
        shear = M2.inv() * shear * M1.inv();
        rot = M2.inv() * rot * M1.inv();
        double theta = asin(rot.at<double>(1,0));
        LOGI("cum_h decom result: trans: %f, %f; perp: %f, %f; sca: %f, %f; shear: %f; rot: %f",
             trans.at<double>(0,2), trans.at<double>(1,2), perp.at<double>(2,0), perp.at<double>(2,1),
             sca.at<double>(0,0), sca.at<double>(1,1), shear.at<double>(0,1), theta);

        //comp = trans_for_cumh * cum_H.clone();
        comp = cum_H.clone();
    }
    else if(mode == 1)
    {
        keep_num --;
        if(keep_num < 0)
        {
            keep_num = -1;
        }
        LOGI("mode-1");
    }
}
