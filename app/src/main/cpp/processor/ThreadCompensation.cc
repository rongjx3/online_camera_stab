//
// Created by 张哲华 on 19/09/2017.
//

#include "ThreadCompensation.h"

#include <android/log.h>
#include <jni.h>
#include <opencv2/core/mat.hpp>

#define LOG_TAG    "c_ThreadCompensation"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace cv;
using namespace std;
using namespace threads;
static int frame_count = 0;
static cv::Mat point_before = (cv::Mat_<double>(3, 1) << 960, 540, 0);
static cv::Mat point_after = (cv::Mat_<double>(3, 1) << 960, 540, 0);
static FILE* file;
static FILE* file_after;
static std::queue<cv::Mat> trans_que;
static std::queue<cv::Mat> r_temp_queue;
static cv::Mat tran_cumm = cv::Mat::eye(3, 3, CV_64F);
static std::ofstream r_temp_file("/data/data/me.zhehua.gryostable/data/r_temp.txt");
static std::ofstream r_temp1_file("/data/data/me.zhehua.gryostable/data/r_temp1.txt");
static std::ofstream scale_info_file("/data/data/me.zhehua.gryostable.version_new2/data/scale_info.txt");
static std::ofstream rota_xyz_file("/data/data/me.zhehua.gryostable.version_new2/data/rota_xyz.txt");

static int angle_frame_count = 0, rota_frame_count = 0;
double point_distance(cv::Point2f p1,cv::Point2f p2)
{
    cv::Point2f d = p1 - p2;
    double d_mu;
    d_mu = sqrt(d.x * d.x + d.y * d.y);
    return d_mu;
}

bool outOfImg(const cv::Point2f &point, const cv::Size &size)
{
    return (point.x <= 0 || point.y <= 0 || point.x >= size.width - 1 || point.y >= size.height - 1 );
}

ThreadCompensation::~ThreadCompensation() {
    worker_thread_.join();
}

void ThreadCompensation::start() {
//    file = fopen("/data/data/me.zhehua.gryostable/data/rotatemat.txt", "a");
//    file_after = fopen("/data/data/me.zhehua.gryostable/data/track_after.txt", "a");
//    homoExtractor.setDrawStatus(drawFlag);
    filter = Filter(30, 40, Filter::delta_T);
    filter1 = AutoFilter(20, 40);
    worker_thread_ = thread(&ThreadCompensation::worker, this);
}

void ThreadCompensation::worker()
{
    //LOGI("ThreadCompensation::worker");
//    filter = Filter(ThreadContext::SEGSIZE * 2 , 5);
    lastRot[0]=0;
    lastRot[1]=0;
    lastRot[2]=0;
    //cropRation = 0.8;
    pthread_setname_np(pthread_self(), "CompensationThread"); // set the name (pthread_self() returns the pthread_t of the current thread
    while(true)
    {
        //LOGI("loop start");
        ThreadContext::mc_semaphore->Wait();//取已经完成特征点轨迹构造的资源，若无，线程等待
//        __android_log_print(ANDROID_LOG_DEBUG, "NThreadMC", "before");
        if( cm_las_index_ < 0 )
        {
            ThreadContext::out_semaphore->Signal();
            break;
        }

        frameCompensate();
        //LOGI("loop end");

//        ex_index_ = (ex_index_ + 1) % ThreadContext::SEGSIZE;
        cm_las_index_ = (cm_las_index_ + 1) % ThreadContext::BUFFERSIZE;
        cm_cur_index_ = (cm_cur_index_ + 1) % ThreadContext::BUFFERSIZE;
        //ThreadContext::out_semaphore->Signal();//唤醒显示和保存线程

    }
}

double line_distance(cv::Point2f p0, cv::Point2f p1, cv::Point2f p2)
{
    double d = (fabs((p2.y - p1.y) * p0.x +(p1.x - p2.x) * p0.y + ((p2.x * p1.y) -(p1.x * p2.y)))) / (sqrt(pow(p2.y - p1.y, 2) + pow(p1.x - p2.x, 2)));
    return d;
}//中心点到p1p2组成直线的距离

//检测是否抖动
bool ThreadCompensation::stable_count(double e)
{
    /*double height = curGray.rows * ThreadContext::DOWNSAMPLE_SCALE;
    double width = curGray.cols * ThreadContext::DOWNSAMPLE_SCALE;
    cv::Point2f cen(width/2, height/2);
    double long_side = width > height ? width : height;
    double limit_cor = sqrt(pow(height, 2) + pow(width, 2))/15;

    double sta_sca_limit = 0.00015;
    double sta_limit = long_side * sta_sca_limit;
    double sca_limit = long_side * sta_sca_limit * 4;
    int numStable = 0,numScale = 0;

    std::vector<cv::Point2f> last_sc,cur_sc;

    int num=(curFeatures.size()<lastFeatures.size()?curFeatures.size():lastFeatures.size());
    for(int i=0;i<num;i++)
    {
        cv::Point2f d = curFeatures[i] - lastFeatures[i];
        float d_mu;
        d_mu = sqrt(d.x * d.x + d.y * d.y);
        if (d_mu > sca_limit || status_choose[i]==0)
        {

        }
        else if (d_mu > sta_limit && d_mu <= sca_limit)
        {
            cv::Point2f cp1 = curFeatures[i] * ThreadContext::DOWNSAMPLE_SCALE, lp1 = lastFeatures[i] * ThreadContext::DOWNSAMPLE_SCALE;
            double d=line_distance(cen,cp1,lp1);

            if(d<limit_cor)
            {
                cur_sc.push_back(cp1);
                last_sc.push_back(lp1);
                numScale++;
            } else
            {
                //numStable++;
            }
        }
        else
        {
            numStable++;
        }
    }*/

    //LOGE("stable and scale: %d / %d", numStable , numScale);
    if(e < 1e-7)
    {
        LOGE("is stable");
        return true;
    }
    else
    {
        return false;
    }
}

Mat ThreadCompensation::computeAffine(cv::Mat &R)
{
    //LOGI("step1");

    homoExtractor.setDrawStatus(drawFlag);
    Mat lastFrame = ThreadContext::frameVec[cm_las_index_];
    Mat frame = ThreadContext::frameVec[cm_cur_index_];
    frameSize.height=frame.rows * 2 / 3;
    frameSize.width=frame.cols;

    center.x=frameSize.width/2;
    center.y=frameSize.height/2;

    if(is_first_use_rtheta){
        ThreadContext::rTheta.pop();
        is_first_use_rtheta = false;
    }
    if(is_first_frame)
    {
        lastf=frame.clone();
        is_first_frame = false;
    } else{
        lastFrame=lastf.clone();
        lastf=frame.clone();
    }
    Vec<double, 3> rot = ThreadContext::rTheta.front();//前一帧的旋转矩阵
    ThreadContext::rTheta.pop();

    Vec<double, 3> er = rot;
    lastRot = rot;
    std::string gyro_info = "theta:";
    gyro_info = gyro_info + std::to_string(rot[0]/PI*180)+" "+std::to_string(rot[1]/PI*180)+" "+std::to_string(rot[2]/PI*180);

    //LOGI("see r : %f, %f, %f ", er[0], er[1], er[2]);
    double error_x = er[1]*er[1];
    double error_y = er[0]*er[0];
    double error_z = er[2]*er[2];
    double error = er[0]*er[0] + er[1]*er[1] + er[2]*er[2];

    //rota_xyz_file << rota_frame_count << " " << er[1] << " " << er[0] << " " << er[2] << std::endl;
    //rota_frame_count++;

    bool sc = false;
    int m = 0;
    //sc = stable_count(error);
    error_q.push_back(error);
    error_q_x.push_back(error_x);
    error_q_y.push_back(error_y);
    error_q_z.push_back(error_z);
    if(error_q.size()>10)
    {
        error_q.erase(error_q.begin());
        error_q_x.erase(error_q_x.begin());
        error_q_y.erase(error_q_y.begin());
        error_q_z.erase(error_q_z.begin());

    }
    LOGE("queue_size: %d",error_q.size());
    double error_sum = 0;
    double error_sum_x = 0, error_sum_y = 0, error_sum_z = 0;
    for(int i = 0;i < error_q.size(); i++)
    {
        error_sum += error_q[i];
        error_sum_x += error_q_x[i];
        error_sum_y += error_q_y[i];
        error_sum_z += error_q_z[i];
    }

    if(error < 1e-7)
    {
        //静止
        LOGE("is stable");
        sc = true;
        m = 0;
    }
    else
    {
        //非静止
        sc = false;
        double l_s = 0.02, l_e = 0.01;
        if(error_sum < l_s)
        {
            m = 1;  //平移
            R = cv::Mat::eye(3, 3, CV_64F);
        }
        else
        {
            m = 2;  //旋转
            if(error_sum_x > l_e && error_sum_y < l_e && error_sum_z < l_e)
            {
                m = 3;  //x轴旋转
            }
            if(error_sum_x < l_e && error_sum_y > l_e && error_sum_z < l_e)
            {
                m = 4;  //y轴旋转
            }
            if(error_sum_x < l_e && error_sum_y < l_e && error_sum_z > l_e)
            {
                m = 5;  //z轴旋转
                R = cv::Mat::eye(3, 3, CV_64F);
            }

        }
    }
    homoExtractor.set_move_status(m);
    filter1.set_move_status(m);

    is_stable_ = sc;
    LOGI("see error : %f, %d error_sum: %f, each: %f, %f, %f", error, is_stable_, error_sum, er[0], er[1], er[2]);
    LOGI("see each sum error : %f, %f, %f; m: %d", error_sum_x, error_sum_y, error_sum_z, m );

    Mat aff = cv::Mat::eye(3,3,CV_64F);
    if(sc && shakeDetect)
    {
        LOGI("is stable!");
        aff = RR2stableVec * stableVec2RR;
    }
    else
    {
        aff = homoExtractor.extractHomo(lastFrame, frame, R);
        //aff = cv::Mat::eye(3, 3, CV_64F);
        LOGI("aff extractHomo:[%f, %f, %f, %f, %f, %f, %f, %f, %f]", aff.at<double>(0, 0), aff.at<double>(0, 1), aff.at<double>(0, 2),
             aff.at<double>(1, 0), aff.at<double>(1, 1), aff.at<double>(1, 2),
             aff.at<double>(2, 0), aff.at<double>(2, 1), aff.at<double>(2, 2));
    }
    if(homoExtractor.draw_information){
        //cv::putText(frame, gyro_info, cv::Point(200,300), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(255, 0, 0), 2, 8, 0);
    }

    return aff;
}

double Mat_error_I(cv::Mat m)
{
    cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat e = I - m;
    double res = 0;
    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            res += e.at<double>(i,j) * e.at<double>(i,j);
        }
    }
    return res;
}

void ThreadCompensation::frameCompensate()
{
    cv::Mat old_r_mat = threads::ThreadContext::r_convert_que.front();
    cv::Mat old_r_mat1 = threads::ThreadContext::r_convert_que1.front();
    threads::ThreadContext::r_convert_que1.pop();
    threads::ThreadContext::r_convert_que.pop();

    cv::Mat temp = old_r_mat;//old_r_mat是从上一帧到这一帧的变化
    cv::Mat r_temp = inmat * temp * inmat.inv();
    //r_temp = cv::Mat::eye(3,3,CV_64F);
    /*为旋转插值准备数据，即计算仿射矩阵，计算本段非关键帧到前关键帧的仿射矩阵与前关键帧到后关键帧的仿射矩阵*/
    //LOGI("cal aff");
    Mat aff = computeAffine(r_temp);
    //Mat aff=cv::Mat::eye(3, 3, CV_64F);
    cv::Mat M1 = (cv::Mat_<double>(3, 3) << 1, 0, - center.x, 0, 1, - center.y, 0, 0, 1);
    cv::Mat M2 = (cv::Mat_<double>(3, 3) << 1, 0, center.x, 0, 1, center.y, 0, 0, 1);

    LOGI("centerxy: x:%f, y:%f", center.x, center.y);

    double e_i;
    cv::Mat a_perp, a_sca, a_shear, a_rot, a_trans;
    cv::Mat r_perp, r_sca, r_shear, r_rot, r_trans;
    decomposeHomo(aff, center, a_perp, a_sca, a_shear, a_rot, a_trans);
    decomposeHomo(r_temp, center, r_perp, r_sca, r_shear, r_rot, r_trans);
    /*temp = r_perp.clone();
    e_i=Mat_error_I(temp);
    //LOGI("rt_perp Matinthreads: %f; %f, %f, %f, %f, %f, %f, %f, %f, %f", e_i, temp.at<double>(0,0), temp.at<double>(0,1), temp.at<double>(0,2), temp.at<double>(1,0), temp.at<double>(1,1), temp.at<double>(1,2), temp.at<double>(2,0), temp.at<double>(2,1), temp.at<double>(2,2));
    temp = r_sca.clone();
    e_i=Mat_error_I(temp);
    //LOGI("rt_sca Matinthreads: %f; %f, %f, %f, %f, %f, %f, %f, %f, %f", e_i, temp.at<double>(0,0), temp.at<double>(0,1), temp.at<double>(0,2), temp.at<double>(1,0), temp.at<double>(1,1), temp.at<double>(1,2), temp.at<double>(2,0), temp.at<double>(2,1), temp.at<double>(2,2));
    temp = r_shear.clone();
    e_i=Mat_error_I(temp);
    //LOGI("rt_shear Matinthreads: %f; %f, %f, %f, %f, %f, %f, %f, %f, %f", e_i, temp.at<double>(0,0), temp.at<double>(0,1), temp.at<double>(0,2), temp.at<double>(1,0), temp.at<double>(1,1), temp.at<double>(1,2), temp.at<double>(2,0), temp.at<double>(2,1), temp.at<double>(2,2));
    temp = r_rot.clone();
    e_i=Mat_error_I(temp);
    //LOGI("rt_rot Matinthreads: %f; %f, %f, %f, %f, %f, %f, %f, %f, %f", e_i, temp.at<double>(0,0), temp.at<double>(0,1), temp.at<double>(0,2), temp.at<double>(1,0), temp.at<double>(1,1), temp.at<double>(1,2), temp.at<double>(2,0), temp.at<double>(2,1), temp.at<double>(2,2));
    temp = r_trans.clone();
    e_i=Mat_error_I(temp);
    //LOGI("rt_trans Matinthreads: %f; %f, %f, %f, %f, %f, %f, %f, %f, %f", e_i, temp.at<double>(0,0), temp.at<double>(0,1), temp.at<double>(0,2), temp.at<double>(1,0), temp.at<double>(1,1), temp.at<double>(1,2), temp.at<double>(2,0), temp.at<double>(2,1), temp.at<double>(2,2));
    */
    //

    //LOGI("new_aff Matinthreads: %f, %f, %f, %f, %f, %f, %f, %f, %f", new_aff.at<double>(0,0), new_aff.at<double>(0,1), new_aff.at<double>(0,2), new_aff.at<double>(1,0), new_aff.at<double>(1,1), new_aff.at<double>(1,2), new_aff.at<double>(2,0), new_aff.at<double>(2,1), new_aff.at<double>(2,2));
    //temp = perp.clone();
    //e_i=Mat_error_I(temp);
    //LOGI("aff_perp Matinthreads: %f; %f, %f, %f, %f, %f, %f, %f, %f, %f", e_i, temp.at<double>(0,0), temp.at<double>(0,1), temp.at<double>(0,2), temp.at<double>(1,0), temp.at<double>(1,1), temp.at<double>(1,2), temp.at<double>(2,0), temp.at<double>(2,1), temp.at<double>(2,2));
    //aff = perp * sca * shear * rot;

    auto new_aff = aff;
    e_i=Mat_error_I(aff);

    if(!is_stable_ ){

    } else {
        r_temp = cv::Mat::eye(3, 3, CV_64F);
        new_aff = cv::Mat::eye(3, 3, CV_64F);
    }
    threads::ThreadContext::last_old_Rotation_ = old_r_mat1;

    //测试旋转矩阵

    new_aff =  new_aff * r_temp;
    cv::Mat perp, sca, shear, rot, trans;
    decomposeHomo(new_aff, center, perp, sca, shear, rot, trans);
    sca = M2.inv() * sca * M1.inv();

    if(homo_type == 1)
    {

        //new_aff =  r_trans * r_rot * r_shear * r_sca * r_perp;
        //new_aff =  a_trans * r_rot * a_shear * a_sca * r_perp;
    }
    else
    {
        //new_aff = r_temp;
        //new_aff =  new_aff * r_trans * r_shear * r_sca * r_perp;
        //new_aff =  a_trans * a_rot * a_shear * a_sca * a_perp;
        //new_aff = cv::Mat::eye(3, 3, CV_64F);
        double a_s = (sca.at<double>(0,0) + sca.at<double>(1,1)) / 2;
        sca.at<double>(0,0) = a_s;
        sca.at<double>(1,1) = a_s;

        new_aff =  trans * rot * M2 * sca * M1;

    }

    new_aff = limit_Mat(new_aff);

    //new_aff =  trans * rot * shear * perp;

    LOGI("new_aff Matinthreads: %f, %f, %f, %f, %f, %f, %f, %f, %f", new_aff.at<double>(0,0), new_aff.at<double>(0,1), new_aff.at<double>(0,2), new_aff.at<double>(1,0), new_aff.at<double>(1,1), new_aff.at<double>(1,2), new_aff.at<double>(2,0), new_aff.at<double>(2,1), new_aff.at<double>(2,2));

    //LOGI("new_aff after_Matinthreads: %f, %f, %f, %f, %f, %f, %f, %f, %f", new_aff.at<double>(0,0), new_aff.at<double>(0,1), new_aff.at<double>(0,2), new_aff.at<double>(1,0), new_aff.at<double>(1,1), new_aff.at<double>(1,2), new_aff.at<double>(2,0), new_aff.at<double>(2,1), new_aff.at<double>(2,2));

    if(true){
//        WriteToFile(file, temp);
//        r_temp_file << angle_frame_count << " " << temp.at<double>(0,0) << " " << temp.at<double>(0,1) << " " << temp.at<double>(0,2)
//                << " " << temp.at<double>(1,0) << " " << temp.at<double>(1,1) << " " << temp.at<double>(1,2)
//                << " " << temp.at<double>(2,0) << " " << temp.at<double>(2,1) << " " << temp.at<double>(2,2) << std::endl;
//        r_temp1_file << angle_frame_count << " " << temp1.at<double>(0,0) << " " << temp1.at<double>(0,1) << " " << temp1.at<double>(0,2)
//                    << " " << temp1.at<double>(1,0) << " " << temp1.at<double>(1,1) << " " << temp1.at<double>(1,2)
//                    << " " << temp1.at<double>(2,0) << " " << temp1.at<double>(2,1) << " " << temp1.at<double>(2,2) << std::endl;
        scale_info_file << angle_frame_count << " " << sca.at<double>(0,0) << " " << sca.at<double>(1,1) << " " << trans.at<double>(0,2) << " " << trans.at<double>(1,2) << std::endl;
        angle_frame_count++;

    }

//    aff = aff * r_temp;
    trans_que.push(new_aff);
    filter1.write_status_ = is_write_to_file_;
    bool readyToPull = filter1.push(new_aff.clone());
    LOGI("readyToPull : %f", readyToPull);
    if (readyToPull) {
        cv::Mat gooda = filter1.pop();
        cv::Mat goodar = gooda;
//        goodar = ThreadContext::stableRVec[out_index_];

        if( cropControlFlag )
        {
            cv::Mat scale = cv::Mat::eye(3,3,CV_64F);
            cv::Mat move = cv::Mat::eye(3,3,CV_64F);
            double croph=frameSize.height*cropRation;
            double cropw=frameSize.width*cropRation;
            double mh=(frameSize.height-croph)/2;
            double mw=(frameSize.width-cropw)/2;
            scale.at<double>(0,0) = 1.0 / cropRation;
            scale.at<double>(1,1) = 1.0 / cropRation;
            move.at<double>(0,2) = -mw;
            move.at<double>(1,2) = -mh;

            goodar = scale * move * goodar;
            //goodar = scale * move;
        } else {
//            cropControl(cropRation, frameSize, goodar);
//            cv::Point2d p1(crop_vertex.at<double>(0, 0), crop_vertex.at<double>(1, 0));
//            cv::Point2d p2(crop_vertex.at<double>(0, 1), crop_vertex.at<double>(1, 1));
//            cv::Point2d p3(crop_vertex.at<double>(0, 2), crop_vertex.at<double>(1, 2));
//            cv::Point2d p4(crop_vertex.at<double>(0, 3), crop_vertex.at<double>(1, 3));
//            cv::Mat frame = threads::ThreadContext::frameVec[cm_cur_index_];
//
//            cv::line(frame,p1,p2,cv::Scalar(0,255,0),8);
//            cv::line(frame,p2,p3,cv::Scalar(0,255,0),8);
//            cv::line(frame,p3,p4,cv::Scalar(0,255,0),8);
//            cv::line(frame,p4,p1,cv::Scalar(0,255,0),8);
        }

        goodar.copyTo(ThreadContext::stableTransformVec[out_index_]);
        out_index_ = (out_index_ + 1) % ThreadContext::BUFFERSIZE;

//        frame_count++;
        ThreadRollingShutter::getStableStatus(is_stable_);
        ThreadContext::rs_semaphore_->Signal();
    }

    //LOGI("compensate end");
}

bool isInside(cv::Mat cropvertex ,cv::Mat newvertex)
{
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


//计算最大旋转角
//img_line：起点为原点，与X轴重合，长度为图像长或宽
//crop_line：裁剪线
//degree：旋转角度，正值为逆时针旋转
//center：旋转中心
double ThreadCompensation::computeMaxDegree( vector<Point2f> img_line , vector<Point2f> crop_line , double degree , Point2f center )
{
    if( degree > 0 )
    {
        if( center.x <= crop_line[0].x )
        {
            return 3.1415926;
        }
        else
        {
            float dis = sqrt( pow(crop_line[0].x - center.x,2) + pow(crop_line[0].y - center.y,2) );
            if( dis <= center.y )
            {
                return 3.1415926;
            }
            else
            {
                /*计算切点*/
                float a1 , a2 , a3;
                a1 = center.x - crop_line[0].x;
                a2 = center.y - crop_line[0].y;
                a3 = center.x*crop_line[0].x - center.x*center.x + center.y*crop_line[0].y;
                float k , n;
                k = -a2 / a1;
                n = -a3 / a1;
                float a , b , c;
                a = k*k + 1;
                b = 2*k*n - 2*center.x*k - 2*center.y;
                c = n*n - 2*center.x*n + center.x*center.x;
                Point2f pointofContact;
                float y1 , y2;
                y1 = (-b + sqrt(b*b - 4*a*c)) / (2*a);
                y2 = (-b - sqrt(b*b - 4*a*c)) / (2*a);
                pointofContact.y = (y1<y2)?y1:y2;
                pointofContact.x = k*pointofContact.y + n;
                /**/

                Point2f vec1 , vec2;
                vec1 = pointofContact - crop_line[0];
                vec2 = crop_line[1] - crop_line[0];
                float cos_alpha = (vec1.x*vec2.x + vec1.y*vec2.y) / (sqrt(vec1.x*vec1.x+vec1.y*vec1.y) * sqrt(vec2.x*vec2.x+vec2.y*vec2.y));
                double alpha = acos(cos_alpha);

                return alpha;
            }
        }
    }
    else
    {
        if( center.x >= crop_line[1].x )
        {
            return -3.1415926;
        }
        else
        {
            float dis = (float) sqrt(pow(crop_line[1].x - center.x, 2) + pow(crop_line[1].y - center.y, 2) );
            if( dis <= center.y )
            {
                return -3.1415926;
            }
            else
            {
                /*计算切点*/
                float a1 , a2 , a3;
                a1 = center.x - crop_line[1].x;
                a2 = center.y - crop_line[1].y;
                a3 = center.x*crop_line[1].x - center.x*center.x + center.y*crop_line[1].y;
                float k , n;
                k = -a2 / a1;
                n = -a3 / a1;
                float a , b , c;
                a = k*k + 1;
                b = 2*k*n - 2*center.x*k - 2*center.y;
                c = n*n - 2*center.x*n + center.x*center.x;
                Point2f pointofContact;
                float y1 , y2;
                y1 = (-b + sqrt(b*b - 4*a*c)) / (2*a);
                y2 = (-b - sqrt(b*b - 4*a*c)) / (2*a);
                pointofContact.y = (y1<y2)?y1:y2;
                pointofContact.x = k*pointofContact.y + n;
                /**/

                Point2f vec1 , vec2;
                vec1 = pointofContact - crop_line[1];
                vec2 = crop_line[0] - crop_line[1];
                float cos_alpha = (vec1.x*vec2.x + vec1.y*vec2.y) / (sqrt(vec1.x*vec1.x+vec1.y*vec1.y) * sqrt(vec2.x*vec2.x+vec2.y*vec2.y));
                double alpha = acos(cos_alpha);

                return -alpha;
            }
        }
    }
}
void ThreadCompensation::WriteToFile(FILE* old_file, cv::Mat mat) {
    char content_before[600];
    sprintf(content_before, "%d %f %f %f %f %f %f %f %f %f\n", frame_count,
            mat.at<double>(0, 0), mat.at<double>(0, 1), mat.at<double>(0, 2),
            mat.at<double>(1, 0), mat.at<double>(1, 1), mat.at<double>(1, 2),
            mat.at<double>(2, 0), mat.at<double>(2, 1), mat.at<double>(2, 2));
    fwrite(content_before, sizeof(char), strlen(content_before), old_file);
    frame_count++;
}
cv::Vec2f ThreadCompensation::CalTranslationByR(cv::Mat r) {
    std::vector<cv::Point2f> feature = threads::ThreadContext::feature_by_r_.front();
    threads::ThreadContext::feature_by_r_.pop();
    float t_x = 0;
    float t_y = 0;
    for(auto p : feature){
        cv::Mat p1 = (cv::Mat_<double>(3, 1) << p.x, p.y, 1.0f);
        cv::Mat p2 = inmat * r * inmat.inv() * p1;

        t_x += (p2.at<double>(0, 0) - p1.at<double>(0, 0));
        t_y += (p2.at<double>(1, 0) - p1.at<double>(1, 0));
    }
    if(feature.size()){
        return cv::Vec2f(t_x/feature.size(), t_y/feature.size());
    } else {
        return cv::Vec2f(0, 0);
    }


}

cv::Mat ThreadCompensation::limit_Mat(cv::Mat homo)
{
    cv::Mat r_h = homo.clone();
    cv::Mat perp, sca, shear, rot, trans;
    decomposeHomo(homo, center, perp, sca, shear, rot, trans);

    Mat M1=(cv::Mat_<double>(3, 3) <<
            1.0, 0.0, -center.x,
            0.0, 1.0, -center.y,
            0.0, 0.0, 1.0);
    Mat M2=(cv::Mat_<double>(3, 3) <<
            1.0, 0.0, center.x,
            0.0, 1.0, center.y,
            0.0, 0.0, 1.0);

    perp = M2.inv() * perp * M1.inv();
    sca = M2.inv() * sca * M1.inv();
    shear = M2.inv() * shear * M1.inv();
    rot = M2.inv() * rot * M1.inv();
    double theta = asin(rot.at<double>(1,0));
    LOGI("decom result: trans: %f, %f; perp: %f, %f; sca: %f, %f; shear: %f; rot: %f",
            trans.at<double>(0,2), trans.at<double>(1,2), perp.at<double>(2,0), perp.at<double>(2,1),
            sca.at<double>(0,0), sca.at<double>(1,1), shear.at<double>(0,1), theta);

    if(perp.at<double>(2,0)>0.001)
    {
        perp.at<double>(2,0) = 0.001;
    }
    else if(perp.at<double>(2,0)<-0.001)
    {
        perp.at<double>(2,0) = -0.001;
    }
    if(perp.at<double>(2,1)>0.001)
    {
        perp.at<double>(2,1) = 0.001;
    }
    else if(perp.at<double>(2,1)<-0.001)
    {
        perp.at<double>(2,1) = -0.001;
    }

    if(sca.at<double>(0,0)>1.05)
    {
        sca.at<double>(0,0) = 1.05;
    }
    else if(sca.at<double>(0,0)<0.95)
    {
        sca.at<double>(0,0) = 0.95;
    }
    if(sca.at<double>(1,1)>1.05)
    {
        sca.at<double>(1,1) = 1.05;
    }
    else if(sca.at<double>(1,1)<0.95)
    {
        sca.at<double>(1,1) = 0.95;
    }

    if(shear.at<double>(0,1)>0.02)
    {
        shear.at<double>(0,1) = 0.02;
    }
    else if(shear.at<double>(0,1)<-0.02)
    {
        shear.at<double>(0,1) = -0.02;
    }

    if(theta > PI/3)
    {
        theta = PI/3;
    }
    else if(theta < -PI/3)
    {
        theta = -PI/3;
    }

    perp = M2 * perp * M1;
    sca = M2 * sca * M1;
    shear = M2 * shear * M1;
    rot=(cv::Mat_<double>(3, 3) <<
            cos(theta), -sin(theta), 0.0,
            sin(theta), cos(theta), 0.0,
            0.0, 0.0, 1.0);
    rot = M2 * rot * M1;

    r_h =  trans * rot * shear * sca * perp;

    return r_h;
}

void ThreadCompensation::decomposeHomo(cv::Mat h, Point2f cen, cv::Mat &perp, cv::Mat &sca,
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
