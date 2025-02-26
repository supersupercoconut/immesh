#include "voxel_mapping.hpp"

Eigen::Matrix3d   extR;
Eigen::Vector3d   extT;
Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_ext_R;
Eigen::Matrix<double, 3, 1>  camera_ext_t;
Eigen::Matrix3d cam_k;

inline void image_equalize(cv::Mat &img, int amp)
{
    cv::Mat img_temp;
    cv::Size eqa_img_size = cv::Size(std::max(img.cols * 32.0 / 640, 4.0), std::max(img.cols * 32.0 / 640, 4.0));
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(amp, eqa_img_size);
    // Equalize gray image.
    clahe->apply(img, img_temp);
    img = img_temp;
}

inline cv::Mat equalize_color_image_Ycrcb(cv::Mat &image)
{
    cv::Mat hist_equalized_image;
    cv::cvtColor(image, hist_equalized_image, cv::COLOR_BGR2YCrCb);

    //Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
    std::vector<cv::Mat> vec_channels;
    cv::split(hist_equalized_image, vec_channels);

    //Equalize the histogram of only the Y channel
    // cv::equalizeHist(vec_channels[0], vec_channels[0]);
    image_equalize( vec_channels[0], 1 );
    cv::merge(vec_channels, hist_equalized_image);
    cv::cvtColor(hist_equalized_image, hist_equalized_image, cv::COLOR_YCrCb2BGR);
    return hist_equalized_image;
}

void Voxel_mapping::kitti_log( FILE *fp )
{
    // Eigen::Matrix4d T_lidar_to_cam;
    // T_lidar_to_cam << 0.00042768, -0.999967, -0.0080845, -0.01198, -0.00721062,
    //     0.0080811998, -0.99994131, -0.0540398, 0.999973864, 0.00048594,
    //     -0.0072069, -0.292196, 0, 0, 0, 1.0;
    double          time_stamp = 0;
    Eigen::Matrix4d T_lidar_to_cam;
    T_lidar_to_cam << 0.00554604, -0.999971, -0.00523653, 0.0316362, -0.000379382, 0.00523451, -0.999986, 0.0380934, 0.999985, 0.00554795,
            -0.000350341, 0.409066, 0, 0, 0, 1;

    // V3D rot_ang(Log(state.rot_end));
    MD( 4, 4 ) T;
    T.block< 3, 3 >( 0, 0 ) = state.rot_end;
    T.block< 3, 1 >( 0, 3 ) = state.pos_end;
    T( 3, 0 ) = 0;
    T( 3, 1 ) = 0;
    T( 3, 2 ) = 0;
    T( 3, 3 ) = 1;
    T = T_lidar_to_cam * T * T_lidar_to_cam.inverse();

    Eigen::Matrix3d    camera_rotation = T.block< 3, 3 >( 0, 0 );
    Eigen::Vector3d    camera_translation = T.block< 3, 1 >( 0, 3 );
    Eigen::Quaterniond q( camera_rotation );
    fprintf( fp, "%lf %lf %lf %lf %lf %lf %lf %lf\n", m_last_timestamp_lidar, camera_translation[ 0 ], camera_translation[ 1 ],
             camera_translation[ 2 ], q.x(), q.y(), q.z(), q.w() );
    fflush( fp );
}

void Voxel_mapping::SigHandle( int sig )
{
    m_flg_exit = true;
    ROS_WARN( "catch sig %d", sig );
    m_sig_buffer.notify_all();
}

void Voxel_mapping::dump_lio_state_to_log( FILE *fp )
{
#ifdef USE_IKFOM
    // state_ikfom write_state = kf.get_x();
    V3D rot_ang( Log( state_point.rot.toRotationMatrix() ) );
    fprintf( fp, "%lf ", LidarMeasures.lidar_beg_time - first_lidar_time );
    fprintf( fp, "%lf %lf %lf ", rot_ang( 0 ), rot_ang( 1 ), rot_ang( 2 ) ); // Angle
    fprintf( fp, "%lf %lf %lf ", state_point.pos( 0 ), state_point.pos( 1 ),
             state_point.pos( 2 ) );              // Pos
    fprintf( fp, "%lf %lf %lf ", 0.0, 0.0, 0.0 ); // omega
    fprintf( fp, "%lf %lf %lf ", state_point.vel( 0 ), state_point.vel( 1 ),
             state_point.vel( 2 ) );              // Vel
    fprintf( fp, "%lf %lf %lf ", 0.0, 0.0, 0.0 ); // Acc
    fprintf( fp, "%lf %lf %lf ", state_point.bg( 0 ), state_point.bg( 1 ),
             state_point.bg( 2 ) ); // Bias_g
    fprintf( fp, "%lf %lf %lf ", state_point.ba( 0 ), state_point.ba( 1 ),
             state_point.ba( 2 ) ); // Bias_a
    fprintf( fp, "%lf %lf %lf ", state_point.grav[ 0 ], state_point.grav[ 1 ],
             state_point.grav[ 2 ] ); // Bias_a
    fprintf( fp, "\r\n" );
    fflush( fp );
#else
    V3D rot_ang( Log( state.rot_end ) );
    fprintf( fp, "%lf ", m_Lidar_Measures.lidar_beg_time - m_first_lidar_time );
    fprintf( fp, "%lf %lf %lf ", rot_ang( 0 ), rot_ang( 1 ), rot_ang( 2 ) ); // Angle
    fprintf( fp, "%lf %lf %lf ", state.pos_end( 0 ), state.pos_end( 1 ),
             state.pos_end( 2 ) );                // Pos
    fprintf( fp, "%lf %lf %lf ", 0.0, 0.0, 0.0 ); // omega
    fprintf( fp, "%lf %lf %lf ", state.vel_end( 0 ), state.vel_end( 1 ),
             state.vel_end( 2 ) );                // Vel
    fprintf( fp, "%lf %lf %lf ", 0.0, 0.0, 0.0 ); // Acc
    fprintf( fp, "%lf %lf %lf ", state.bias_g( 0 ), state.bias_g( 1 ),
             state.bias_g( 2 ) ); // Bias_g
    fprintf( fp, "%lf %lf %lf ", state.bias_a( 0 ), state.bias_a( 1 ),
             state.bias_a( 2 ) ); // Bias_a
    fprintf( fp, "%lf %lf %lf ", state.gravity( 0 ), state.gravity( 1 ),
             state.gravity( 2 ) ); // Bias_a
    fprintf( fp, "\r\n" );
    fflush( fp );
#endif
}

void Voxel_mapping::pointBodyToWorld( const PointType &pi, PointType &po )
{
    V3D p_body( pi.x, pi.y, pi.z );

    V3D p_global( state.rot_end * ( m_extR * p_body + m_extT ) + state.pos_end );

    po.x = p_global( 0 );
    po.y = p_global( 1 );
    po.z = p_global( 2 );
    po.intensity = pi.intensity;
}

void Voxel_mapping::frameBodyToWorld( const PointCloudXYZI::Ptr &pi, PointCloudXYZI::Ptr &po )
{
    int pi_size = pi->points.size();
    po->resize( pi_size );
    for ( int i = 0; i < pi_size; i++ )
    {
        /* transform to world frame */
        pointBodyToWorld( pi->points[ i ], po->points[ i ] );
    }
}

void Voxel_mapping::get_NED_transform()
{
    if ( false )
    {
        V3D        grav_vec( -0.0463686846197, -0.194593831897, 0.996038079262 );
        double     gravity_correct_ang = std::acos( grav_vec.dot( V3D( 0, 0, 9.8 ) ) / ( grav_vec.norm() * 9.8 ) );
        AngleAxisd gravity_correct_vec( gravity_correct_ang, ( grav_vec.cross( V3D( 0, 0, 9.8 ) ) ).normalized() );
        // gravity_correct_vec = gravity_correct_vec / gravity_correct_vec.norm() *
        // gravity_correct_ang;
        _gravity_correct_rotM = gravity_correct_vec.toRotationMatrix();

        Eigen::Quaterniond _gravity_correct_q;
        _gravity_correct_q.x() = 0.0983599;
        _gravity_correct_q.y() = 0.00420122;
        _gravity_correct_q.z() = -0.377381;
        _gravity_correct_q.w() = 0.92081;

        _gravity_correct_rotM = _gravity_correct_q.toRotationMatrix().transpose();

        cout << "gravity_correct_rotM: " << _gravity_correct_rotM << endl;
        cout << "corrected gravity: " << grav_vec.transpose() * _gravity_correct_rotM.transpose() << endl;
    }
}

void Voxel_mapping::RGBpointBodyToWorld( PointType const *const pi, PointType *const po )
{
    V3D p_body( pi->x, pi->y, pi->z );
#ifdef USE_IKFOM
    // state_ikfom transfer_state = kf.get_x();
    V3D p_global( state_point.rot * ( state_point.offset_R_L_I * p_body + state_point.offset_T_L_I ) + state_point.pos );
#else
    V3D p_global( state.rot_end * ( m_extR * p_body + m_extT ) + state.pos_end );
#endif

    p_global = _gravity_correct_rotM * p_global;

    po->x = p_global( 0 );
    po->y = p_global( 1 );
    po->z = p_global( 2 );
    po->intensity = pi->intensity;

    float intensity = pi->intensity;
    intensity = intensity - floor( intensity );

    int reflection_map = intensity * 10000;
}

void Voxel_mapping::RGBpointBodyLidarToIMU( PointType const *const pi, PointType *const po )
{
    V3D p_body_lidar( pi->x, pi->y, pi->z );
#ifdef USE_IKFOM
    // state_ikfom transfer_state = kf.get_x();
    V3D p_body_imu( state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I );
#else
    V3D p_body_imu( m_extR * p_body_lidar + m_extT );
#endif

    po->x = p_body_imu( 0 );
    po->y = p_body_imu( 1 );
    po->z = p_body_imu( 2 );
    po->intensity = pi->intensity;
}

void Voxel_mapping::points_cache_collect()
{
    PointVector points_history;
    m_ikdtree.acquire_removed_points( points_history );
    m_points_cache_size = points_history.size();
}

void Voxel_mapping::laser_map_fov_segment()
{
    m_cub_need_rm.clear();
    m_kdtree_delete_counter = 0;
    m_kdtree_delete_time = 0.0;
    pointBodyToWorld( m_XAxis_Point_body, m_XAxis_Point_world );
#ifdef USE_IKFOM
    // state_ikfom fov_state = kf.get_x();
    // V3D pos_LiD = fov_state.pos + fov_state.rot * fov_state.offset_T_L_I;
    V3D pos_LiD = pos_lid;
#else
    V3D pos_LiD = state.pos_end;
#endif
    if ( !m_localmap_Initialized )
    {
        // if (cube_len <= 2.0 * MOV_THRESHOLD * DETECTION_RANGE) throw
        // std::invalid_argument("[Error]: Local Map Size is too small! Please
        // change parameter \"cube_side_length\" to larger than %d in the launch
        // file.\n");
        for ( int i = 0; i < 3; i++ )
        {
            m_LocalMap_Points.vertex_min[ i ] = pos_LiD( i ) - m_cube_len / 2.0;
            m_LocalMap_Points.vertex_max[ i ] = pos_LiD( i ) + m_cube_len / 2.0;
        }
        m_localmap_Initialized = true;
        return;
    }
    // printf("Local Map is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n",
    // LocalMap_Points.vertex_min[0],LocalMap_Points.vertex_max[0],LocalMap_Points.vertex_min[1],LocalMap_Points.vertex_max[1],LocalMap_Points.vertex_min[2],LocalMap_Points.vertex_max[2]);
    float dist_to_map_edge[ 3 ][ 2 ];
    bool  need_move = false;
    for ( int i = 0; i < 3; i++ )
    {
        dist_to_map_edge[ i ][ 0 ] = fabs( pos_LiD( i ) - m_LocalMap_Points.vertex_min[ i ] );
        dist_to_map_edge[ i ][ 1 ] = fabs( pos_LiD( i ) - m_LocalMap_Points.vertex_max[ i ] );
        if ( dist_to_map_edge[ i ][ 0 ] <= MOV_THRESHOLD * DETECTION_RANGE || dist_to_map_edge[ i ][ 1 ] <= MOV_THRESHOLD * DETECTION_RANGE )
            need_move = true;
    }
    if ( !need_move )
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = m_LocalMap_Points;
    float mov_dist = max( ( m_cube_len - 2.0 * MOV_THRESHOLD * DETECTION_RANGE ) * 0.5 * 0.9, double( DETECTION_RANGE * ( MOV_THRESHOLD - 1 ) ) );
    for ( int i = 0; i < 3; i++ )
    {
        tmp_boxpoints = m_LocalMap_Points;
        if ( dist_to_map_edge[ i ][ 0 ] <= MOV_THRESHOLD * DETECTION_RANGE )
        {
            New_LocalMap_Points.vertex_max[ i ] -= mov_dist;
            New_LocalMap_Points.vertex_min[ i ] -= mov_dist;
            tmp_boxpoints.vertex_min[ i ] = m_LocalMap_Points.vertex_max[ i ] - mov_dist;
            m_cub_need_rm.push_back( tmp_boxpoints );
            // printf("Delete Box is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n",
            // tmp_boxpoints.vertex_min[0],tmp_boxpoints.vertex_max[0],tmp_boxpoints.vertex_min[1],tmp_boxpoints.vertex_max[1],tmp_boxpoints.vertex_min[2],tmp_boxpoints.vertex_max[2]);
        }
        else if ( dist_to_map_edge[ i ][ 1 ] <= MOV_THRESHOLD * DETECTION_RANGE )
        {
            New_LocalMap_Points.vertex_max[ i ] += mov_dist;
            New_LocalMap_Points.vertex_min[ i ] += mov_dist;
            tmp_boxpoints.vertex_max[ i ] = m_LocalMap_Points.vertex_min[ i ] + mov_dist;
            m_cub_need_rm.push_back( tmp_boxpoints );
            // printf("Delete Box is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n",
            // tmp_boxpoints.vertex_min[0],tmp_boxpoints.vertex_max[0],tmp_boxpoints.vertex_min[1],tmp_boxpoints.vertex_max[1],tmp_boxpoints.vertex_min[2],tmp_boxpoints.vertex_max[2]);
        }
    }
    m_LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if ( m_cub_need_rm.size() > 0 )
        m_kdtree_delete_counter = m_ikdtree.Delete_Point_Boxes( m_cub_need_rm );
    m_kdtree_delete_time = omp_get_wtime() - delete_begin;
    // printf( "Delete time: %0.6f, delete size: %d\n", m_kdtree_delete_time, m_kdtree_delete_counter );
    // printf("Delete Box: %d\n",int(cub_needrm.size()));
}

void Voxel_mapping::standard_pcl_cbk( const sensor_msgs::PointCloud2::ConstPtr &msg )
{
    if ( !m_lidar_en )
        return;
    m_mutex_buffer.lock();
    // cout<<"got feature"<<endl;
    if ( msg->header.stamp.toSec() < m_last_timestamp_lidar )
    {
        ROS_ERROR( "lidar loop back, clear buffer" );
        m_lidar_buffer.clear();
    }
    // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
    PointCloudXYZI::Ptr ptr( new PointCloudXYZI() );
    m_p_pre->process( msg, ptr );
    m_lidar_buffer.push_back( ptr );
    m_time_buffer.push_back( msg->header.stamp.toSec() );
    m_last_timestamp_lidar = msg->header.stamp.toSec();

    m_mutex_buffer.unlock();
    m_sig_buffer.notify_all();
}

void Voxel_mapping::livox_pcl_cbk( const livox_ros_driver::CustomMsg::ConstPtr &msg )
{
    if ( !m_lidar_en )
        return;
    m_mutex_buffer.lock();
    // ROS_INFO( "get LiDAR, its header time: %.6f", msg->header.stamp.toSec() );
    if ( msg->header.stamp.toSec() < m_last_timestamp_lidar )
    {
        ROS_ERROR( "lidar loop back, clear buffer" );
        m_lidar_buffer.clear();
    }
    // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
    PointCloudXYZI::Ptr ptr( new PointCloudXYZI() );
    m_p_pre->process( msg, ptr );
    m_lidar_buffer.push_back( ptr );
    m_time_buffer.push_back( msg->header.stamp.toSec() );
    m_last_timestamp_lidar = msg->header.stamp.toSec();

    m_mutex_buffer.unlock();
    m_sig_buffer.notify_all();
}

void Voxel_mapping::imu_cbk( const sensor_msgs::Imu::ConstPtr &msg_in )
{
    if ( !m_imu_en )
        return;

    if ( m_last_timestamp_lidar < 0.0 )
        return;
    m_publish_count++;
    // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
    sensor_msgs::Imu::Ptr msg( new sensor_msgs::Imu( *msg_in ) );

    double timestamp = msg->header.stamp.toSec();
    m_mutex_buffer.lock();

    if ( m_last_timestamp_imu > 0.0 && timestamp < m_last_timestamp_imu )
    {
        m_mutex_buffer.unlock();
        m_sig_buffer.notify_all();
        ROS_ERROR( "imu loop back \n" );
        return;
    }
    // old 0.2
    if ( m_last_timestamp_imu > 0.0 && timestamp > m_last_timestamp_imu + 0.4 )
    {
        m_mutex_buffer.unlock();
        m_sig_buffer.notify_all();
        ROS_WARN( "imu time stamp Jumps %0.4lf seconds \n", timestamp - m_last_timestamp_imu );
        return;
    }

    m_last_timestamp_imu = timestamp;

    m_imu_buffer.push_back( msg );
    // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
    m_mutex_buffer.unlock();
    m_sig_buffer.notify_all();
}

void Voxel_mapping::image_cbk(const sensor_msgs::CompressedImageConstPtr &msg)
{

    if(!m_img_en)
        return ;

    m_mutex_buffer.lock();
    if(msg->header.stamp.toSec() < m_last_timestamp_img)
    {
        LOG(ERROR) << "Image loop back, clear buffer";
        m_img_buffer.clear();
    }

    m_last_timestamp_img = msg->header.stamp.toSec();
    cv_bridge::CvImagePtr cv_ptr_compressed = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 );
    double img_rec_time = msg->header.stamp.toSec();
    m_img = cv_ptr_compressed->image;
    cv_ptr_compressed->image.release();

    // 对RGB图像去除畸变
    cv::remap( m_img, m_img, m_ud_map1, m_ud_map2, cv::INTER_LINEAR );
    cv::cvtColor(m_img, m_img_gray, cv::COLOR_RGB2GRAY);
    // 这里尝试不使用均衡化，但是最后生成的颜色还是有些失真
    image_equalize(m_img_gray, 3.0);
    m_img = equalize_color_image_Ycrcb(m_img);
    // 设置数据值
    m_img_buffer.push_back(m_img);
    m_img_time_buffer.push_back(msg->header.stamp.toSec());

//    LOG(INFO) << "[m_img_buffer] : " << m_img_buffer.size();
//    LOG(INFO) << "[m_img_time_buffer] : " << m_img_time_buffer.size();

    m_mutex_buffer.unlock();
    m_sig_buffer.notify_all();
}


bool Voxel_mapping::sync_packages( LidarMeasureGroup &meas )
{
    /* imu + lidar + camera 传感器下的数据集打包策略 */
    if ( m_lidar_buffer.empty() || m_imu_buffer.empty() || m_img_buffer.empty() )
    {
//        LOG(INFO) << " lidar buffer or img buffer or imu buffer is empty ! " ;
        return false;
    }

    // 打包lidar数据 - 先将lidar数据提取出来
    if ( !m_lidar_pushed )
    {
        meas.lidar = m_lidar_buffer.front();
        if ( meas.lidar->points.size() <= 1 )
        {
            m_lidar_buffer.pop_front();
            return false;
        }
        meas.lidar_beg_time = m_time_buffer.front();
        m_lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double( 1000 );
        m_lidar_pushed = true;
    }


    // m_last_timestamp_imu对应的是最后一帧imu数据的时间戳 | m_lidar_end_time对应的部分是一帧点云(加上其开始部分)最后对应的时间部分
    if ( m_imu_en && m_last_timestamp_imu < m_lidar_end_time )
    {
        return false;
    }

    // 管理image | 小于lidar时间, 清空所有数据 | 图像buffer里面不包含时间信息
    if(m_last_timestamp_img < m_time_buffer.front())
    {
        m_img_buffer.clear();
        m_img_time_buffer.clear();  // 时间戳与图像一起处理
//        LOG(INFO) << "clear buffer";
        return false;
    }
    else
    {
        // 上面只是判断最后一帧图像，这里可能有多个图像在 m_img_buffer
        while( !m_img_time_buffer.empty())
        {
            auto i = m_img_time_buffer.front();
            // 之前的数据情况
            if( i < m_time_buffer.front())
            {
                m_img_buffer.pop_front();
                m_img_time_buffer.pop_front();
                continue;
            }
            else
            {
                // 组装LidarMeasurement
                struct MeasureGroup m;
                m.img = m_img_buffer.front();

                double imu_time = m_imu_buffer.front()->header.stamp.toSec();
                m.imu.clear();

                while ( ( !m_imu_buffer.empty() && ( imu_time < m_lidar_end_time ) ) )
                {
                    imu_time = m_imu_buffer.front()->header.stamp.toSec();
                    if ( imu_time > m_lidar_end_time )
                        break;
                    m.imu.push_back( m_imu_buffer.front() );
                    m_imu_buffer.pop_front();
                }

                // 只有所有的部分都完成了,才会继续进行处理
                m_lidar_pushed = false;
                m_lidar_buffer.pop_front();
                m_time_buffer.pop_front();
                m_img_buffer.pop_front();
                m_img_time_buffer.pop_front();

                m_sig_buffer.notify_all();

                meas.is_lidar_end = true; // process lidar topic, so timestamp should be lidar scan end.
                meas.measures.push_back( m );
                return true;
            }
        }
        return false;
    }
//    if ( !m_imu_en )
//    {
//        if ( !m_lidar_buffer.empty() )
//        {
//            meas.lidar = m_lidar_buffer.front();
//            meas.lidar_beg_time = m_time_buffer.front();
//            m_lidar_buffer.pop_front();
//            m_time_buffer.pop_front();
//            return true;
//        }
//
//        return false;
//    }
//
//    if ( m_lidar_buffer.empty() || m_imu_buffer.empty() )
//    {
//        return false;
//    }
//    /*** push a lidar scan ***/
//    if ( !m_lidar_pushed )
//    {
//        meas.lidar = m_lidar_buffer.front();
//        if ( meas.lidar->points.size() <= 1 )
//        {
//            m_lidar_buffer.pop_front();
//            return false;
//        }
//        meas.lidar_beg_time = m_time_buffer.front();
//        m_lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double( 1000 );
//        m_lidar_pushed = true;
//    }
//    if ( m_last_timestamp_imu < m_lidar_end_time )
//    {
//        return false;
//    }
//    /*** push imu data, and pop from imu buffer ***/
//
//    // no img topic, means only has lidar topic
//    if ( m_imu_en && m_last_timestamp_imu < m_lidar_end_time )
//    { // imu message needs to be larger than lidar_end_time, keep complete propagate.
//        // ROS_ERROR("out sync");
//        return false;
//    }
//    struct MeasureGroup m; // standard method to keep imu message.
//    if ( !m_imu_buffer.empty() )
//    {
//        double imu_time = m_imu_buffer.front()->header.stamp.toSec();
//        m.imu.clear();
//        m_mutex_buffer.lock();
//        while ( ( !m_imu_buffer.empty() && ( imu_time < m_lidar_end_time ) ) )
//        {
//            imu_time = m_imu_buffer.front()->header.stamp.toSec();
//            if ( imu_time > m_lidar_end_time )
//                break;
//            m.imu.push_back( m_imu_buffer.front() );
//            m_imu_buffer.pop_front();
//        }
//    }
//    m_lidar_buffer.pop_front();
//    m_time_buffer.pop_front();
//    m_mutex_buffer.unlock();
//    m_sig_buffer.notify_all();
//    m_lidar_pushed = false;   // sync one whole lidar scan.
//    meas.is_lidar_end = true; // process lidar topic, so timestamp should be lidar scan end.
//    meas.measures.push_back( m );
//
//    return true;
}

void Voxel_mapping::publish_voxel_point( const ros::Publisher &pubLaserCloudVoxel, const PointCloudXYZI::Ptr &pcl_wait_pub )
{
    uint                  size = pcl_wait_pub->points.size();
    PointCloudXYZRGB::Ptr laserCloudWorldRGB( new PointCloudXYZRGB( size, 1 ) );
    for ( int i = 0; i < size; i++ )
    {
        PointTypeRGB pointRGB;

        pointRGB.x = pcl_wait_pub->points[ i ].x;
        pointRGB.y = pcl_wait_pub->points[ i ].y;
        pointRGB.z = pcl_wait_pub->points[ i ].z;

        V3D point( pointRGB.x, pointRGB.y, pointRGB.z );
        V3F pixel = RGBFromVoxel( point, m_max_voxel_size, m_layer_size, m_min_eigen_value, m_feat_map );
        pointRGB.r = pixel[ 0 ];
        pointRGB.g = pixel[ 1 ];
        pointRGB.b = pixel[ 2 ];
        laserCloudWorldRGB->push_back( pointRGB );
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    if ( m_img_en )
    {
//        cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
        pcl::toROSMsg( *laserCloudWorldRGB, laserCloudmsg );
    }
    else
    {
        pcl::toROSMsg( *pcl_wait_pub, laserCloudmsg );
    }
    laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudVoxel.publish( laserCloudmsg );
    m_publish_count -= PUBFRAME_PERIOD;
}

void Voxel_mapping::publish_visual_world_map( const ros::Publisher &pubVisualCloud )
{
    PointCloudXYZI::Ptr laserCloudFullRes( m_map_cur_frame_point );
    int                 size = laserCloudFullRes->points.size();
    if ( size == 0 )
        return;
    // PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

    // for (int i = 0; i < size; i++)
    // {
    //     RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
    //                         &laserCloudWorld->coints[i]);
    // }
    // mutex_buffer_pointcloud.lock();
    *m_pcl_visual_wait_pub = *laserCloudFullRes;
    if ( 1 ) // if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg( *m_pcl_visual_wait_pub, laserCloudmsg );
        laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubVisualCloud.publish( laserCloudmsg );
        m_publish_count -= PUBFRAME_PERIOD;
        // pcl_wait_pub->clear();
    }
    // mutex_buffer_pointcloud.unlock();
}

void Voxel_mapping::publish_visual_world_sub_map( const ros::Publisher &pubSubVisualCloud )
{
    PointCloudXYZI::Ptr laserCloudFullRes( m_sub_map_cur_frame_point );
    int                 size = laserCloudFullRes->points.size();
    if ( size == 0 )
        return;
    // PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

    // for (int i = 0; i < size; i++)
    // {
    //     RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
    //                         &laserCloudWorld->points[i]);
    // }
    // mutex_buffer_pointcloud.lock();
    *m_sub_pcl_visual_wait_pub = *laserCloudFullRes;
    if ( 1 ) // if(publish_count >= PUBFRAME_PERIOD)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg( *m_sub_pcl_visual_wait_pub, laserCloudmsg );
        laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubSubVisualCloud.publish( laserCloudmsg );
        m_publish_count -= PUBFRAME_PERIOD;
        // pcl_wait_pub->clear();
    }
    // mutex_buffer_pointcloud.unlock();
}

void Voxel_mapping::publish_effect_world( const ros::Publisher &pubLaserCloudEffect )
{
    PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI( m_effct_feat_num, 1 ) );
    for ( int i = 0; i < m_effct_feat_num; i++ )
    {
        RGBpointBodyToWorld( &m_laserCloudOri->points[ i ], &laserCloudWorld->points[ i ] );
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg( *laserCloudWorld, laserCloudFullRes3 );
    laserCloudFullRes3.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish( laserCloudFullRes3 );
}

void Voxel_mapping::publish_map( const ros::Publisher &pubLaserCloudMap )
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg( *m_featsFromMap, laserCloudMap );
    laserCloudMap.header.stamp = ros::Time::now();
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish( laserCloudMap );
}

void Voxel_mapping::publish_odometry( const ros::Publisher &pubOdomAftMapped )
{
    m_odom_aft_mapped.header.frame_id = "camera_init";
    m_odom_aft_mapped.child_frame_id = "aft_mapped";
    m_odom_aft_mapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
    set_pose_timestamp( m_odom_aft_mapped.pose.pose );
    // odomAftMapped.twist.twist.linear.x = state_point.vel(0);
    // odomAftMapped.twist.twist.linear.y = state_point.vel(1);
    // odomAftMapped.twist.twist.linear.z = state_point.vel(2);
    // if (Measures.imu.size()>0) {
    //     Vector3d tmp(Measures.imu.back()->angular_velocity.x,
    //     Measures.imu.back()->angular_velocity.y,Measures.imu.back()->angular_velocity.z);
    //     odomAftMapped.twist.twist.angular.x = tmp[0] - state_point.bg(0);
    //     odomAftMapped.twist.twist.angular.y = tmp[1] - state_point.bg(1);
    //     odomAftMapped.twist.twist.angular.z = tmp[2] - state_point.bg(2);
    // }
    // static tf::TransformBroadcaster br;
    // tf::Transform                   transform;
    // tf::Quaternion                  q;
    // transform.setOrigin(tf::Vector3(state.pos_end(0), state.pos_end(1),
    // state.pos_end(2))); q.setW(geoQuat.w); q.setX(geoQuat.x);
    // q.setY(geoQuat.y);
    // q.setZ(geoQuat.z);
    // transform.setRotation( q );
    // br.sendTransform( tf::StampedTransform( transform,
    // odomAftMapped.header.stamp, "camera_init", "aft_mapped" ) );
    pubOdomAftMapped.publish( m_odom_aft_mapped );
}

void Voxel_mapping::publish_mavros( const ros::Publisher &mavros_pose_publisher )
{
    m_msg_body_pose.header.stamp = ros::Time::now();
    m_msg_body_pose.header.frame_id = "camera_odom_frame";
    set_pose_timestamp( m_msg_body_pose.pose );
    mavros_pose_publisher.publish( m_msg_body_pose );
}

void Voxel_mapping::publish_frame_world( const ros::Publisher &pubLaserCloudFullRes, const int point_skip )
{
    PointCloudXYZI::Ptr laserCloudFullRes( m_dense_map_en ? m_feats_undistort : m_feats_down_body );
    int                 size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI( size, 1 ) );
    for ( int i = 0; i < size; i++ )
    {
        RGBpointBodyToWorld( &laserCloudFullRes->points[ i ], &laserCloudWorld->points[ i ] );
    }
    PointCloudXYZI::Ptr laserCloudWorldPub( new PointCloudXYZI );
    for ( int i = 0; i < size; i += point_skip )
    {
        laserCloudWorldPub->points.push_back( laserCloudWorld->points[ i ] );
    }
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg( *laserCloudWorldPub, laserCloudmsg );
    laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish( laserCloudmsg );
}

void Voxel_mapping::publish_path( const ros::Publisher pubPath )
{
    set_pose_timestamp( m_msg_body_pose.pose );
    m_msg_body_pose.header.stamp = ros::Time::now();
    m_msg_body_pose.header.frame_id = "camera_init";
    m_pub_path.poses.push_back( m_msg_body_pose );
    pubPath.publish( m_pub_path );
}

void Voxel_mapping::read_ros_parameters( ros::NodeHandle &nh )
{
    nh.param< int >( "dense_map_enable", m_dense_map_en, 1 );
    nh.param< int >( "lidar_enable", m_lidar_en, 1 );
    nh.param< int >( "debug", m_debug, 0 );
    nh.param< int >( "max_iteration", NUM_MAX_ITERATIONS, 4 );
    nh.param< int >( "min_img_count", MIN_IMG_COUNT, 150000 );
    nh.param< string >( "pc_name", m_pointcloud_file_name, " " );

    nh.param< int >( "gui_font_size", m_GUI_font_size, 16 );

//    nh.param< double >( "cam_fx", cam_fx, 453.483063 );
//    nh.param< double >( "cam_fy", cam_fy, 453.254913 );
//    nh.param< double >( "cam_cx", cam_cx, 318.908851 );
//    nh.param< double >( "cam_cy", cam_cy, 234.238189 );

    nh.param< double >( "laser_point_cov", LASER_POINT_COV, 0.001 );
    nh.param< double >( "img_point_cov", IMG_POINT_COV, 1000 );
    nh.param< string >( "map_file_path", m_map_file_path, "" );

    nh.param< string >( "hilti/seq", m_hilti_seq_name, "01" );
    nh.param< bool >( "hilti/en", m_hilti_en, false );
//    nh.param< string >( "camera/img_topic", m_img_topic, "/usb_cam/image_raw" );
    nh.param< double >( "filter_size_corner", m_filter_size_corner_min, 0.5 );
    nh.param< double >( "filter_size_surf", m_filter_size_surf_min, 0.4 );
    nh.param< double >( "filter_size_map", m_filter_size_map_min, 0.4 );
    nh.param< double >( "cube_side_length", m_cube_len, 1000 );
    nh.param< double >( "mapping/fov_degree", m_fov_deg, 180 );
    nh.param< double >( "mapping/gyr_cov", m_gyr_cov, 0.3 );
    nh.param< double >( "mapping/acc_cov", m_acc_cov, 0.5 );
    nh.param< int >( "voxel/max_points_size", m_max_points_size, 100 );
    nh.param< int >( "voxel/max_layer", m_max_layer, 2 );

    nh.param< vector< int > >( "voxel/layer_init_size", m_layer_init_size, vector< int >({5,5,5,5,5}) );
    nh.param< int >( "mapping/imu_int_frame", m_imu_int_frame, 30 );
    nh.param< bool >( "mapping/imu_en", m_imu_en, true );
    nh.param< bool >( "voxel/voxel_map_en", m_use_new_map, true );
    nh.param< bool >( "voxel/pub_plane_en", m_is_pub_plane_map, true );
    nh.param< double >( "voxel/match_eigen_value", m_match_eigen_value, 0.0025 );
    nh.param< int >( "voxel/layer", m_voxel_layer, 1 );
    nh.param< double >( "voxel/match_s", m_match_s, 0.90 );
    nh.param< double >( "voxel/voxel_size", m_max_voxel_size, 0.5 );
    nh.param< double >( "voxel/min_eigen_value", m_min_eigen_value, 0.01 );
    nh.param< double >( "voxel/sigma_num", m_sigma_num, 3 );
    nh.param< double >( "voxel/beam_err", m_beam_err, 0.02 );
    nh.param< double >( "voxel/dept_err", m_dept_err, 0.05 );
    nh.param< double >( "preprocess/blind", m_p_pre->blind, 1 );
    nh.param< double >( "image_save/rot_dist", m_keyf_rotd, 0.01 );
    nh.param< double >( "image_save/pos_dist", m_keyf_posd, 0.01 );
    nh.param< int >( "preprocess/scan_line", m_p_pre->N_SCANS, 6 );
    nh.param< int >( "preprocess/timestamp_unit", m_p_pre->time_unit, US );
    nh.param< bool >( "preprocess/calib_laser", m_p_pre->calib_laser, true );
    nh.param< int >( "point_filter_num", m_p_pre->point_filter_num, 2 );
    nh.param< int >( "pcd_save/interval", m_pcd_save_interval, -1 );
    nh.param< int >( "image_save/interval", m_img_save_interval, 1 );
    nh.param< int >( "pcd_save/type", m_pcd_save_type, 0 );
    nh.param< bool >( "pcd_save/pcd_save_en", m_pcd_save_en, false );
    nh.param< bool >( "image_save/img_save_en", m_img_save_en, false );
    nh.param< bool >( "feature_extract_enable", m_p_pre->feature_enabled, false );

    ///////////////////////////// 注意这里的参数可能会在后面的部分被修改了 因为没有使用ros来读取yaml参数, 只好手动读取参数值 ///////////////////
    nh.param< vector< double > >( "mapping/extrinsic_T", m_extrin_T, vector< double >() );
    nh.param< vector< double > >( "mapping/extrinsic_R", m_extrin_R, vector< double >() );
    // 后面的这两个参数应该没有被使用
    nh.param< vector< double > >( "camera/Pcl", m_camera_extrin_T, vector< double >() );
    nh.param< vector< double > >( "camera/Rcl", m_camera_extrin_R, vector< double >() );

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    nh.param< int >( "grid_size", m_grid_size, 40 );
    nh.param< int >( "patch_size", m_patch_size, 8 );
    nh.param< double >( "outlier_threshold", m_outlier_threshold, 78 );
    nh.param< bool >( "publish/effect_point_pub", m_effect_point_pub, false );
    nh.param< int >( "publish/pub_point_skip", m_pub_point_skip, 1 );
    nh.param< double >( "meshing/distance_scale", m_meshing_distance_scale, 0.5 );
    nh.param< double >( "meshing/points_minimum_scale", m_meshing_points_minimum_scale, 0.1 );
    nh.param< double >( "meshing/voxel_resolution", m_meshing_voxel_resolution, 0.5 );
    nh.param< double >( "meshing/region_size", m_meshing_region_size, 10.0 );
    nh.param< int >( "meshing/if_draw_mesh", m_if_draw_mesh, 1.0 );
    nh.param< int >( "meshing/enable_mesh_rec", m_if_enable_mesh_rec, 1 );
    nh.param< int >( "meshing/maximum_thread_for_rec_mesh", m_meshing_maximum_thread_for_rec_mesh, 32 );
    nh.param< int >( "meshing/number_of_pts_append_to_map", m_meshing_number_of_pts_append_to_map, 100000 );


    /////////////////// 补充 /////////////////////////
    LOG(INFO) << "Loading camera parameter";
    nh.param< int >( "img_used", m_img_en, 1 );
    nh.param< string >( "common/lid_topic", m_lid_topic, "/livox/lidar" );
    nh.param< string >( "common/imu_topic", m_imu_topic, "/livox/imu" );
    nh.param< int >( "preprocess/lidar_type", m_p_pre->lidar_type,  1);         // 1代表是avia雷达据
    nh.param<string>("image/image_topic", m_img_topic, "/camera/color/image_raw/compressed" );
    m_bag_file = "/home/supercoconut/Myfile/datasets/R3LIVE/hku_campus_seq_03.bag";

    // 相机内参(后期可以更换成ros中读取参数)
    std::vector< double > camera_intrinsic_data, camera_dist_coeffs_data, camera_ext_R_data, camera_ext_t_data;

    /* m2DGR数据集 */
//    m_lid_topic = "/velodyne_points";
//    m_imu_topic = "/camera/imu";
//    m_p_pre->lidar_type = 2;
//    nh.param<string>("image/image_topic", m_img_topic, "/camera/color/image_raw/compressed" );
//    m_bag_file = "/home/supercoconut/Myfile/datasets/m2DGR/walk_01.bag";
//
//    int width = 640;
//    int height = 480;
//    camera_intrinsic_data = { 617.971050917033,0.0,327.710279392468,
//                              0.0, 616.445131524790, 253.976983707814,
//                              0.0, 0.0, 1};
//    camera_dist_coeffs_data = { 0.148000794688248, -0.217835187249065, 0.0, 0.0 ,0.0};
//    // Lidar到camera的旋转+平移
//    camera_ext_R_data ={0, 0, 1,
//                        -1, 0, 0,
//                        0, -1, 0};
//
//    camera_ext_t_data = {0.30456, 0.00065, 0.65376};
//    m_extrin_R = {1,0,0,
//                  0,1,0,
//                  0,0,1};
//    m_extrin_T = {0,0,0};
    /*** /////////////////// ***/

    // /*r3live数据集*/
    m_img_topic = "/camera/image_color/compressed";
    int width = 1280;
    int height = 1024;
    camera_intrinsic_data = { 863.4241, 0.0, 640.6808,
                              0.0,  863.4171, 518.3392,
                              0.0, 0.0, 1.0};
    camera_dist_coeffs_data = { -0.1080, 0.1050, -1.2872e-04, 5.7923e-05, -0.0222};
    // Lidar到camera的旋转+平移
    camera_ext_R_data ={-0.00113207, -0.0158688, 0.999873,
                        -0.9999999,  -0.000486594, -0.00113994,
                        0.000504622,  -0.999874,  -0.0158682};

    camera_ext_t_data = {0.0, 0.0, 0.0};
    /*** lidar转换到imu ***/
    m_extrin_R = {1,0,0,
                  0,1,0,
                  0,0,1};

    m_extrin_T = {0.04165, 0.02326, -0.0284};

    /* M3DGR数据集 */
    // m_lid_topic = "/livox/avia/lidar";
    // m_imu_topic = "/livox/avia/imu";
    // m_img_topic = "/camera/color/image_raw/compressed";
    // int width = 640;
    // int height = 480;
    // camera_intrinsic_data = {607.79772949, 0.0, 328.797943115234,
    //                           0.0,  607.835266113281, 245.53321838789,
    //                           0.0, 0.0, 1.0};
    // camera_dist_coeffs_data = { 0.00, 0.00, 0.0, 0.0, 0.0};
    // // Lidar到camera的旋转+平移
    // // camera_ext_R_data ={0.02108, -0.00387, 0.9997703,
    // //                     -0.9994,  0.02752, 0.02118,
    // //                     -0.02760, -0.99961, -0.00329};
    // // 2-10 手动调试参数
    // // camera_ext_R_data ={-0.0247904,  0.0023921,  0.9996898,
    // // -0.9992074,  0.0310943, -0.0248528,
    // // -0.0311441, -0.9995136,  0.0016194 };
    // // camera_ext_t_data = {-0.20649940639000343, 0.7369323900722907, -0.40486292124905459};
    //
    // camera_ext_R_data ={-0.0670257, -0.0473156,  0.9966287,
    //     -0.9973689,  0.0308274, -0.0656119,
    //     -0.0276190, -0.9984042, -0.0492573};
    //
    // camera_ext_t_data = {-0.7594152828450921,  2.6594789842645086, 2.5159170074605206};

    /* M3DGR 图像放大版本 */
    // int width = 1280;
    // int height = 720;
    // camera_intrinsic_data = {911.696533203125, 0.0, 653.196960449219,
    //                           0.0,  911.7528652344, 368.299835205078,
    //                           0.0, 0.0, 1.0};
    // camera_dist_coeffs_data = { 0.00, 0.00, 0.0, 0.0, 0.0};
    // // Lidar到camera的旋转+平移
    // camera_ext_R_data ={ -0.0134829,  0.0162040,  0.9997778,
    //                         -0.9994130,  0.0312761, -0.0139849,
    //                         -0.0314957, -0.9993794,  0.0157728};
    //
    // camera_ext_t_data = {-0.11477650734217243, -0.03333126591191406, -0.22590150443123};


    /*** lidar转换到imu ***/
    m_extrin_R = {1,0,0,
                  0,1,0,
                  0,0,1};

    m_extrin_T = {0.04165, 0.02326, -0.0284};

    /*** /////////////////// ***/

    /*Global中的类成员读取参数 */
    m_camera_intrinsic = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( camera_intrinsic_data.data() );
    m_camera_dist_coeffs = Eigen::Map< Eigen::Matrix< double, 5, 1 > >( camera_dist_coeffs_data.data() );
    /*** camera转换到lidar ***/
    m_camera_ext_R = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( camera_ext_R_data.data() );
    m_camera_ext_t = Eigen::Map< Eigen::Matrix< double, 3, 1 > >( camera_ext_t_data.data() );

    cv::eigen2cv(m_camera_intrinsic, intrinsic);
    cv::eigen2cv(m_camera_dist_coeffs, dist_coeffs);
    /*设置去畸变参数等等*/
    initUndistortRectifyMap( intrinsic, dist_coeffs, cv::Mat(), intrinsic, cv::Size(width, height),
                             CV_16SC2, m_ud_map1, m_ud_map2 );

//    g_cam_k <<  intrinsic[ 0 ], intrinsic[ 1 ], intrinsic[ 2 ],
//            intrinsic[ 3 ], intrinsic[ 4 ], intrinsic[ 5 ],
//            intrinsic[ 6 ], intrinsic[ 7 ], intrinsic[ 8 ];

    g_cam_k = m_camera_intrinsic;

    LOG(INFO) << "g_cam_k: ";
    LOG(INFO) << g_cam_k;


    LOG(INFO) << "Subscribing to topic: " << m_lid_topic.c_str();
    LOG(INFO) << "Subscribing to topic: " << m_imu_topic.c_str();
    LOG(INFO) << "Subscribing to topic: " << m_img_topic.c_str();

    LOG(INFO) << "m_p_pre->lidar_type: " << m_p_pre->lidar_type;

    cout << "[Ros_parameter]: Camera Intrinsic: " << endl;
    cout << m_camera_intrinsic << endl;
    cout << "[Ros_parameter]: Camera distcoeff: " << m_camera_dist_coeffs.transpose() << endl;
    cout << "[Ros_parameter]: Camera extrinsic R(camera to lidar): " << endl;
    cout << m_camera_ext_R << endl;
    cout << "[Ros_parameter]: Camera extrinsic T(camera to lidar): " << m_camera_ext_t.transpose() << endl;
    /////////////////////////////////////////////////

    m_p_pre->blind_sqr = m_p_pre->blind * m_p_pre->blind;
    cout << "Ranging cov:" << m_dept_err << " , angle cov:" << m_beam_err << std::endl;
    cout << "Meshing distance scale:" << m_meshing_distance_scale << " , points minimum scale:" << m_meshing_points_minimum_scale << std::endl;

    //// 将部分内外参数据直接给Global map ( 因为之前的投影线程是在voxelmap中，而且验证了其在r3live的数据集中是可以使用的, 这里直接给Global map赋值) )
//    g_map_rgb_pts_mesh.m_extR << MAT_FROM_ARRAY( m_extrin_R );
//    g_map_rgb_pts_mesh.m_extT << VEC_FROM_ARRAY( m_extrin_T );
//    g_map_rgb_pts_mesh.m_camera_ext_R = m_camera_ext_R;
//    g_map_rgb_pts_mesh.m_camera_ext_t = m_camera_ext_t;

    extR << MAT_FROM_ARRAY( m_extrin_R );
    extT << VEC_FROM_ARRAY( m_extrin_T );
    camera_ext_R = m_camera_ext_R;
    camera_ext_t = m_camera_ext_t;
    cam_k = g_cam_k;
}

void Voxel_mapping::transformLidar( const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                                    pcl::PointCloud< pcl::PointXYZI >::Ptr &trans_cloud )
{
    trans_cloud->clear();
    for ( size_t i = 0; i < input_cloud->size(); i++ )
    {
        pcl::PointXYZINormal p_c = input_cloud->points[ i ];
        Eigen::Vector3d      p( p_c.x, p_c.y, p_c.z );
        // p = rot * p + t;
        p = ( rot * ( m_extR * p + m_extT ) + t );
        pcl::PointXYZI pi;
        pi.x = p( 0 );
        pi.y = p( 1 );
        pi.z = p( 2 );
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back( pi );
    }
}


void Voxel_mapping::image_equalize(cv::Mat &img, int amp)
{
    cv::Mat img_temp;
    cv::Size eqa_img_size = cv::Size(std::max(img.cols * 32.0 / 640, 4.0), std::max(img.cols * 32.0 / 640, 4.0));
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(amp, eqa_img_size);
    // Equalize gray image.
    clahe->apply(img, img_temp);
    img = img_temp;
}



double last_accept_time = 0;
int    buffer_max_frame = 0;
int    total_frame_count = 0;
void Voxel_mapping::process_image(cv::Mat &temp_img, double msg_time)
{
    cv::Mat img_get;
    if ( temp_img.rows == 0 )
    {
        cout << "Process image error, image rows =0 " << endl;
        return;
    }
    if ( msg_time < last_accept_time )
    {
        cout << "Error, image time revert!!" << endl;
        return;
    }
    last_accept_time = msg_time;

    if ( m_camera_start_ros_tim < 0 )
    {
        m_camera_start_ros_tim = msg_time;
    }

    img_get = temp_img;

    std::shared_ptr< Image_frame > img_pose = std::make_shared< Image_frame >( g_cam_k );
    cv::remap( img_get, img_pose->m_img, m_ud_map1, m_ud_map2, cv::INTER_LINEAR );
    img_pose->m_timestamp = msg_time;
    img_pose->init_cubic_interpolation();   // 这个函数只是设置了一些基本信息(剩下所有Image_frame的长宽)
    img_pose->image_equalize();
    // 需要重新找一个容器用于存放数据
//    m_camera_data_mutex.lock();
//    m_queue_image_with_pose.push_back( img_pose );
//    m_camera_data_mutex.unlock();

    total_frame_count++;
}
