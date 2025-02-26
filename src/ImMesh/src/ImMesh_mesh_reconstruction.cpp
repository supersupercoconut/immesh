#include "voxel_mapping.hpp"
#include "meshing/mesh_rec_display.hpp"
#include "meshing/mesh_rec_geometry.hpp"
#include "tools/tools_thread_pool.hpp"

extern Global_map       g_map_rgb_pts_mesh;
extern Triangle_manager g_triangles_manager;
extern int              g_current_frame;

extern double                       minimum_pts;
extern double                       g_meshing_voxel_size;
extern FILE *                       g_fp_cost_time;
extern FILE *                       g_fp_lio_state;
extern bool                         g_flag_pause;
extern const int                    number_of_frame;
extern int                          appending_pts_frame;
extern LiDAR_frame_pts_and_pose_vec g_eigen_vec_vec;

int        g_maximum_thread_for_rec_mesh;
std::mutex g_mutex_append_map;
std::mutex g_mutex_reconstruct_mesh;

extern double g_LiDAR_frame_start_time;
double        g_vx_map_frame_cost_time;
static double g_LiDAR_frame_avg_time;

namespace{
    const double image_obs_cov = 1.5;
    const double process_noise_sigma = 0.15;
}

namespace {
    std::unique_ptr<std::thread> g_pub_thr = nullptr;
    int flag = 0;
}

struct Rec_mesh_data_package
{
    pcl::PointCloud< pcl::PointXYZI >::Ptr m_frame_pts;
    Eigen::Quaterniond                     m_pose_q;
    Eigen::Vector3d                        m_pose_t;
    int                                    m_frame_idx;
    Rec_mesh_data_package( pcl::PointCloud< pcl::PointXYZI >::Ptr frame_pts, Eigen::Quaterniond pose_q, Eigen::Vector3d pose_t, int frame_idx )
    {
        m_frame_pts = frame_pts;
        m_pose_q = pose_q;
        m_pose_t = pose_t;
        m_frame_idx = frame_idx;
    }
};

std::mutex                                  g_mutex_data_package_lock;
std::list< Rec_mesh_data_package >          g_rec_mesh_data_package_list;
std::shared_ptr< Common_tools::ThreadPool > g_thread_pool_rec_mesh = nullptr;

extern int                                  g_enable_mesh_rec;
extern int                                  g_save_to_offline_bin;

LiDAR_frame_pts_and_pose_vec                                                                               g_ponintcloud_pose_vec;


void incremental_mesh_reconstruction( pcl::PointCloud< pcl::PointXYZI >::Ptr frame_pts, cv::Mat img, Eigen::Quaterniond pose_q, Eigen::Vector3d pose_t, int frame_idx )
{
    // std::cout << "frame_idx: "<< frame_idx << std::endl;
    while ( g_flag_pause )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    }

    Eigen::Matrix< double, 7, 1 > pose_vec;
    pose_vec.head< 4 >() = pose_q.coeffs().transpose();
    pose_vec.block( 4, 0, 3, 1 ) = pose_t;
    for ( int i = 0; i < frame_pts->points.size(); i++ )
    {
        g_eigen_vec_vec[ frame_idx ].first.emplace_back( frame_pts->points[ i ].x, frame_pts->points[ i ].y, frame_pts->points[ i ].z,
                                                         frame_pts->points[ i ].intensity );
    }
    g_eigen_vec_vec[ frame_idx ].second = pose_vec;
    // g_eigen_vec_vec.push_back( std::make_pair( empty_vec, pose_vec ) );
    // TODO : add time tic toc

    int                 append_point_step = std::max( ( int ) 1, ( int ) std::round( frame_pts->points.size() / appending_pts_frame ) );
    Common_tools::Timer tim, tim_total, tim_append, time_mesh, tim_render;


    tim_append.tic();
    int acc = 0;
    int rej = 0;
    std::unordered_set< std::shared_ptr< RGB_Voxel > > voxels_recent_visited;
    voxels_recent_visited.clear();

    int pt_size = frame_pts->points.size();
    KDtree_pt_vector     pt_vec_vec;
    std::vector< float > dist_vec;

    RGB_voxel_ptr* temp_box_ptr_ptr;
    double minimum_pts_size = g_map_rgb_pts_mesh.m_minimum_pts_size;
    double voxel_resolution = g_map_rgb_pts_mesh.m_voxel_resolution;

    g_mutex_append_map.lock();
    for ( long pt_idx = 0; pt_idx < pt_size; pt_idx += append_point_step )
    {
        int  add = 1;
        int  grid_x = std::round( frame_pts->points[ pt_idx ].x / minimum_pts_size );
        int  grid_y = std::round( frame_pts->points[ pt_idx ].y / minimum_pts_size );
        int  grid_z = std::round( frame_pts->points[ pt_idx ].z / minimum_pts_size );
        int  box_x = std::round( frame_pts->points[ pt_idx ].x / voxel_resolution );
        int  box_y = std::round( frame_pts->points[ pt_idx ].y / voxel_resolution );
        int  box_z = std::round( frame_pts->points[ pt_idx ].z / voxel_resolution );
        auto pt_ptr = g_map_rgb_pts_mesh.m_hashmap_3d_pts.get_data( grid_x, grid_y, grid_z );
        if ( pt_ptr != nullptr )
            add = 0;

        /// @bug 这里的 box_ptr也会出现 {use count 1811941585 weak count 32762} 这种情况 | 这里不是上锁导致的问题, 因为单线程运行这部分的程序, 程序一样出现问题！！！ 原因是是这里的hashmap_voxels中的数据出现了问题 !!!
        /// @bug Hash表读取部分出现bug insert数据的时候不会出现问题,但是在get_data的时候获取到的shared_ptr出现问题 —— 根据不断的地注释发现是因为mesh重建中的一些部分代码影响了这里的数据
        RGB_voxel_ptr box_ptr;
        temp_box_ptr_ptr = g_map_rgb_pts_mesh.m_hashmap_voxels.get_data( box_x, box_y, box_z );
        if ( temp_box_ptr_ptr == nullptr )
        {
            box_ptr = std::make_shared< RGB_Voxel >( box_x, box_y, box_z );
            // m_hashmap_voxels 该变量是用于存储所有的voxel数据的 | 这里是对voxel数据进行了上锁
            g_map_rgb_pts_mesh.m_hashmap_voxels.insert( box_x, box_y, box_z, box_ptr );
            // 这个数据只被储存, 但是没有被后续使用
//            g_map_rgb_pts_mesh.m_voxel_vec.push_back( box_ptr );
        }
        else
        {
            box_ptr = *temp_box_ptr_ptr;
        }
        voxels_recent_visited.insert( box_ptr );
        box_ptr->m_last_visited_time = frame_idx;
        if ( add == 0 )
        {
            rej++;
            continue;
        }

        /// TODO 这里可以考虑对 KDtree 进行保护 ——
        acc++;
        KDtree_pt kdtree_pt( vec_3( frame_pts->points[ pt_idx ].x, frame_pts->points[ pt_idx ].y, frame_pts->points[ pt_idx ].z ), 0 );
        if ( g_map_rgb_pts_mesh.m_kdtree.Root_Node != nullptr )
        {
            g_map_rgb_pts_mesh.m_kdtree.Nearest_Search( kdtree_pt, 1, pt_vec_vec, dist_vec );
            if ( pt_vec_vec.size() )
            {
                if ( sqrt( dist_vec[ 0 ] ) < minimum_pts_size )
                {
                    continue;
                }
            }
        }

        /// @attention 考虑保护m_rgb_pts_vec这个部分
        std::shared_ptr< RGB_pts > pt_rgb = std::make_shared< RGB_pts >();

        double x = frame_pts->points[pt_idx].x;
        double y = frame_pts->points[pt_idx].y;
        double z = frame_pts->points[pt_idx].z;
        pt_rgb->set_pos(vec_3(x, y, z));

        pt_rgb->m_pt_index = g_map_rgb_pts_mesh.m_rgb_pts_vec.size();
        kdtree_pt.m_pt_idx = pt_rgb->m_pt_index;
        g_map_rgb_pts_mesh.m_rgb_pts_vec.push_back( pt_rgb );

        g_map_rgb_pts_mesh.m_hashmap_3d_pts.insert( grid_x, grid_y, grid_z, pt_rgb );
        if ( box_ptr != nullptr )
        {
            box_ptr->m_pts_in_grid.push_back( pt_rgb );
            // box_ptr->add_pt(pt_rgb);
            box_ptr->m_new_added_pts_count++;
            box_ptr->m_meshing_times = 0;
//            file << "new points | " << "use_count " << box_ptr.use_count() << " | m_pts_in_grid.size() " << box_ptr->m_pts_in_grid.size() << endl;
        }
        else
        {
            scope_color( ANSI_COLOR_RED_BOLD );
            for ( int i = 0; i < 100; i++ )
            {
                cout << "box_ptr is nullptr!!!" << endl;
            }
        }
        // Add to kdtree
        g_map_rgb_pts_mesh.m_kdtree.Add_Point( kdtree_pt, false );

    }
    g_mutex_append_map.unlock();

    g_map_rgb_pts_mesh.m_mutex_m_box_recent_hitted->lock();
    g_map_rgb_pts_mesh.m_voxels_recent_visited = voxels_recent_visited;
    g_map_rgb_pts_mesh.m_mutex_m_box_recent_hitted->unlock();

    double time_a = tim_append.toc();


    Eigen::Matrix3d rot_i2w = pose_q.toRotationMatrix();
    Eigen::Vector3d pos_i2w = pose_t;

    /*** m_extR: l2i || m_camera_ext_R: c2l ***/
    Eigen::Matrix3d R_w2c;
    R_w2c = rot_i2w * extR * camera_ext_R;
    Eigen::Vector3d t_w2c;
    t_w2c = rot_i2w * extR * camera_ext_t + extR * extT + pos_i2w;

    std::shared_ptr<Image_frame> image_pose = std::make_shared<Image_frame>(cam_k);
    image_pose->set_pose(eigen_q(R_w2c), t_w2c);
    image_pose->m_img = img;
    image_pose->m_timestamp = ros::Time::now().toSec();
    image_pose->init_cubic_interpolation();
    // 先注释这个部分避免其影响最终渲染的颜色
    image_pose->image_equalize();

    /*** 渲染部分 ***/
    tim_render.tic();
    auto numbers_of_voxels = voxels_recent_visited.size();
    std::vector<shared_ptr<RGB_Voxel>> voxels_for_render;
    for ( Voxel_set_iterator it = voxels_recent_visited.begin(); it != voxels_recent_visited.end(); it++ )
    {
        voxels_for_render.push_back( *it );
    }

    image_pose->m_acc_render_count = 0;
    image_pose->m_acc_photometric_error = 0;
    // 模仿immesh里面mesh重建的部分 —— 直接在这里写成tbb的加速 | 因为这个部分只会调用全局地图中的点云数据,所以上的是与之前相同的锁
    try
    {
        /// @attention 这里使用值传递与引用传递的区别有多少 —— 多线程里面两者是不是有些区别
        cv::parallel_for_(cv::Range(0, numbers_of_voxels), [&](const cv::Range &r) {
            vec_3 pt_w;
            vec_3 rgb_color;
            double u, v;
            double pt_cam_norm;
            g_mutex_append_map.lock();
            for (int voxel_idx = r.start; voxel_idx < r.end; voxel_idx++)
            {
                RGB_voxel_ptr voxel_ptr = voxels_for_render[voxel_idx];
                for (int pt_idx = 0; pt_idx < voxel_ptr->m_pts_in_grid.size(); pt_idx++)
                {
                    pt_w = voxel_ptr->m_pts_in_grid[pt_idx]->get_pos();
                    if (image_pose->project_3d_point_in_this_img(pt_w, u, v, nullptr, 1.0) == false) {
                        continue;
                    }


                    pt_cam_norm = (pt_w - image_pose->m_pose_w2c_t).norm();
                    // 在图像上获取点云的颜色信息 | 然后对这个voxel中的所有点云的颜色信息进行更新
                    rgb_color = image_pose->get_rgb(u, v, 0);

                    if (voxel_ptr->m_pts_in_grid[pt_idx]->update_rgb(
                            rgb_color, pt_cam_norm, vec_3(image_obs_cov, image_obs_cov, image_obs_cov),
                            image_pose->m_timestamp)) {
//                        my_render_pts_count++;
                    }

                }
            }
            g_mutex_append_map.unlock();
            // 开启彩色点云的发布线程来获取color点云
        });
    }
    catch ( ... )
    {
        for ( int i = 0; i < 100; i++ )
        {
            cout << ANSI_COLOR_RED_BOLD << "Exception in tbb parallels...in rendering" << ANSI_COLOR_RESET << endl;
        }
        return;
    }
    double time_b = tim_render.toc();



    /*** mesh重建 ***/
    std::atomic< int >    voxel_idx( 0 );

    std::mutex mtx_triangle_lock, mtx_single_thr;
    typedef std::unordered_set< std::shared_ptr< RGB_Voxel > >::iterator set_voxel_it;
    std::unordered_map< std::shared_ptr< RGB_Voxel >, Triangle_set >     removed_triangle_list;
    std::unordered_map< std::shared_ptr< RGB_Voxel >, Triangle_set >     added_triangle_list;
    g_mutex_reconstruct_mesh.lock();
    // std::cout<< "frame_idx: " << frame_idx <<  "append_points_to_global_map: " <<  tim_append.toc() << std::endl;
    
    tim.tic();
    tim_total.tic();


    time_mesh.tic();
    try
    {
        tbb::parallel_for_each( voxels_recent_visited.begin(), voxels_recent_visited.end(), [ & ]( const std::shared_ptr< RGB_Voxel > &voxel ) {
            // std::unique_lock<std::mutex> thr_lock(mtx_single_thr);
            // printf_line;
            if ( ( voxel->m_meshing_times >= 1 ) || ( voxel->m_new_added_pts_count < 0 ) )
            {
                return;
            }
            Common_tools::Timer tim_lock;
            tim_lock.tic();
            voxel->m_meshing_times++;
            voxel->m_new_added_pts_count = 0;
//            vec_3 pos_1 = vec_3( voxel->m_pos[ 0 ], voxel->m_pos[ 1 ], voxel->m_pos[ 2 ] );

            // printf("Voxels [%d], (%d, %d, %d) ", count, pos_1(0), pos_1(1), pos_1(2) );
            std::unordered_set< std::shared_ptr< RGB_Voxel > > neighbor_voxels;
            neighbor_voxels.insert( voxel );
            g_mutex_append_map.lock();
            std::vector< RGB_pt_ptr > pts_in_voxels = retrieve_pts_in_voxels( neighbor_voxels );
            if ( pts_in_voxels.size() < 3 )
            {
                g_mutex_append_map.unlock();
                return;
            }
            g_mutex_append_map.unlock();
            // Voxel-wise mesh pull
            pts_in_voxels = retrieve_neighbor_pts_kdtree( pts_in_voxels );
            pts_in_voxels = remove_outlier_pts( pts_in_voxels, voxel );

            std::set< long > relative_point_indices;
            for ( RGB_pt_ptr tri_ptr : pts_in_voxels )
            {
                relative_point_indices.insert( tri_ptr->m_pt_index );
            }

            int iter_count = 0;
            g_triangles_manager.m_enable_map_edge_triangle = 0;

           pts_in_voxels.clear();
           for ( auto p : relative_point_indices )
           {
               pts_in_voxels.push_back( g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ] );
           }
           
            std::set< long > convex_hull_index, inner_pts_index;
            // mtx_triangle_lock.lock();
            voxel->m_short_axis.setZero();
            std::vector< long > add_triangle_idx = delaunay_triangulation( pts_in_voxels, voxel->m_long_axis, voxel->m_mid_axis,
                                                                               voxel->m_short_axis, convex_hull_index, inner_pts_index );

            /// @attention 后续部分取消可以避免程序出现段错误导致崩溃(其原因很有可能是出现了shared_ptr之间互相指向导致的错误)
//           for ( auto p : inner_pts_index )
//           {
//               if ( voxel->if_pts_belong_to_this_voxel( g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ] ) )
//               {
//                   g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_is_inner_pt = true;
//                   g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_parent_voxel = voxel;
//               }
//           }
//
//           for ( auto p : convex_hull_index )
//           {
//               g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_is_inner_pt = false;
//               g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_parent_voxel = voxel;
//           }
            Triangle_set triangles_sets = g_triangles_manager.find_relative_triangulation_combination( relative_point_indices );
            Triangle_set triangles_to_remove, triangles_to_add, existing_triangle;

            // 为了保证颜色信息可以被后续更新 | 这里将之前生成的三角形全部更新 - 实际思路肯定不是这样的(得确定哪些三角形的颜色信息是需要更新的)
            triangle_compare( triangles_sets, add_triangle_idx, triangles_to_remove, triangles_to_add, &existing_triangle );
            
            // Refine normal index
            for ( auto triangle_ptr : triangles_to_add )
            {
                correct_triangle_index( triangle_ptr, g_eigen_vec_vec[ frame_idx ].second.block( 4, 0, 3, 1 ), voxel->m_short_axis );
            }
            for ( auto triangle_ptr : existing_triangle )
            {
                correct_triangle_index( triangle_ptr, g_eigen_vec_vec[ frame_idx ].second.block( 4, 0, 3, 1 ), voxel->m_short_axis );
            }

            std::unique_lock< std::mutex > lock( mtx_triangle_lock );
            
            removed_triangle_list.emplace( std::make_pair( voxel, triangles_to_remove ) );
            added_triangle_list.emplace( std::make_pair( voxel, triangles_to_add ) );
            
            voxel_idx++;
        } );
    }
    catch ( ... )
    {
        for ( int i = 0; i < 100; i++ )
        {
            cout << ANSI_COLOR_RED_BOLD << "Exception in tbb parallels..." << ANSI_COLOR_RESET << endl;
        }
        return;
    }

    // note 开启发布线程
    g_map_rgb_pts_mesh.m_last_updated_frame_idx++;
    if( g_pub_thr == nullptr  && flag == 0 )
    {
        LOG(INFO) << "Prepare for RGB cloud";
        // 对于10000个RGB点构建一个彩色点云的发布器，然后创建出多个发布器之后再发布数据(不过不理解的部分在与这里明明是进行数据的读取，为什么这里不需要上锁-也访问全局地图了)
        g_pub_thr = std::make_unique<std::thread>(&Global_map::service_pub_rgb_maps, &g_map_rgb_pts_mesh);
        flag = 1;           // 因为创建线程写在了while()循环中, 为了避免线程的重复创建，这里设置flag
    }


    double              mul_thr_cost_time = tim.toc( " ", 0 );
    Common_tools::Timer tim_triangle_cost;
    int                 total_delete_triangle = 0, total_add_triangle = 0;
    // Voxel-wise mesh push
    for ( auto &triangles_set : removed_triangle_list )
    {
        total_delete_triangle += triangles_set.second.size();
        g_triangles_manager.remove_triangle_list( triangles_set.second );
    }

    for ( auto &triangle_list : added_triangle_list )
    {
        Triangle_set triangle_idx = triangle_list.second;
        total_add_triangle += triangle_idx.size();
        for ( auto triangle_ptr : triangle_idx )
        {
            Triangle_ptr tri_ptr = g_triangles_manager.insert_triangle( triangle_ptr->m_tri_pts_id[ 0 ], triangle_ptr->m_tri_pts_id[ 1 ],
                                                                        triangle_ptr->m_tri_pts_id[ 2 ], 1 );
            tri_ptr->m_index_flip = triangle_ptr->m_index_flip;
        }
    }
    
    g_mutex_reconstruct_mesh.unlock();
    tim_total.toc();

    double time_c = time_mesh.toc();
    double time_total = time_a + time_b + time_c;
    LOG(INFO) << "frame_idx: " << frame_idx << " | " << time_a << " | " << time_b   << " | " << time_c << "| "<< time_total;
    // std::cout<< "frame_idx: " << frame_idx << "mesh reconstruction cost: " << tim_total.toc() << std::endl;

    if ( g_fp_cost_time )
    {
        if ( frame_idx > 0 )
            g_LiDAR_frame_avg_time = g_LiDAR_frame_avg_time * ( frame_idx - 1 ) / frame_idx + ( g_vx_map_frame_cost_time ) / frame_idx;
        fprintf( g_fp_cost_time, "%d %lf %d %lf %lf\r\n", frame_idx, tim.toc( " ", 0 ), ( int ) voxel_idx.load(), g_vx_map_frame_cost_time,
                 g_LiDAR_frame_avg_time );
        fflush( g_fp_cost_time );
    }
    if ( g_current_frame < frame_idx )
    {
        g_current_frame = frame_idx;
    }
    else
    {
        if ( g_eigen_vec_vec[ g_current_frame + 1 ].second.size() > 7 )
        {
            g_current_frame++;
        }
    }
}




void service_reconstruct_mesh()
{
    if ( g_thread_pool_rec_mesh == nullptr )
    {
        g_thread_pool_rec_mesh = std::make_shared< Common_tools::ThreadPool >( g_maximum_thread_for_rec_mesh );
        LOG(INFO) << "g_maximum_thread_for_rec_mesh: " << g_maximum_thread_for_rec_mesh;
    }
    int drop_frame_num = 0;
    while ( 1 )
    {
        
        while ( g_rec_color_data_package_list.size() == 0 )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
        }

        g_mutex_all_data_package_lock.lock();
        auto data_pack_front = g_rec_color_data_package_list.front();
        g_rec_color_data_package_list.pop_front();
        if(data_pack_front.m_frame_pts == nullptr || data_pack_front.m_img.empty() || data_pack_front.m_frame_pts->points.size() == 0)
        {
            g_mutex_all_data_package_lock.unlock();
            continue;
        }

        g_mutex_all_data_package_lock.unlock();

        if ( g_enable_mesh_rec )
        {
            g_thread_pool_rec_mesh->commit_task( incremental_mesh_reconstruction, data_pack_front.m_frame_pts, data_pack_front.m_img ,data_pack_front.m_pose_q,
                                                 data_pack_front.m_pose_t, data_pack_front.m_frame_idx );
        }

        std::this_thread::sleep_for( std::chrono::microseconds( 5 ) );
    }
}
extern bool  g_flag_pause;
int          g_frame_idx = 0;
std::thread *g_rec_mesh_thr = nullptr;

void start_mesh_threads( int maximum_threads = 20 )
{
    if ( g_eigen_vec_vec.size() <= 0 )
    {
        g_eigen_vec_vec.resize( 1e6 );
    }
    if ( g_rec_mesh_thr == nullptr )
    {
        g_maximum_thread_for_rec_mesh = maximum_threads;
        g_rec_mesh_thr = new std::thread( service_reconstruct_mesh );
    }
}

void reconstruct_mesh_from_pointcloud( pcl::PointCloud< pcl::PointXYZI >::Ptr frame_pts, double minimum_pts_distance )
{
    start_mesh_threads();
    cout << "=== reconstruct_mesh_from_pointcloud ===" << endl;
    cout << "Input pointcloud have " << frame_pts->points.size() << " points." << endl;
    pcl::PointCloud< pcl::PointXYZI >::Ptr all_cloud_ds( new pcl::PointCloud< pcl::PointXYZI > );

    pcl::VoxelGrid< pcl::PointXYZI > sor;
    sor.setInputCloud( frame_pts );
    sor.setLeafSize( minimum_pts_distance, minimum_pts_distance, minimum_pts_distance );
    sor.filter( *all_cloud_ds );

    cout << ANSI_COLOR_BLUE_BOLD << "Raw points number = " << frame_pts->points.size()
         << ", downsampled points number = " << all_cloud_ds->points.size() << ANSI_COLOR_RESET << endl;
    g_mutex_data_package_lock.lock();
    g_rec_mesh_data_package_list.emplace_back( all_cloud_ds, Eigen::Quaterniond::Identity(), vec_3::Zero(), 0 );
    g_mutex_data_package_lock.unlock();
}

void open_log_file()
{
    if ( g_fp_cost_time == nullptr || g_fp_lio_state == nullptr )
    {
        Common_tools::create_dir( std::string( Common_tools::get_home_folder() ).append( "/ImMesh_output" ).c_str() );
        std::string cost_time_log_name = std::string( Common_tools::get_home_folder() ).append( "/ImMesh_output/mesh_cost_time.log" );
        std::string lio_state_log_name = std::string( Common_tools::get_home_folder() ).append( "/ImMesh_output/lio_state.log" );
        // cout << ANSI_COLOR_BLUE_BOLD ;
        // cout << "Record cost time to log file:" << cost_time_log_name << endl;
        // cout << "Record LIO state to log file:" << cost_time_log_name << endl;
        // cout << ANSI_COLOR_RESET;
        g_fp_cost_time = fopen( cost_time_log_name.c_str(), "w+" );
        g_fp_lio_state = fopen( lio_state_log_name.c_str(), "w+" );
    }
}

std::vector< vec_4 > convert_pcl_pointcloud_to_vec( pcl::PointCloud< pcl::PointXYZI > &pointcloud )
{
    int                  pt_size = pointcloud.points.size();
    std::vector< vec_4 > eigen_pt_vec( pt_size );
    for ( int i = 0; i < pt_size; i++ )
    {
        eigen_pt_vec[ i ]( 0 ) = pointcloud.points[ i ].x;
        eigen_pt_vec[ i ]( 1 ) = pointcloud.points[ i ].y;
        eigen_pt_vec[ i ]( 2 ) = pointcloud.points[ i ].z;
        eigen_pt_vec[ i ]( 3 ) = pointcloud.points[ i ].intensity;
    }
    return eigen_pt_vec;
}

void Voxel_mapping::map_incremental_grow()
{
    start_mesh_threads( m_meshing_maximum_thread_for_rec_mesh );
    if ( m_use_new_map )
    {
        while ( g_flag_pause )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
        }
        // startTime = clock();
        pcl::PointCloud< pcl::PointXYZI >::Ptr world_lidar( new pcl::PointCloud< pcl::PointXYZI > );
        pcl::PointCloud< pcl::PointXYZI >::Ptr world_lidar_full( new pcl::PointCloud< pcl::PointXYZI > );

        std::vector< Point_with_var > pv_list;
        // TODO: saving pointcloud to file
        // pcl::io::savePCDFileBinary(Common_tools::get_home_folder().append("/r3live_output/").append("last_frame.pcd").c_str(), *m_feats_down_body);
        transformLidar( state.rot_end, state.pos_end, m_feats_down_body, world_lidar );
        for ( size_t i = 0; i < world_lidar->size(); i++ )
        {
            Point_with_var pv;
            pv.m_point << world_lidar->points[ i ].x, world_lidar->points[ i ].y, world_lidar->points[ i ].z;
            M3D point_crossmat = m_cross_mat_list[ i ];
            M3D var = m_body_cov_list[ i ];
            var = ( state.rot_end * m_extR ) * var * ( state.rot_end * m_extR ).transpose() +
                  ( -point_crossmat ) * state.cov.block< 3, 3 >( 0, 0 ) * ( -point_crossmat ).transpose() + state.cov.block< 3, 3 >( 3, 3 );
            pv.m_var = var;
            pv_list.push_back( pv );
        }

        // pcl::PointCloud< pcl::PointXYZI >::Ptr world_lidar( new pcl::PointCloud< pcl::PointXYZI > );
        std::sort( pv_list.begin(), pv_list.end(), var_contrast );
        updateVoxelMap( pv_list, m_max_voxel_size, m_max_layer, m_layer_init_size, m_max_points_size, m_min_eigen_value, m_feat_map );
        double vx_map_cost_time = omp_get_wtime();
        g_vx_map_frame_cost_time = ( vx_map_cost_time - g_LiDAR_frame_start_time ) * 1000.0;
        // cout << "vx_map_cost_time = " <<  g_vx_map_frame_cost_time << " ms" << endl;

//        transformLidar( state.rot_end, state.pos_end, m_feats_undistort, world_lidar_full );
//        g_mutex_data_package_lock.lock();
//        g_rec_mesh_data_package_list.emplace_back( world_lidar_full, Eigen::Quaterniond( state.rot_end ), state.pos_end, g_frame_idx );
//        g_mutex_data_package_lock.unlock();
        open_log_file();
        if ( g_fp_lio_state != nullptr )
        {
            dump_lio_state_to_log( g_fp_lio_state );
        }
        g_frame_idx++;
    }

    if ( !m_use_new_map )
    {
        for ( int i = 0; i < m_feats_down_size; i++ )
        {
            /* transform to world frame */
            pointBodyToWorld( m_feats_down_body->points[ i ], m_feats_down_world->points[ i ] );
        }
        
        // add_to_offline_bin( state, m_Lidar_Measures.lidar_beg_time, m_feats_down_world );
        
#ifdef USE_ikdtree
#ifdef USE_ikdforest
        ikdforest.Add_Points( feats_down_world->points, lidar_end_time );
#else
        m_ikdtree.Add_Points( m_feats_down_world->points, true );
#endif
#endif
    }
}
