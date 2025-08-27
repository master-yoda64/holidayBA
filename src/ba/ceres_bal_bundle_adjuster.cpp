
#include <ba/ceres_bal_bundle_adjuster.hpp>

void CeresBalBundleAdjuster::add_residuals_camera_id(int camera_id)
{
    CameraModelPinholeBal& camera = cameras_[camera_id];
    std::vector<Observation> observations;
    for (const auto &obs : observations_) 
    {
        if (obs.camera_idx == camera.get_camera_id()) 
        {
            observations.push_back(obs);
        }
    }
    for (int i=0; i < observations.size(); i++)
    {
        const Observation& obs = observations[i];
        Eigen::Vector3d point3d = points_[obs.point_idx];
        Eigen::Vector2d point2d(obs.x, obs.y);
        ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
            point2d(0), 
            point2d(1), 
            point3d(0),
            point3d(1),
            point3d(2),
            camera.get_fx()
        );
        std::vector<double> rt_vec;
        rt_vec.resize(6);
        rt_vec[0] = camera.get_rotation()(0);
        rt_vec[1] = camera.get_rotation()(1);
        rt_vec[2] = camera.get_rotation()(2);
        rt_vec[3] = camera.get_translation()(0);
        rt_vec[4] = camera.get_translation()(1);
        rt_vec[5] = camera.get_translation()(2);
        camera_params_.push_back(rt_vec);
        problem_.AddResidualBlock(
            cost_function,
            nullptr,
            camera_params_[i].data()
        );
    }
};
std::vector<OptResult> CeresBalBundleAdjuster::optimize()
{
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem_, &summary);
    std::cout << summary.FullReport() << std::endl;
    std::vector<OptResult> results;
    for (int i = 0; i < camera_params_.size(); i++)
    {
        OptResult result;
        double* camera = camera_params_[i].data();
        Eigen::Vector3d t(camera[3], camera[4], camera[5]);
        Eigen::Matrix3d R;
        ceres::AngleAxisToRotationMatrix(camera, R.data());
        result.R = R;
        result.t = t;
        results.push_back(result);
        result.residual = summary.final_cost;
    }
    return results;
};

OptResult CeresBalBundleAdjuster::optimize_camera(
    const std::vector<Observation>& obs,
    const CameraModelPinholeBal& cam
) 
{
    int cam_id = cam.get_camera_id(); 
    add_residuals_camera_id(cam_id);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem_, &summary);
    std::cout << summary.FullReport() << std::endl;
    OptResult result;
    double* camera = camera_params_[0].data();
    Eigen::Vector3d t(camera[3], camera[4], camera[5]);
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(camera, R.data());
    result.R = R;
    result.t = t;
    result.residual = summary.final_cost;
    return result;
};

void CeresBalBundleAdjuster::load_data(std::string path) 
{
    std::ifstream ifs(path);
    if (!ifs) 
    {
        std::cerr << "Failed to open: " << path << "\n";
        return;
    }
    int num_cameras, num_points, num_observations;
    ifs >> num_cameras >> num_points >> num_observations;
    if (!ifs) 
    {
        std::cerr << "Failed reading header.\n";
        return;
    }
    std::cout << "Cameras: " << num_cameras 
            << ", Points: " << num_points 
            << ", Observations: " << num_observations << std::endl;
    observations_.reserve(num_observations);
    cameras_.reserve(num_cameras);
    points_.reserve(num_points);
    for (int i = 0; i < num_observations; ++i) 
    {
        Observation obs;
        ifs >> obs.camera_idx >> obs.point_idx >> obs.x >> obs.y;
        if (!ifs) 
        {
        std::cerr << "Failed reading observation #" << i << "\n";
        return;
        }
        observations_.push_back(obs);
    }
    for (int i = 0; i < num_cameras; ++i) 
    {
        double r0, r1, r2;
        ifs >> r0 >> r1 >> r2;
        Eigen::Vector3d rot = Eigen::Vector3d(r0, r1, r2);
        double t0, t1, t2;
        ifs >> t0 >> t1 >> t2;
        Eigen::Vector3d t = Eigen::Vector3d(t0, t1, t2);
        double f, k1, k2;
        ifs >> f >> k1 >> k2;
        if (!ifs) 
        {
        std::cerr << "Failed reading camera #" << i << "\n";
        return;
        }
        cameras_.emplace_back(
            f, 
            k1, 
            k2,
            rot,
            t,
            i
        );
    }
    for (int i = 0; i < num_points; ++i) 
    {
        double X, Y, Z;
        ifs >> X >> Y >> Z;
        if (!ifs) {
        std::cerr << "Failed reading point #" << i << "\n";
        return;
        }
        points_.emplace_back(X, Y, Z);
    }
    ifs.close();
    return;
}