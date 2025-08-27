#include <bits/stdc++.h>
#include <math.h>
#include <iostream>

#include <Eigen/Core>
#include <camera/camera_model_pinhole_bal.hpp>
#include <ba/ceres_bal_bundle_adjuster.hpp>

void print_summary(
    const OptResult& result, 
    const std::vector<Observation>& obsavations,
    const std::vector<Point3D>& points,
    const std::vector<CameraModelPinholeBal>& cameras,
    int camera_id
) 
{
    CameraModelPinholeBal camera0 = cameras[camera_id];
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Camera ID: " << camera_id << std::endl;
    std::cout << "Residual: " << result.residual << std::endl;
    std::cout << "Optimized Rotation Matrix:\n" << result.R << std::endl;
    std::cout << "Optimized Translation Vector: " << result.t.transpose() << std::endl;
    std::cout << "Initial Rotation Matrix:\n" << camera0.rotation_matrix() << std::endl;
    std::cout << "Initial Translation Vector: " << camera0.get_translation().transpose() << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "R * R^T ? : \n" << result.R * result.R.transpose()  << std::endl;
    std::cout << " det(R) = 1 ?: det(R) = " << result.R.determinant() << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    int print_cnt = 0;
    for (Observation obs: obsavations) {
        if (print_cnt++ >= 5) break; // Limit to first 5 observations
        Eigen::Vector2d projected_point = camera0.project(points[obs.point_idx], 
                                                  result.R, 
                                                  result.t);
        std::cout << "Observation: Camera " << obs.camera_idx 
                  << ", Point " << obs.point_idx 
                  << ", 2D Point: (" << obs.x << ", " << obs.y << ")"
                  << ", Projected: (" << projected_point(0) << ", " << projected_point(1) << ")" << "\n";
    }
    std::cout << std::string(70, '=') << std::endl;
}
int main(int argc, char** argv) {

    int camera_id = argv[1] ? std::stoi(argv[1]) : 0;
    CeresBalBundleAdjuster ceres_optimizer;
    ceres_optimizer.load_data("data/problem-16-22106-pre.txt");

    std::vector<CameraModelPinholeBal> cameras = ceres_optimizer.get_cameras();
    std::vector<Observation> observations = ceres_optimizer.get_observations();
    std::vector<Point3D> points = ceres_optimizer.get_points();
    CameraModelPinholeBal camera0 = cameras[camera_id];

    std::vector<Observation> obs0;
    for (const auto &obs : observations) {
        if (obs.camera_idx == camera0.get_camera_id()) {
            obs0.push_back(obs);
        }
    }
    auto ceres_opt_start_time = std::chrono::high_resolution_clock::now();
    OptResult ceres_result  = ceres_optimizer.optimize_camera(obs0, camera0);
    //ceres_optimizer.optimize();
    auto ceres_opt_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> ceres_opt_duration = ceres_opt_end_time - ceres_opt_start_time;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Ceres Optimization time: " << ceres_opt_duration.count() << " seconds" << std::endl;

    print_summary(ceres_result, obs0, points, cameras, camera_id);
    return 0;
}

