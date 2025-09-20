#include <bits/stdc++.h>
#include <iostream>
#include <expected>
#include <system_error>


#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <camera/camera_model_pinhole_bal.hpp>
#include <ba/bundle_adjuster_base.hpp>

class BalBundleAdjuster: public BundleAdjusterBase<CameraModelPinholeBal>
{
public:
    BalBundleAdjuster() = default;
    ~BalBundleAdjuster() = default;
    std::vector<OptResult> optimize() override;
    OptResult optimize_camera(
        const std::vector<Observation>& obs,
        const CameraModelPinholeBal& cam
    ) override;
    std::expected<bool, std::error_code> load_data(std::string path);
    
private:
    int max_iter_ = 30;
    double convergence_threshold_ = 1e-12; //1e-15だとcamera4が収束しない、数値誤差の範囲は1e-14~15くらい？

    std::expected<Eigen::Matrix<double, 3, 3>, std::error_code> skew_symmetric(
        const Eigen::Vector3d& v
    );
    std::expected<Eigen::Matrix<double, 4, 6>, std::error_code> get_dproj_dxi(
        const Eigen::Matrix<double, 3, 3>& hatp
    );
    std::expected<std::tuple<Eigen::Matrix<double, 6, 6>, 
        Eigen::Matrix<double, 6, 1>>, std::error_code> compute_H_b
    (
        Eigen::Vector2d point2d,  
        Eigen::Vector2d projected_2d,
        Eigen::Matrix3d R_ini,
        Eigen::Vector3d t_ini,
        Eigen::Matrix<double, 2, 4> jacobian,
        Eigen::Matrix<double, 4, 6> dproj_dxi
    );
    void compute_H_process_chunk(
        int start, int end,
        const std::vector<Observation>& obs,
        const std::vector<Point3D> points,
        const CameraModelPinholeBal& cam,
        const Eigen::Matrix3d& R_ini,
        const Eigen::Vector3d& t_ini,
        Eigen::MatrixXd& H_local,
        Eigen::VectorXd& b_local
    );
};

