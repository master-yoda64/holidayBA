#include <bits/stdc++.h>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <camera/camera_model_pinhole_bal.hpp>
#include <ba/bundle_adjuster_base.hpp>

class BalBundleAdjuster: public BundleAdjusterBase
{
public:
    BalBundleAdjuster() = default;
    ~BalBundleAdjuster() = default;
    std::vector<OptResult> optimize() override;
    std::vector<OptResult> optimize_thread();
    OptResult optimize_camera(
        const std::vector<Observation>& obs,
        const CameraModelPinholeBal cam,
        const std::vector<Point3D> points
    );
    OptResult optimize_camera_thread(
        const std::vector<Observation>& obs,
        const CameraModelPinholeBal cam,
        const std::vector<Point3D> points
    );
    void load_data(std::string path);
    std::vector<CameraModelPinholeBal> get_cameras()
    {
        return static_cast<std::vector<CameraModelPinholeBal>>(cameras_);
    }
    
private:
    std::vector<CameraModelPinholeBal> cameras_;
    int max_iter_ = 30;
    double convergence_threshold_ = 1e-12; //1e-15だとcamera4が収束しない、数値誤差の範囲は1e-14~15くらい？

    Eigen::Matrix<double, 3, 3> skew_symmetric(
        const Eigen::Vector3d& v
    );
    Eigen::Matrix<double, 4, 6> get_dproj_dxi(
        const Eigen::Vector3d& point
    );
    std::tuple<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> compute_H_b
    (
        Eigen::Vector3d point3d,
        Eigen::Vector2d point2d,  
        Eigen::Vector2d projected_2d,
        Eigen::Matrix3d R_ini,
        Eigen::Vector3d t_ini,
        Eigen::Matrix<double, 2, 4> jacobian
    );
    void process_chunk(
        int start, int end,
        const std::vector<Observation>& obs,
        const std::vector<Point3D> points,
        const CameraModelPinholeBal& cam,
        const Eigen::Matrix3d& R_ini,
        const Eigen::Vector3d& t_ini,
        Eigen::MatrixXd& H_local,
        Eigen::VectorXd& b_local)
    {
        for (int j = start; j < end; ++j) {
            const Observation& ob = obs[j];

            Eigen::Vector3d X = points[ob.point_idx];
            Eigen::Vector2d z(ob.x, ob.y);

            double f = cam.get_fx();
            double k1 = cam.get_k1();
            double k2 = cam.get_k2();

            Eigen::Vector2d z_hat = cam.project(X, R_ini, t_ini);
            Eigen::Matrix<double, 2, 4> Jprj = cam.get_prj_jacobian(X);

            auto [Hi, bi] = compute_H_b(X, z, z_hat, R_ini, t_ini, Jprj);

            H_local += Hi;
            b_local += bi;
        }
    }
};

