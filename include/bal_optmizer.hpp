#include <bits/stdc++.h>
#include <math.h>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <sophus/so3.hpp>
#include <Eigen/Geometry>
#include <camera/camera_model_pinhole_bal.hpp>

using Point3D = Eigen::Vector3d;
struct Observation {
  int camera_idx;
  int point_idx;
  double x, y;
};

struct OptResult {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    double residual;

};

class BaOptimizer 
{
public:
    BaOptimizer() = default;
    ~BaOptimizer() = default;

    OptResult optimize(
        const std::vector<Observation> obs,
        const CameraModelPinholeBal cam,
        const std::vector<Point3D> points,
        int max_iter=10,
        double convergence_threshold=1e-15
    ) {
        double residual = 0.0;
        int cam_idx = cam.get_camera_id();
        Eigen::Matrix3d R_ini = cam.rotationMatrix();
        Eigen::Vector3d t_ini = cam.get_translation();

        for (int it = 0; it < max_iter; ++it) {
            Eigen::Matrix<double,6,6> H = Eigen::Matrix<double,6,6>::Zero();
            Eigen::Matrix<double,6,1> b = Eigen::Matrix<double,6,1>::Zero();

            auto start_time = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < obs.size(); j++) {
                const Observation& ob = obs[j];
                //if (cam_idx != ob.camera_idx) continue;

                Eigen::Vector3d X = points[ob.point_idx];
                Eigen::Vector2d z(ob.x, ob.y);
                double f = cam.get_fx();
                double k1 = cam.get_k1();
                double k2 = cam.get_k2();
                Eigen::Vector2d z_hat = cam.project(X, R_ini, t_ini);
                Eigen::Matrix<double,2,4> Jprj = cam.get_prj_jacobian(X);
                auto [Hi, bi] = compute_H_b(X, z, z_hat, R_ini, t_ini, Jprj);
                H += Hi;
                b += bi;
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            std::cout << "Iteration " << it << ": Time taken for computation: " << duration.count() << " seconds" << std::endl;

            // （任意）ダンピングを入れると安定
            // double lambda = 1e-6;
            // H_sum.diagonal().array() += lambda;

            Eigen::SparseMatrix<double> H_sparse = H.sparseView();
            Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
            cg.setTolerance(1e-4);
            cg.compute(H_sparse);
            Eigen::VectorXd delta = cg.solve(b);
                
            Eigen::Matrix<double, 3, 1> delta_t = delta.tail<3>();
            Sophus::SO3<double> delta_R = Sophus::SO3<double>::exp(delta.head<3>());
            R_ini = delta_R.matrix() * R_ini;
            t_ini += delta_t;
            residual = (H * delta - b).squaredNorm();
            if (residual < convergence_threshold) {
                std::cout << "Converged at iteration " << it << " with residual: " << residual << std::endl;
                break;
            }
        }
        return OptResult{R_ini, t_ini, residual};
    }

    void load_data(std::string path) {
        std::ifstream ifs(path);
        if (!ifs) {
            std::cerr << "Failed to open: " << path << "\n";
            return;
        }
        int num_cameras, num_points, num_observations;
        ifs >> num_cameras >> num_points >> num_observations;
        if (!ifs) {
            std::cerr << "Failed reading header.\n";
            return;
        }
        observations_.reserve(num_observations);
        cameras_.reserve(num_cameras);
        points_.reserve(num_points);
        for (int i = 0; i < num_observations; ++i) {
            Observation obs;
            ifs >> obs.camera_idx >> obs.point_idx >> obs.x >> obs.y;
            if (!ifs) {
            std::cerr << "Failed reading observation #" << i << "\n";
            return;
            }
            observations_.push_back(obs);
        }
        for (int i = 0; i < num_cameras; ++i) {
            double r0, r1, r2;
            ifs >> r0 >> r1 >> r2;
            Eigen::Vector3d rot = Eigen::Vector3d(r0, r1, r2);
            double t0, t1, t2;
            ifs >> t0 >> t1 >> t2;
            Eigen::Vector3d t = Eigen::Vector3d(t0, t1, t2);
            double f, k1, k2;
            ifs >> f >> k1 >> k2;
            if (!ifs) {
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
        for (int i = 0; i < num_points; ++i) {
            double X, Y, Z;
            ifs >> X >> Y >> Z;
            if (!ifs) {
            std::cerr << "Failed reading point #" << i << "\n";
            return;
            }
            points_[i] = Point3D(X, Y, Z);
        }    
        ifs.close();
        return;
    }

    std::vector<CameraModelPinholeBal> get_cameras() const {
        return cameras_;
    }
    std::vector<Observation> get_observations() const {
        return observations_;
    }
    std::vector<Point3D> get_points() const {
        return points_;
    }
    std::vector<CameraModelPinholeBal> cameras_;
    std::vector<Observation> observations_;
    std::vector<Point3D> points_;
private:
    Eigen::Matrix<double, 3, 3> skew_symmetric(
        const Eigen::Vector3d& v
    )
    {
        Eigen::Matrix<double, 3, 3> skew;
        skew << 0, -v(2), v(1),
                v(2), 0, -v(0),
                -v(1), v(0), 0;
        return skew;
    }
    Eigen::Matrix<double, 4, 6> get_dproj_dxi(
        const Eigen::Vector3d& point
    )
    {
        Eigen::Matrix<double, 4, 6> dproj_dxi = Eigen::Matrix<double, 4, 6>::Zero();
        // a = [ω]x * p = -[p]x * ω
        // thus derivation of ω is [-p]x
        Eigen::Matrix<double, 3, 3> hatp = skew_symmetric(point);
        dproj_dxi.block<3, 3>(0, 0) = -hatp;
        dproj_dxi.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        return dproj_dxi;
    }

    std::tuple<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> compute_H_b
    (
        Eigen::Vector3d point3d,
        Eigen::Vector2d point2d,  
        Eigen::Vector2d projected_2d,
        Eigen::Matrix3d R_ini,
        Eigen::Vector3d t_ini,
        Eigen::Matrix<double, 2, 4> jacobian
    )
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();    
        Eigen::Matrix<double, 4, 6> dproj_dxi = get_dproj_dxi(point3d);
        Eigen::Matrix<double, 2, 6> jacobian_full = jacobian * dproj_dxi;

        Eigen::Vector2d e = point2d - projected_2d;
        // H = J^T * J
        H += jacobian_full.transpose() * jacobian_full;
        // b = J^T * r
        b += jacobian_full.transpose() * e;
        return std::make_tuple(H, b);
    }
};

