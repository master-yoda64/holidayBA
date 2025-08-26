#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <camera/camera_model_pinhole_bal.hpp>
#include <ba/bundle_adjuster_base.hpp>


struct SnavelyReprojectionError {
    SnavelyReprojectionError(
        double observed_x, 
        double observed_y,
        double X,
        double Y,
        double Z,
        double f

    ) : observed_x_(observed_x), 
        observed_y_(observed_y),
        X_(X),
        Y_(Y),
        Z_(Z),
        f_(f) {};
    ~SnavelyReprojectionError() {};

    // operator & functions
    template <typename T>
    bool operator()(const T* const camera,
                    T* residuals) const 
    {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        T point[3];
        point[0] = T(X_);
        point[1] = T(Y_);
        point[2] = T(Z_);
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // homogenios coordinates.
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Compute final projected point position.
        T predicted_x = f_ * xp; 
        T predicted_y = f_ * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x_;
        residuals[1] = predicted_y - observed_y_;
        return true;
    }

    static ceres::CostFunction* Create(
        double observed_x,
        double observed_y,
        double X,
        double Y,
        double Z,
        double f
    ) 
    {
        return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 1, 6>(
            new SnavelyReprojectionError(
                observed_x, 
                observed_y,
                X,
                Y,
                Z,
                f)
        );
    }
    //variables
    double observed_x_;
    double observed_y_;
    double f_;
    double X_;
    double Y_;
    double Z_;

};

class CeresBalBundleAdjuster : public BundleAdjusterBase
{
public:
    CeresBalBundleAdjuster(){};
    ~CeresBalBundleAdjuster(){};

    void add_residuals_camera_id(int camera_id)
    {
        CameraModelPinholeBal& camera = cameras_[camera_id];
        std::vector<Observation> observations;
        double angle_axis[3] = {
            camera.get_rotation()(0),
            camera.get_rotation()(1),
            camera.get_rotation()(2)
        };
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
            rt_vec[0] = angle_axis[0];
            rt_vec[1] = angle_axis[1];
            rt_vec[2] = angle_axis[2];
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
    std::vector<OptResult> optimize()
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
            Eigen::Vector3d angle_axis(camera[0], camera[1], camera[2]);
            Eigen::Vector3d t(camera[3], camera[4], camera[5]);
            Eigen::Matrix3d R;
            ceres::AngleAxisToRotationMatrix(camera, R.data());
            result.R = R;
            result.t = t;
            results.push_back(result);
        }
        return results;
    };
    void load_data(std::string path) 
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
private:
    ceres::Problem problem_;
    std::vector<CameraModelPinholeBal> cameras_;
    std::vector<std::vector<double>> camera_params_; 
};