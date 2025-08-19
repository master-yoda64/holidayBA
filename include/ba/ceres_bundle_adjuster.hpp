#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

struct SnavelyReprojectionError {
    SnavelyReprojectionError(
        double observed_x, 
        double observed_y,
        double focal_length,
        double principal_point_x,
        double principal_point_y
    ) : observed_x_(observed_x), 
        observed_y_(observed_y),
        focal_length_(focal_length),
        principal_point_x_(principal_point_x),
        principal_point_y_(principal_point_y) {};

    ~SnavelyReprojectionError() {};

    // operator & functions
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const 
    {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // homogenios coordinates.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Compute final projected point position.
        T predicted_x = focal_length_ * xp - principal_point_x_;
        T predicted_y = focal_length_ * yp - principal_point_y_;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x_;
        residuals[1] = predicted_y - observed_y_;

        return true;
    }

    static ceres::CostFunction* Create(
        double observed_x,
        double observed_y,
        double focal_length,
        double principal_point_x,
        double principal_point_y
    ) 
    {
        return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
            new SnavelyReprojectionError(
                observed_x, 
                observed_y,
                focal_length,
                principal_point_x,
                principal_point_y
            )
        );
    }
    //variables
    double observed_x_;
    double observed_y_;
    double focal_length_;  
    double principal_point_x_;
    double principal_point_y_;
};

class BundleAdjuster
{
    public:
        BundleAdjuster(){};
        ~BundleAdjuster(){};
        void set_camera_matrix
        (
            double focal_length, 
            double principal_point_x, 
            double principal_point_y
        )
        {
            f_ = focal_length;
            cx_ = principal_point_x;
            cy_ = principal_point_y;
        }
        void add_residuals(
            const std::vector<cv::Point2f>  &pnp_input_2d, 
            const std::vector<cv::Point3f>  &pnp_input_3d,
            const cv::Mat &rotation_vector,
            const cv::Mat &translation_vector
        )
        {
            camera_params_.clear();
            point_params_.clear();

            double angle_axis[3] = { rotation_vector.at<double>(0), rotation_vector.at<double>(1), rotation_vector.at<double>(2) };

            for (int i = 0; i < pnp_input_3d.size(); i++)
            {
                cv::Point3d point3d = static_cast<cv::Point3d>(pnp_input_3d[i]);
                cv::Point2d point2d = static_cast<cv::Point2d>(pnp_input_2d[i]);
                ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
                    point2d.x, 
                    point2d.y, 
                    f_, 
                    cx_, 
                    cy_
                );

                // パラメータをvectorにpush_backし、そのアドレスを渡す
                std::vector<double> point;
                point.resize(3);
                point[0] = point3d.x;
                point[1] = point3d.y;
                point[2] = point3d.z;
                
                std::vector<double> rt_vec;
                rt_vec.resize(6);
                rt_vec[0] = angle_axis[0];
                rt_vec[1] = angle_axis[1];
                rt_vec[2] = angle_axis[2];
                rt_vec[3] = translation_vector.at<double>(0);
                rt_vec[4] = translation_vector.at<double>(1);
                rt_vec[5] = translation_vector.at<double>(2);
                point_params_.push_back(point);
                camera_params_.push_back(rt_vec);

                problem_.AddResidualBlock(
                    cost_function,
                    nullptr,
                    camera_params_[i].data(),
                    point_params_[i].data()
                );
            }
        }
        void optimize()
        {
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem_, &summary);
            std::cout << summary.FullReport() << std::endl;
            std::cout << "Final cost: " << summary.final_cost << std::endl;
        };
    private:
        double f_;
        double cx_;
        double cy_;
        ceres::Problem problem_;
        std::vector<std::vector<double>> camera_params_; 
        std::vector<std::vector<double>> point_params_;  // Ceresでの計算中は変数が生存していなければいけないのでメンバ変数で保持
};