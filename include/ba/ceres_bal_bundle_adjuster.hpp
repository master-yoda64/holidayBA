#pragma once

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
                f
            )
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

class CeresBalBundleAdjuster : public BundleAdjusterBase<CameraModelPinholeBal>
{
public:
    CeresBalBundleAdjuster(){};
    ~CeresBalBundleAdjuster(){};

    std::vector<OptResult> optimize() override;
    OptResult optimize_camera(
        const std::vector<Observation>& obs,
        const CameraModelPinholeBal& cam
    ) override;
    void load_data(std::string path);
    std::vector<CameraModelPinholeBal> get_cameras()
    {
        return cameras_;
    }
  
private:
    ceres::Problem problem_;
    std::vector<std::vector<double>> camera_params_; 
    void add_residuals_camera_id(int camera_id);
};