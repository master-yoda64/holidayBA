#pragma once
#include <camera/camera_model_base.hpp>
#include <camera/camera_model_pinhole_bal.hpp>

template <typename CameraModelT>
class CameraTypeTraits 
{
    CameraTypeTraits() = delete;
};

template <>
class CameraTypeTraits<CameraModelPinholeBal>
{
public:
    using CameraModelType = CameraModelPinholeBal;
};

template <>
class CameraTypeTraits<CameraModelBase>
{
public:
    using CameraModelType = CameraModelBase;
};

template<typename T>
using CameraModelType_t = typename CameraTypeTraits<T>::CameraModelType;