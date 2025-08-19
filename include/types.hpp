#pragma once
#include <eigen3/Eigen/Dense>

using Mat22_t = Eigen::Matrix2f;

using Mat33_t = Eigen::Matrix3f;

using Mat44_t = Eigen::Matrix4f;

using Mat34_t = Eigen::Matrix<float, 3, 4>;

using MatX_t = Eigen::MatrixXf;

using IntMatX_t = Eigen::MatrixXi;

using Vec2_t = Eigen::Vector2f;

using Vec3_t = Eigen::Vector3f;

using Vec4_t = Eigen::Vector4f;

using VecX_t = Eigen::VectorXf;

using IntVecX_t = Eigen::VectorXi;

typedef Eigen::Matrix<float, 4, 4> qRepMat_t; // reprentation for the base (1, i, j, k)

typedef Eigen::MatrixXf LieAlgMat_t; // 

typedef LieAlgMat_t TSO3Mat_t; // Lie algebra for SO(3)

typedef LieAlgMat_t TSE3Mat_t; // Lie algebra for SE(3)

typedef Mat33_t  SO3Mat_t; // SO(3) matrices

typedef Vec3_t Trans_t; // translation vector

typedef Mat44_t  SE3Mat_t; // SE(3) matrices


// mathematical type checker on compiler
template<typename T> concept CheckLieAlgMat_t = std::is_same<std::remove_cvref_t<T>, LieAlgMat_t>::value;

template<typename T> concept CheckTSO3Mat_t = std::is_same<std::remove_cvref_t<T>, TSO3Mat_t>::value;

template<typename T> concept CheckTSE3Mat_t = std::is_same<std::remove_cvref_t<T>, TSE3Mat_t>::value;

template<typename T> concept CheckSO3Mat_t = std::is_same<std::remove_cvref_t<T>, SO3Mat_t>::value;

template<typename T> concept CheckSE3Mat_t = std::is_same<std::remove_cvref_t<T>, SE3Mat_t>::value;

template<typename T> concept CheckqRepMat_t = std::is_same<std::remove_cvref_t<T>, qRepMat_t>::value;

template<typename T> concept CheckDiagonalMatrix3f = std::is_same<std::remove_cvref_t<T>, Eigen::DiagonalMatrix<float, 3, 3>>::value;