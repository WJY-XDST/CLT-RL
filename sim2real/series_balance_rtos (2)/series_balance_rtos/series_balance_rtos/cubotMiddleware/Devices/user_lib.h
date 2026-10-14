/**
 ******************************************************************************
 * @file	 user_lib.h
 * @author  Wang Hongxi
 * @version V1.0.0
 * @date    2021/2/18
 * @brief
 ******************************************************************************
 * @attention
 *
 ******************************************************************************
 */
#ifndef _USER_LIB_H_
#define _USER_LIB_H_

#include "stdint.h"
#include "main.h"
//#include "cmsis_os.h"
#include "arm_math.h"
#include "stm32h7xx_hal.h"

#ifndef user_malloc
#ifdef _CMSIS_OS_H
#define user_malloc pvPortMalloc
#else
#define user_malloc malloc
#endif
#endif

#define msin(x) (arm_sin_f32(x))
#define mcos(x) (arm_cos_f32(x))

typedef arm_matrix_instance_f32 mat;
// 若运算速度不够,可以使用q31代替f32,但是精度会降低
#define MatAdd       arm_mat_add_f32
#define MatSubtract  arm_mat_sub_f32
#define MatMultiply  arm_mat_mult_f32
#define MatTranspose arm_mat_trans_f32
#define MatInverse   arm_mat_inverse_f32
void MatInit(mat *m, uint8_t row, uint8_t col);

/* boolean type definitions */
#ifndef TRUE
#define TRUE 1 /**< boolean true  */
#endif

#ifndef FALSE
#define FALSE 0 /**< boolean fails */
#endif

/* circumference ratio */
#ifndef PI
#define PI 3.14159265354f
#endif

#define VAL_LIMIT(val, min, max)     \
    do {                             \
        if ((val) <= (min)) {        \
            (val) = (min);           \
        } else if ((val) >= (max)) { \
            (val) = (max);           \
        }                            \
    } while (0)

#define ANGLE_LIMIT_360(val, angle)     \
    do {                                \
        (val) = (angle) - (int)(angle); \
        (val) += (int)(angle) % 360;    \
    } while (0)

#define ANGLE_LIMIT_360_TO_180(val) \
    do {                            \
        if ((val) > 180)            \
            (val) -= 360;           \
    } while (0)

#define VAL_MIN(a, b)                       ((a) < (b) ? (a) : (b))                                                                                     // 取较小值
#define VAL_MAX(a, b)                       ((a) > (b) ? (a) : (b))                                                                                     // 取较大值
#define LIMIT(LIMIT_x, x_min, x_max)        ((LIMIT_x) < (x_min) ? (x_min) : ((LIMIT_x) > (x_max) ? (x_max) : (LIMIT_x)))                               // 外限幅
#define INWARD_LIMIT(LIMIT_y, y_min, y_max) ((LIMIT_y) > 0 ? ((LIMIT_y) < (y_max) ? (y_max) : (LIMIT_y)) : ((LIMIT_y) > (y_min) ? (y_min) : (LIMIT_y))) // 内限幅
#define ABS(ABS_x)                          ((ABS_x) > 0 ? (ABS_x) : (-(ABS_x)))                                                                        // 取绝对值
#define SYNC_SIGN(SYNC_a, SYNC_b)           ((SYNC_b) = ((SYNC_a) >= 0 ? ABS(SYNC_b) : -ABS(SYNC_b)))                                                   // 使b与a同号

/**
 * @brief 返回一块干净的内??,不过仍然需要强制转??为你需要的类型
 *
 * @param size 分配大小
 * @return void*
 */
void *zmalloc(size_t size);
/**
 * @brief 自定义1/sqrt(x),速度更快
 *
 * @param x x
 * @return float
 */
float invSqrt(float x);
// ???????
float Sqrt(float x);
// ????????
float abs_limit(float num, float Limit);
// ?ж????λ
float sign(float value);
// ????????
float float_deadband(float Value, float minValue, float maxValue);
// ???????
float float_constrain(float Value, float minValue, float maxValue);
// ???????
int16_t int16_constrain(int16_t Value, int16_t minValue, int16_t maxValue);
// ??????????
float loop_float_constrain(float Input, float minValue, float maxValue);
// ??? ????? 180 ~ -180
float theta_format(float Ang);

int float_to_uint(float x, float x_min, float x_max, int bits);
float uint_to_float(int x_int, float x_min, float x_max, int bits);

int float_rounding(float raw);

float *Norm3d(float *v);

float NormOf3d(float *v);

void Cross3d(float *v1, float *v2, float *res);

float Dot3d(float *v1, float *v2);

float AverageFilter(float new_data, float *buf, uint8_t len);

#define rad_format(Ang) loop_float_constrain((Ang), -PI, PI)

/**
 * @brief 四元数更新函数,即实现dq/dt=0.5Ωq
 *
 * @param q  四元数
 * @param gx
 * @param gy
 * @param gz
 * @param dt 距离上次调用的时间间隔
 */
void QuaternionUpdate(float *q, float gx, float gy, float gz, float dt);

/**
 * @brief 四元数转换成欧拉角 ZYX
 *
 * @param q
 * @param yaw
 * @param pitch
 * @param roll
 */
void QuaternionToEularAngle(float *q, float *yaw, float *pitch, float *roll);

/**
 * @brief ZYX欧拉角转换为四元数
 *
 * @param yaw
 * @param pitch
 * @param roll
 * @param q
 */
void EularAngleToQuaternion(float yaw, float pitch, float roll, float *q);

#endif
