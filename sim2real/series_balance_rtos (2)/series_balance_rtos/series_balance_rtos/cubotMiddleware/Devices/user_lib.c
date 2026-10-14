/**
 ******************************************************************************
 * @file	    user_lib.c
 * @author      Wang Hongxi
 * @author      modified by EmberLuo
 * @brief       算法层，
 * @details     
 * @date        2024-07-24
 * @version     V1.0
 * @copyright   Copyright (c) 2021-2121  中国矿业大学CUBOT战队
 **********************************************************************************
 * @attention
 * 硬件平台: STM32H750VBT \n
 * SDK版本：-++++
 * @par 修改日志:
 * <table>
 * <tr><th>Date         <th>Version     <th>Author      <th>Description
 * <tr><td>2024-06-04   <td>1.0         <td>EmberLuo    <td>移植哈尔滨工程大学创梦之翼战队王洪玺的惯导姿态解算开源
 * </table>
 *
 **********************************************************************************
 ==============================================================================
                            How to use this driver
 ==============================================================================

    添加 user_lib.h

    1. 创建

 **********************************************************************************
 * @attention
 * 硬件平台: STM32H750VBT \n
 * SDK版本：-++++
 * if you had modified this file, please make sure your code does not have many
 * bugs, update the version NO., write dowm your name and the date, the most
 * important is make sure the users will have clear and definite understanding
 * through your new brief.
 **********************************************************************************
 */
#include "stdlib.h"
#include "math.h"
#include "main.h"
#include "user_lib.h"

#ifdef _CMSIS_OS_H
#define user_malloc pvPortMalloc
#else
#define user_malloc malloc
#endif

void *zmalloc(size_t size)
{
    void *ptr = malloc(size);
    memset(ptr, 0, size);
    return ptr;
}

// 快速开方
float Sqrt(float x)
{
    float y;
    float delta;
    float maxError;
    if (x <= 0)
    {
        return 0;
    }
    // initial guess
    y = x / 2;
    // refine
    maxError = x * 0.001f;
    do
    {
        delta = (y * y) - x;
        y -= delta / (2 * y);
    } while (delta > maxError || delta < -maxError);
    return y;
}

/**
 * @brief 快速求平方根倒数，自定义1/sqrt(x)，速度更快
 * 
 * @param x x
 * @return float
 * @note 
 *	This is a piece of code implementing the **Fast Inverse Square Root** algorithm. 
 *	It's a method used to calculate the inverse (or reciprocal) of a square root of a floating point number in IEEE 754 format. 
 *	This method is famous for its use in the 3D computer graphics of the Quake III Arena game, where fast computation was more important than high precision.
 *	Here's a step-by-step explanation of how it works:
 * @details 代码讲解：
 *	1. `float halfx = 0.5f * x;` : This line calculates half of the input number `x`. This will be used later in the final calculation.
 *	2. `float y = x;` : This line just copies the input number `x` into `y`.
 *	3. `long i = *(long*)&y;` : This line is a bit tricky. It's using a technique called "type punning" to interpret the bits of the floating point number `y` as an integer. 
 *		This is necessary because the next step of the algorithm operates on the bit level.
 *	4. `i = 0x5f3759df - (i>>1);` : This is the core of the algorithm. It's using a magic constant `0x5f3759df` and subtracting half of the integer representation of `y`. 
 *		This gives a good first approximation of the inverse square root.
 *	5. `y = *(float*)&i;` : This line converts the integer `i` back into a floating point number.
 *	6. `y = y * (1.5f - (halfx * y * y));` : This line is a single iteration of Newton's method, which refines the approximation of the inverse square root.
 *	7. `return y;` : Finally, the function returns the calculated inverse square root.
 */
int float_to_uint(float x, float x_min, float x_max, int bits)
{
    /// Converts a float to an unsigned int, given range and number of bits ///
    float span = x_max - x_min;
    float offset = x_min;
    return (int) ((x-offset)*((float)((1<<bits)-1))/span);
}
float uint_to_float(int x_int, float x_min, float x_max, int bits)
{
	/* converts unsigned int to float, given range and number of bits */
	float span = x_max - x_min;
	float offset = x_min;
	return ((float)x_int)*span/((float)((1<<bits)-1)) + offset;
}
float invSqrt(float x) 
{
	float halfx = 0.5f * x;
	float y = x;
	long i = *(long*)&y;
	i = 0x5f3759df - (i>>1);
	y = *(float*)&i;
	y = y * (1.5f - (halfx * y * y));
	return y;
}

// 绝对值限制
float abs_limit(float num, float Limit)
{
    if (num > Limit)
    {
        num = Limit;
    }
    else if (num < -Limit)
    {
        num = -Limit;
    }
    return num;
}

// 判断符号位
float sign(float value)
{
    if (value >= 0.0f)
    {
        return 1.0f;
    }
    else
    {
        return -1.0f;
    }
}

// 浮点死区
float float_deadband(float Value, float minValue, float maxValue)
{
    if (Value < maxValue && Value > minValue)
    {
        Value = 0.0f;
    }
    return Value;
}

// 限幅函数
float float_constrain(float Value, float minValue, float maxValue)
{
    if (Value < minValue)
        return minValue;
    else if (Value > maxValue)
        return maxValue;
    else
        return Value;
}

// 限幅函数
int16_t int16_constrain(int16_t Value, int16_t minValue, int16_t maxValue)
{
    if (Value < minValue)
        return minValue;
    else if (Value > maxValue)
        return maxValue;
    else
        return Value;
}

// 循环限幅函数
float loop_float_constrain(float Input, float minValue, float maxValue)
{
    if (maxValue < minValue)
    {
        return Input;
    }

    if (Input > maxValue)
    {
        float len = maxValue - minValue;
        while (Input > maxValue)
        {
            Input -= len;
        }
    }
    else if (Input < minValue)
    {
        float len = maxValue - minValue;
        while (Input < minValue)
        {
            Input += len;
        }
    }
    return Input;
}

// 弧度格式化为-PI~PI

// 角度格式化为-180~180
float theta_format(float Ang)
{
    return loop_float_constrain(Ang, -180.0f, 180.0f);
}

int float_rounding(float raw)
{
    static int integer;
    static float decimal;
    integer = (int)raw;
    decimal = raw - integer;
    if (decimal > 0.5f)
        integer++;
    return integer;
}

// 三维向量归一化
float *Norm3d(float *v)
{
    float len = Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= len;
    v[1] /= len;
    v[2] /= len;
    return v;
}

// 计算模长
float NormOf3d(float *v)
{
    return Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// 三维向量叉乘v1 x v2
void Cross3d(float *v1, float *v2, float *res)
{
    res[0] = v1[1] * v2[2] - v1[2] * v2[1];
    res[1] = v1[2] * v2[0] - v1[0] * v2[2];
    res[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

// 三维向量点乘
float Dot3d(float *v1, float *v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// 均值滤波,删除buffer中的最后一个元素,填入新的元素并求平均值
float AverageFilter(float new_data, float *buf, uint8_t len)
{
    float sum = 0;
    for (uint8_t i = 0; i < len - 1; i++)
    {
        buf[i] = buf[i + 1];
        sum += buf[i];
    }
    buf[len - 1] = new_data;
    sum += new_data;
    return sum / len;
}

void MatInit(mat *m, uint8_t row, uint8_t col)
{
    m->numCols = col;
    m->numRows = row;
    m->pData = (float *)zmalloc(row * col * sizeof(float));
}


//------------------------------------functions below are not used in this demo-------------------------------------------------
//----------------------------------you can read them for learning or programming-----------------------------------------------
//----------------------------------they could also be helpful for further design-----------------------------------------------

/**
 * @brief        Update quaternion
 */
void QuaternionUpdate(float *q, float gx, float gy, float gz, float dt)
{
    float qa, qb, qc;

    gx *= 0.5f * dt;
    gy *= 0.5f * dt;
    gz *= 0.5f * dt;
    qa = q[0];
    qb = q[1];
    qc = q[2];
    q[0] += (-qb * gx - qc * gy - q[3] * gz);
    q[1] += (qa * gx + qc * gz - q[3] * gy);
    q[2] += (qa * gy - qb * gz + q[3] * gx);
    q[3] += (qa * gz + qb * gy - qc * gx);
}

/**
 * @brief        Convert quaternion to eular angle
 */
void QuaternionToEularAngle(float *q, float *yaw, float *pitch, float *roll)
{
    *yaw   = atan2f(2.0f * (q[0] * q[3] + q[1] * q[2]), 2.0f * (q[0] * q[0] + q[1] * q[1]) - 1.0f) * 57.295779513f;
    *pitch = atan2f(2.0f * (q[0] * q[1] + q[2] * q[3]), 2.0f * (q[0] * q[0] + q[3] * q[3]) - 1.0f) * 57.295779513f;
    *roll  = asinf(2.0f * (q[0] * q[2] - q[1] * q[3])) * 57.295779513f;
}

/**
 * @brief        Convert eular angle to quaternion
 */
void EularAngleToQuaternion(float yaw, float pitch, float roll, float *q)
{
    float cosPitch, cosYaw, cosRoll, sinPitch, sinYaw, sinRoll;
    yaw /= 57.295779513f;
    pitch /= 57.295779513f;
    roll /= 57.295779513f;
    cosPitch = arm_cos_f32(pitch / 2);
    cosYaw   = arm_cos_f32(yaw / 2);
    cosRoll  = arm_cos_f32(roll / 2);
    sinPitch = arm_sin_f32(pitch / 2);
    sinYaw   = arm_sin_f32(yaw / 2);
    sinRoll  = arm_sin_f32(roll / 2);
    q[0]     = cosPitch * cosRoll * cosYaw + sinPitch * sinRoll * sinYaw;
    q[1]     = sinPitch * cosRoll * cosYaw - cosPitch * sinRoll * sinYaw;
    q[2]     = sinPitch * cosRoll * sinYaw + cosPitch * sinRoll * cosYaw;
    q[3]     = cosPitch * cosRoll * sinYaw - sinPitch * sinRoll * cosYaw;
}
