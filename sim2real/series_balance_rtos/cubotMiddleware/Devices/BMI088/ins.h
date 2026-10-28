/**
 ******************************************************************************
 * @file    ins_task.h
 * @author  Wang Hongxi
 * @author  annotation and modification by NeoZeng
 * @version V2.0.0
 * @date    2022/2/23
 * @brief
 ******************************************************************************
 * @attention INS任务的初始化不要放入实时系统!应该由application拥有实例,随后在
 *            应用层调用初始化函数.
 *
 ******************************************************************************
 */
#ifndef _INS_TASK_H_
#define _INS_TASK_H_

//#include "quaternions_EKF.h"
#include <stdint.h>

#define X               0
#define Y               1
#define Accel_Z         2

#define INS_TASK_PERIOD 1

typedef struct _IMU_InitData_t
{
    float accel[3];
    float gyro[3];
    float temp_when_cali;
    float temperature;
    float accel_scale;
    float gyro_offset[3];
    float g_norm;
    uint8_t (*Init)(void);
    void (*Read)(struct _IMU_InitData_t *);
} IMU_InitData_t;

typedef struct
{
    float accel[3];        // 加速度
    float gyro[3];         // 角速度
    float roll;            // 滚转角? \phi
    float pitch;           // 俯仰角θ \theta
    float yaw;             // 偏航角ψ \psi
    float roll_last;            // 滚转角? \phi
    float pitch_last;           // 俯仰角θ \theta
    float yaw_last; 
	float roll_speed;            // 滚转角? \phi
    float pitch_speed;           // 俯仰角θ \theta
    float yaw_speed;             // 偏航角ψ \psi
    float yaw_total_angle; // 最终解算得到的角度,以及yaw转动的总角度(方便多圈控制)
    float q[4];            // 四元数
} Attitude_t;

typedef struct
{
    float MotionAccel_b[3]; // 机体坐标加速度
    float MotionAccel_n[3]; // 绝对系加速度
    float AccelLPF;         // 加速度低通滤波系数
    // bodyframe在绝对系的向量表示
    float xn[3];
    float yn[3];
    float zn[3];
    // 加速度在机体系和XY两轴的夹角
    float atanxz;        // 加速度在机体系和X轴的夹角
    float atanyz;        // 加速度在机体系和Y轴的夹角
    Attitude_t attitude; // 姿态数据
    uint8_t init_done;   // 初始化成功标志位
} INS_t;

/* 用于修正安装误差的参数 */
typedef struct
{
    uint8_t flag;
    float scale[3];
    float yaw;
    float pitch;
    float roll;
} IMU_CorrectParam_t;

/**
 * @brief IMU初始化
 * 
 */
typedef uint8_t (*IMU_Init)(void);

/**
 * @brief IMU读取
 * 
 */
typedef void (*IMU_Read)(IMU_InitData_t *);

/**
 * @brief 初始化惯导解算系统
 * 
 * @return uint8_t 
 */
uint8_t INS_Init(IMU_InitData_t * imu_data);


Attitude_t *INS_GetAttitude(IMU_InitData_t * imu_data);

/**
 * @brief 机体系到惯性系的变换函数
 *
 * @param vecBF body frame
 * @param vecEF earth frame
 * @param q
 */
void BodyFrameToEarthFrame(const float *vecBF, float *vecEF, float *q);

/**
 * @brief 惯性系转换到机体系
 *
 * @param vecEF
 * @param vecBF
 * @param q
 */
void EarthFrameToBodyFrame(const float *vecEF, float *vecBF, float *q);

void INS_Task(void);

extern Attitude_t *INS_attitude;
extern  IMU_InitData_t *IMU_data;
extern INS_t ins;                           
#endif
