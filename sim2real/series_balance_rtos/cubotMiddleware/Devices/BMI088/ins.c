/**
 ******************************************************************************
 * @file    ins_task.c
 * @author  Wang Hongxi
 * @author  annotation and modificaiton by neozng
 * @version V2.0.0
 * @date    2022/2/23
 * @brief
 ******************************************************************************
 * @attention
 *
 ******************************************************************************
 */
#include "ins.h"
#include "user_lib.h"
#include "bmi088.h"
#include "mpu6050.h"
#include "fast_math_functions.h"
#include "quaternions_EKF.h"
#include "gyro_task.h"
extern float pitch_com;
Attitude_t *INS_attitude;
IMU_InitData_t *IMU_data;
 INS_t ins                           = {0};
static IMU_CorrectParam_t IMU_paramCorrect = {0};

const float xb[3] = {1, 0, 0};
const float yb[3] = {0, 1, 0};
const float zb[3] = {0, 0, 1};

// 用于获取两次采样之间的时间间隔
static uint32_t INS_DWT_Count = 0;

static void IMU_Param_Correction(IMU_CorrectParam_t *param, float gyro[3], float accel[3]);

// 使用加速度计的数据初始化Roll和Pitch,而Yaw置0,这样可以避免在初始时候的姿态估计误差
/**
 * @brief
 *
 * @param init_q4
 */
static void InitQuaternion(float *init_q4, IMU_InitData_t *imu_data)
{
    float acc_init[3]     = {0};
    float gravity_norm[3] = {0, 0, 1}; // 导航系重力加速度矢量,归一化后为(0,0,1)
    float axis_rot[3]     = {0};       // 旋转轴
    // 读取100次加速度计数据,取平均值作为初始值
    for (uint8_t i = 0; i < 100; ++i) {
        imu_data->Read(imu_data);
        acc_init[X] += imu_data->accel[X];
        acc_init[Y] += imu_data->accel[Y];
        acc_init[Accel_Z] += imu_data->accel[Accel_Z];
        DWT_Delay_s(0.001);
    }
    for (uint8_t i = 0; i < 3; ++i)
        acc_init[i] /= 100;
    Norm3d(acc_init);
    // 计算原始加速度矢量和导航系重力加速度矢量的夹角
    float angle = acosf(Dot3d(acc_init, gravity_norm));
    Cross3d(acc_init, gravity_norm, axis_rot);
    Norm3d(axis_rot);
    init_q4[0] = cosf(angle / 2.0f);
    for (uint8_t i = 0; i < 2; ++i)
        init_q4[i + 1] = axis_rot[i] * sinf(angle / 2.0f); // 轴角公式,第三轴为0(没有z轴分量)
}
// 将初始化函数和读取函数传入结构体
static void IMU_FuncInit(void)
{
	bmi088.bmi088_Data.Init   = BMI088_Init;
    bmi088.bmi088_Data.Read   = BMI088_Read;
    mpu6050.mpu6050_Data.Init = MPU6050_Init;
    mpu6050.mpu6050_Data.Read = MPU6050_Read;
}
/**
 * @brief 初始化惯导解算系统
 *
 * @return uint8_t
 */
uint8_t INS_Init(IMU_InitData_t *imu_data)
{
    IMU_FuncInit();
	IMU_data = imu_data;
    if (ins.init_done)
        return ins.init_done;
    else {
        while (imu_data->Init() != 0) ;
        IMU_paramCorrect.scale[X] = 1;
        IMU_paramCorrect.scale[Y] = 1;
        IMU_paramCorrect.scale[Accel_Z] = 1;
        IMU_paramCorrect.yaw      = 0;
        IMU_paramCorrect.pitch    = -2 ;
        IMU_paramCorrect.roll     = -1;
        IMU_paramCorrect.flag     = 1;

        float init_quaternion[4] = {0};
        InitQuaternion(init_quaternion, imu_data);
        IMU_QuaternionEKF_Init(init_quaternion, 10, 0.001, 1000000, 1, 0);
        // imu heat init

        // noise of accel is relatively big and of high freq,thus lpf is used
        ins.AccelLPF = 0.0085;
        DWT_GetDeltaT(&INS_DWT_Count);
        ins.init_done = 1;
		
        return ins.init_done;
    }
}
/**
 * @brief INS解算
 * @attention 此函数放入实时系统中,以1kHz频率运行。p.s. osDelay(1);
 *
 * @return Attitude_t*
 */
Attitude_t *INS_GetAttitude(IMU_InitData_t *imu_data)
{
    /* 注意以1kHz的频率运行此任务 */
    static Attitude_t insAttitude = {0};
    static float dt = 0, t = 0;
    const float gravity[3] = {0, 0, 9.81f};

    dt = DWT_GetDeltaT(&INS_DWT_Count);
    t += dt;

    imu_data->Read(imu_data);

    ins.attitude.accel[X] = imu_data->accel[X] + bmi088_ax;
    ins.attitude.accel[Y] = imu_data->accel[Y] + bmi088_ay;
    ins.attitude.accel[Accel_Z] = imu_data->accel[Accel_Z];
    ins.attitude.gyro[X]  = imu_data->gyro[X];
    ins.attitude.gyro[Y]  = imu_data->gyro[Y];
    ins.attitude.gyro[Accel_Z]  = imu_data->gyro[Accel_Z];

    // demo function,用于修正安装误差,可以不管,暂时没用
    IMU_Param_Correction(&IMU_paramCorrect, ins.attitude.gyro, ins.attitude.accel);

    // 计算重力加速度矢量和b系的XY两轴的夹角,可用作功能扩展,暂时没用
    ins.atanxz = -atan2f(ins.attitude.accel[X], ins.attitude.accel[Accel_Z]) * 180 / PI;
    ins.atanyz = atan2f(ins.attitude.accel[Y], ins.attitude.accel[Accel_Z]) * 180 / PI;

    // 核心函数,EKF更新四元数
    IMU_QuaternionEKF_Update(ins.attitude.gyro[X], ins.attitude.gyro[Y], ins.attitude.gyro[Accel_Z], ins.attitude.accel[X], ins.attitude.accel[Y], ins.attitude.accel[Accel_Z], dt);

    memcpy(ins.attitude.q, QEKF_INS.q, sizeof(QEKF_INS.q));

    // 机体系基向量转换到导航坐标系，选取惯性系为导航系
    BodyFrameToEarthFrame(xb, ins.xn, ins.attitude.q);
    BodyFrameToEarthFrame(yb, ins.yn, ins.attitude.q);
    BodyFrameToEarthFrame(zb, ins.zn, ins.attitude.q);

    // 将重力从导航坐标系n转换到机体系b,随后根据加速度计数据计算运动加速度
    float gravity_b[3];
    EarthFrameToBodyFrame(gravity, gravity_b, ins.attitude.q);
    for (uint8_t i = 0; i < 3; ++i) // 同样过一个低通滤波
    {
        ins.MotionAccel_b[i] = (ins.attitude.accel[i] - gravity_b[i]) * dt / (ins.AccelLPF + dt) + ins.MotionAccel_b[i] * ins.AccelLPF / (ins.AccelLPF + dt);
    }
    BodyFrameToEarthFrame(ins.MotionAccel_b, ins.MotionAccel_n, ins.attitude.q); // 转换回导航系n

    ins.attitude.yaw             = QEKF_INS.yaw;
    ins.attitude.pitch           = QEKF_INS.pitch;
    ins.attitude.roll            = QEKF_INS.roll;
    ins.attitude.yaw_total_angle = QEKF_INS.yaw_total_angle;
	ins.attitude.yaw_speed       = (QEKF_INS.gyro[2]);
	ins.attitude.pitch_speed     = (QEKF_INS.gyro[0]);
	ins.attitude.pitch_last      = ins.attitude.pitch;
    insAttitude                  = ins.attitude;

    return &insAttitude;
}

/**
 * @brief          Transform 3dvector from BodyFrame to EarthFrame
 * @param[1]       vector in BodyFrame
 * @param[2]       vector in EarthFrame
 * @param[3]       quaternion
 */
void BodyFrameToEarthFrame(const float *vecBF, float *vecEF, float *q)
{
    vecEF[0] = 2.0f * ((0.5f - q[2] * q[2] - q[3] * q[3]) * vecBF[0] +
                       (q[1] * q[2] - q[0] * q[3]) * vecBF[1] +
                       (q[1] * q[3] + q[0] * q[2]) * vecBF[2]);

    vecEF[1] = 2.0f * ((q[1] * q[2] + q[0] * q[3]) * vecBF[0] +
                       (0.5f - q[1] * q[1] - q[3] * q[3]) * vecBF[1] +
                       (q[2] * q[3] - q[0] * q[1]) * vecBF[2]);

    vecEF[2] = 2.0f * ((q[1] * q[3] - q[0] * q[2]) * vecBF[0] +
                       (q[2] * q[3] + q[0] * q[1]) * vecBF[1] +
                       (0.5f - q[1] * q[1] - q[2] * q[2]) * vecBF[2]);
}

/**
 * @brief          Transform 3dvector from EarthFrame to BodyFrame
 * @param[1]       vector in EarthFrame
 * @param[2]       vector in BodyFrame
 * @param[3]       quaternion
 */
void EarthFrameToBodyFrame(const float *vecEF, float *vecBF, float *q)
{
    vecBF[0] = 2.0f * ((0.5f - q[2] * q[2] - q[3] * q[3]) * vecEF[0] +
                       (q[1] * q[2] + q[0] * q[3]) * vecEF[1] +
                       (q[1] * q[3] - q[0] * q[2]) * vecEF[2]);

    vecBF[1] = 2.0f * ((q[1] * q[2] - q[0] * q[3]) * vecEF[0] +
                       (0.5f - q[1] * q[1] - q[3] * q[3]) * vecEF[1] +
                       (q[2] * q[3] + q[0] * q[1]) * vecEF[2]);

    vecBF[2] = 2.0f * ((q[1] * q[3] + q[0] * q[2]) * vecEF[0] +
                       (q[2] * q[3] - q[0] * q[1]) * vecEF[1] +
                       (0.5f - q[1] * q[1] - q[2] * q[2]) * vecEF[2]);
}

/**
 * @brief reserved.用于修正IMU安装误差与标度因数误差,即陀螺仪轴和云台轴的安装偏移
 *
 * @param param IMU参数
 * @param gyro  角速度
 * @param accel 加速度
 */
static void IMU_Param_Correction(IMU_CorrectParam_t *param, float gyro[3], float accel[3])
{
    static float lastYawOffset, lastPitchOffset, lastRollOffset;
    static float c_11, c_12, c_13, c_21, c_22, c_23, c_31, c_32, c_33;
    float cosPitch, cosYaw, cosRoll, sinPitch, sinYaw, sinRoll;

    if (fabsf(param->yaw - lastYawOffset) > 0.001f ||
        fabsf(param->pitch - lastPitchOffset) > 0.001f ||
        fabsf(param->roll - lastRollOffset) > 0.001f || param->flag) {
        cosYaw   = arm_cos_f32(param->yaw / 57.295779513f);
        cosPitch = arm_cos_f32(param->pitch / 57.295779513f);
        cosRoll  = arm_cos_f32(param->roll / 57.295779513f);
        sinYaw   = arm_sin_f32(param->yaw / 57.295779513f);
        sinPitch = arm_sin_f32(param->pitch / 57.295779513f);
        sinRoll  = arm_sin_f32(param->roll / 57.295779513f);

        // 1.yaw(alpha) 2.pitch(beta) 3.roll(gamma)
        c_11        = cosYaw * cosRoll + sinYaw * sinPitch * sinRoll;
        c_12        = cosPitch * sinYaw;
        c_13        = cosYaw * sinRoll - cosRoll * sinYaw * sinPitch;
        c_21        = cosYaw * sinPitch * sinRoll - cosRoll * sinYaw;
        c_22        = cosYaw * cosPitch;
        c_23        = -sinYaw * sinRoll - cosYaw * cosRoll * sinPitch;
        c_31        = -cosPitch * sinRoll;
        c_32        = sinPitch;
        c_33        = cosPitch * cosRoll;
        param->flag = 0;
    }
    float gyro_temp[3];
    for (uint8_t i = 0; i < 3; ++i)
        gyro_temp[i] = gyro[i] * param->scale[i];

    gyro[X] = c_11 * gyro_temp[X] +
              c_12 * gyro_temp[Y] +
              c_13 * gyro_temp[Accel_Z];
    gyro[Y] = c_21 * gyro_temp[X] +
              c_22 * gyro_temp[Y] +
              c_23 * gyro_temp[Accel_Z];
    gyro[Accel_Z] = c_31 * gyro_temp[X] +
              c_32 * gyro_temp[Y] +
              c_33 * gyro_temp[Accel_Z];

    float accel_temp[3];
    for (uint8_t i = 0; i < 3; ++i)
        accel_temp[i] = accel[i];

    accel[X] = c_11 * accel_temp[X] +
               c_12 * accel_temp[Y] +
               c_13 * accel_temp[Accel_Z];
    accel[Y] = c_21 * accel_temp[X] +
               c_22 * accel_temp[Y] +
               c_23 * accel_temp[Accel_Z];
    accel[Accel_Z] = c_31 * accel_temp[X] +
               c_32 * accel_temp[Y] +
               c_33 * accel_temp[Accel_Z];

    lastYawOffset   = param->yaw;
    lastPitchOffset = param->pitch;
    lastRollOffset  = param->roll;
}

uint32_t INS_HighWaterMark;
void INS_Task(void)
{
    while (1) {
//        INS_attitude = INS_GetAttitude(IMU_data);
//#if INCLUDE_uxTaskGetStackHighWaterMark
//        INS_HighWaterMark = uxTaskGetStackHighWaterMark(NULL);
//#endif
//        osDelay(1);
    }
}
