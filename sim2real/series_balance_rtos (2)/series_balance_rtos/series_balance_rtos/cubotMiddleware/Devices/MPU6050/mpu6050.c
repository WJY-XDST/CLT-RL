#include "mpu6050.h"
#include "user_lib.h"
#include "kalman_filter.h"

MPU6050_t mpu6050       = {0};
float MPU6050_ACCEL_SEN = MPU6050_ACCEL_4G_SEN;
float MPU6050_GYRO_SEN  = MPU6050_GYRO_1000_SEN;

int16_t mpu6050_cali_count = 0;

// MPU6050 电源开启
static void MPU6050_PowerOn(void)
{
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);
}

// 写数据到MPU6050寄存器
static uint8_t MPU6050_WriteByte(uint8_t reg_add, uint8_t reg_dat)
{
    return Sensors_I2C_WriteRegister(MPU6050_ADDRESS, reg_add, 1, &reg_dat);
}

// 从MPU6050寄存器读取数据
static uint8_t MPU6050_ReadData(uint8_t reg_add, uint8_t *Read, uint8_t num)
{
    return Sensors_I2C_ReadRegister(MPU6050_ADDRESS, reg_add, num, Read);
}

static void MPU6050_Calibrate_Offset(IMU_InitData_t *mpu6050_data);

uint8_t MPU6050_Init(void)
{
    uint8_t TAddr = 0;
    MPU6050_PowerOn(); // 上电

    mpu6050.WriteState = MPU6050_WriteByte(0x6B, 0x01); // 休眠
    DWT_Delay_s(0.5);
    mpu6050.WriteState = MPU6050_WriteByte(0x6B, 0x00); // 解除休眠状态0x00
    DWT_Delay_s(0.05);
    MPU6050_ReadData(MPU6050_RA_WHO_AM_I, &TAddr, 1);
    if (TAddr != 0x68)
        return 1;
    mpu6050.WriteState = MPU6050_WriteByte(0x6B, 0x00); // 解除休眠状态0x00
    DWT_Delay_s(0.05);
    mpu6050.WriteState += MPU6050_WriteByte(0x19, 0x00); // 采样频率（1KHz）
    DWT_Delay_s(0.05);
    mpu6050.WriteState += MPU6050_WriteByte(0x1A, 0x03); // 低通滤波
    DWT_Delay_s(0.05);
    mpu6050.WriteState += MPU6050_WriteByte(0x1B, 0x10); // 陀螺仪量程，当寄存器0x1B的值为0x10时，陀螺仪量程为
    DWT_Delay_s(0.05);
    mpu6050.WriteState += MPU6050_WriteByte(0x1C, 0x09); // 加速度量程，当寄存器0x1C的值为0x09时，加速度量程为
    DWT_Delay_s(0.05);
	MPU6050_Calibrate_Offset(&mpu6050.mpu6050_Data);
    return 0;
}

// 防止零漂
void MPU6050_Calibrate_Offset(IMU_InitData_t *mpu6050_data)
{
    static float startTime;
    static uint16_t CaliTimes = 1000;
    static float gyroDiff[3], gNormDiff;

    uint8_t accBuf[6]  = {0};
    uint8_t gyroBuf[6] = {0};
    int16_t mpu6050_raw_temp;
    float gyroMax[3], gyroMin[3];
    float gNormTemp = 0.0f, gNormMax = 0.0f, gNormMin = 0.0f;

    startTime = DWT_GetTimeline_s();
    do {
        if (DWT_GetTimeline_s() - startTime > 12) {
            // 在线校准超时，使用预置值校准
            mpu6050_data->gyro_offset[0] = MPU6050_GxOFFSET;
            mpu6050_data->gyro_offset[1] = MPU6050_GyOFFSET;
            mpu6050_data->gyro_offset[2] = MPU6050_GzOFFSET;
            mpu6050_data->g_norm         = MPU6050_gNORM;
            break;
        }

        DWT_Delay_s(0.005);
        mpu6050_data->g_norm         = 0;
        mpu6050_data->gyro_offset[0] = 0;
        mpu6050_data->gyro_offset[1] = 0;
        mpu6050_data->gyro_offset[2] = 0;

        for (uint16_t i = 0; i < CaliTimes; ++i) {
            MPU6050_ReadData(MPU6050_ACC_OUT, accBuf, 6);
            mpu6050_raw_temp       = (int16_t)(accBuf[0] << 8) | accBuf[1];
            mpu6050_data->accel[0] = mpu6050_raw_temp * MPU6050_ACCEL_SEN;
            mpu6050_raw_temp       = (int16_t)(accBuf[2] << 8) | accBuf[3];
            mpu6050_data->accel[1] = mpu6050_raw_temp * MPU6050_ACCEL_SEN;
            mpu6050_raw_temp       = (int16_t)(accBuf[4] << 8) | accBuf[5];
            mpu6050_data->accel[2] = mpu6050_raw_temp * MPU6050_ACCEL_SEN;

            gNormTemp = sqrtf(mpu6050_data->accel[0] * mpu6050_data->accel[0] +
                              mpu6050_data->accel[1] * mpu6050_data->accel[1] +
                              mpu6050_data->accel[2] * mpu6050_data->accel[2]);
            mpu6050_data->g_norm += gNormTemp;

            MPU6050_ReadData(MPU6050_GYRO_OUT, gyroBuf, 6);
            mpu6050_raw_temp      = (int16_t)(gyroBuf[0] << 8) | gyroBuf[1];
            mpu6050_data->gyro[0] = mpu6050_raw_temp * MPU6050_GYRO_SEN;
            mpu6050_data->gyro_offset[0] += mpu6050_data->gyro[0];
            mpu6050_raw_temp      = (int16_t)(gyroBuf[2] << 8) | gyroBuf[3];
            mpu6050_data->gyro[1] = mpu6050_raw_temp * MPU6050_GYRO_SEN;
            mpu6050_data->gyro_offset[1] += mpu6050_data->gyro[1];
            mpu6050_raw_temp      = (int16_t)(gyroBuf[4] << 8) | gyroBuf[5];
            mpu6050_data->gyro[2] = mpu6050_raw_temp * MPU6050_GYRO_SEN;
            mpu6050_data->gyro_offset[2] += mpu6050_data->gyro[2];

            if (i == 0) {
                gNormMax = gNormTemp;
                gNormMin = gNormTemp;
                for (uint8_t j = 0; j < 3; ++j) {
                    gyroMax[j] = mpu6050_data->gyro[j];
                    gyroMin[j] = mpu6050_data->gyro[j];
                }
            } else {
                if (gNormTemp > gNormMax)
                    gNormMax = gNormTemp;
                if (gNormTemp < gNormMin)
                    gNormMin = gNormTemp;
                for (uint8_t j = 0; j < 3; ++j) {
                    if (mpu6050_data->gyro[j] > gyroMax[j])
                        gyroMax[j] = mpu6050_data->gyro[j];
                    if (mpu6050_data->gyro[j] < gyroMin[j])
                        gyroMin[j] = mpu6050_data->gyro[j];
                }
            }

            gNormDiff = gNormMax - gNormMin;
            for (uint8_t j = 0; j < 3; ++j)
                gyroDiff[j] = gyroMax[j] - gyroMin[j];
            if (gNormDiff > 0.5f ||
                gyroDiff[0] > 0.15f ||
                gyroDiff[1] > 0.15f ||
                gyroDiff[2] > 0.15f)
                //            {
                break;
            //            }

            DWT_Delay_s(0.0005);
        }

        mpu6050_data->g_norm /= (float)CaliTimes;
        for (uint8_t i = 0; i < 3; ++i)
            mpu6050_data->gyro_offset[i] /= (float)CaliTimes;

        mpu6050_cali_count++;

    } while (gNormDiff > 0.5f ||
             fabsf(mpu6050_data->g_norm - 9.8f) > 1.0f ||
             gyroDiff[0] > 0.15f ||
             gyroDiff[1] > 0.15f ||
             gyroDiff[2] > 0.15f ||
             fabsf(mpu6050_data->gyro_offset[0]) > 0.06f ||
             fabsf(mpu6050_data->gyro_offset[1]) > 0.02f ||
             fabsf(mpu6050_data->gyro_offset[2]) > 0.02f);
    // 若出不了while，说明校准条件恶劣，超时则使用预置值校准
    mpu6050_data->accel_scale = 9.81f / mpu6050_data->g_norm;
}

/**
 * @brief   读取陀螺仪数据
 */
void MPU6050_Read(IMU_InitData_t *mpu6050_data)
{
    static uint8_t accBuf[6];
    static uint8_t gyroBuf[6];
    static int16_t mpu6050_raw_temp;
    MPU6050_ReadData(MPU6050_ACC_OUT, accBuf, 6);
    mpu6050_raw_temp       = (int16_t)(accBuf[0] << 8) | accBuf[1];
    mpu6050_data->accel[0] = mpu6050_raw_temp * MPU6050_ACCEL_SEN * mpu6050_data->accel_scale;
    mpu6050_raw_temp       = (int16_t)(accBuf[2] << 8) | accBuf[3];
    mpu6050_data->accel[1] = mpu6050_raw_temp * MPU6050_ACCEL_SEN * mpu6050_data->accel_scale;
    mpu6050_raw_temp       = (int16_t)(accBuf[4] << 8) | accBuf[5];
    mpu6050_data->accel[2] = mpu6050_raw_temp * MPU6050_ACCEL_SEN * mpu6050_data->accel_scale;

    MPU6050_ReadData(MPU6050_GYRO_OUT, gyroBuf, 6);
    mpu6050_raw_temp      = (int16_t)(gyroBuf[0] << 8) | gyroBuf[1];
    mpu6050_data->gyro[0] = mpu6050_raw_temp * MPU6050_GYRO_SEN - mpu6050_data->gyro_offset[0];
    mpu6050_raw_temp      = (int16_t)(gyroBuf[2] << 8) | gyroBuf[3];
    mpu6050_data->gyro[1] = mpu6050_raw_temp * MPU6050_GYRO_SEN - mpu6050_data->gyro_offset[1];
    mpu6050_raw_temp      = (int16_t)(gyroBuf[4] << 8) | gyroBuf[5];
    mpu6050_data->gyro[2] = mpu6050_raw_temp * MPU6050_GYRO_SEN - mpu6050_data->gyro_offset[2];
}
