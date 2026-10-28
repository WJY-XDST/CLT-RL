/* NX UART link. Final output-shaft Nm, not LQR inputs or VMC references.
 * USART3: PD8 TX / PD9 RX, DMA2 Stream4 RX / Stream5 TX, 2000000 8N1.
 * All application calls share one task/lock. RX interrupts only enqueue bytes.
 * No function in this module enables a motor or writes a CAN/current command.
 */
#ifndef CLT_RL_HOST_H
#define CLT_RL_HOST_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define RL_HOST_UART_BAUD 2000000u
#define RL_HOST_VERSION 1u
#define RL_HOST_PAYLOAD_MAX 192u
#define RL_HOST_HEADER_SIZE 20u
#define RL_HOST_FRAME_MAX 216u
#define RL_HOST_STATUS_BYTES 212u
#define RL_HOST_STATE_HISTORY 16u
#define RL_HOST_HELLO 1u
#define RL_HOST_STATUS 2u
#define RL_HOST_TORQUE 3u
#define RL_HOST_STOP 4u
#define RL_HOST_SENSOR_VALID 1u
#define RL_HOST_VELOCITY_VALID 2u
#define RL_HOST_EXECUTION_ACTIVE 4u
#define RL_HOST_FAULT 8u
#define RL_HOST_CALIBRATION_VERIFIED 16u
#define RL_HOST_EXECUTION_AVAILABLE 32u

typedef enum {
    RL_HOST_OK=0, RL_HOST_BAD_CONFIG=1, RL_HOST_BAD_COMMAND=2,
    RL_HOST_STALE_STATE=3, RL_HOST_COMMAND_TIMEOUT=4,
    RL_HOST_SENSOR_FAULT=5, RL_HOST_UART_FAULT=6,
    RL_HOST_RX_OVERFLOW=7, RL_HOST_BAD_ACK=8
} RLHost_Error;

typedef struct {
    uint8_t kind;
    uint16_t size;
    uint32_t session, sequence, time_us;
    uint8_t payload[RL_HOST_PAYLOAD_MAX];
} RLHost_Packet;
typedef struct {
    uint8_t data[RL_HOST_FRAME_MAX];
    uint16_t used;
    uint32_t rejected;
} RLHost_Parser;
typedef void (*RLHost_ReceiveFn)(void *context,const RLHost_Packet *packet);

typedef struct {
    uint8_t contract_sha256[32];
    float torque_limit_nm[6]; /* LB, LL, LW, RB, RL, RW; positive magnitudes. */
    float action_limit[6];    /* Same six columns; normalized references. */
    float command_min[3],command_max[3]; /* vx[m/s], yaw[rad/s], height[m]. */
    float observation_action_limit;
    uint32_t command_timeout_us,max_source_age_us;
} RLHost_Config;
typedef struct {
    uint32_t sequence,source_observation_sequence,source_sample_us,received_at_us;
    float torque_nm[6],effective_action[6],command[3];
} RLHost_Command;
typedef struct { uint32_t sequence,time_us,flags; uint8_t valid; } RLHost_StateStamp;
typedef struct {
    RLHost_Config config;
    RLHost_Parser parser;
    RLHost_StateStamp states[RL_HOST_STATE_HISTORY];
    RLHost_Command latest,inflight;
    float last_applied_action[6];
    uint32_t session,tx_sequence,last_rx_sequence,last_applied_sequence,rx_now_us;
    uint32_t received_commands,rejected_commands;
    uint8_t initialized,has_rx_sequence,has_command,has_inflight;
    uint8_t execute_requested,execution_allowed,calibration_verified;
    uint8_t execution_active,stop_requested,fault_latched;
    RLHost_Error fault_reason;
} RLHost_Context;

uint32_t RLHost_GetU32(const uint8_t *p);
void RLHost_PutU32(uint8_t *p,uint32_t value);
float RLHost_GetF32(const uint8_t *p);
void RLHost_PutF32(uint8_t *p,float value);
uint32_t RLHost_CRC32(const uint8_t *data,size_t size);
int RLHost_SeqNewer(uint32_t value,uint32_t previous);
size_t RLHost_Encode(const RLHost_Packet *packet,uint8_t *out,size_t capacity);
void RLHost_ParserFeed(RLHost_Parser *parser,const uint8_t *data,size_t size,
                       RLHost_ReceiveFn callback,void *context);

/* Init is the explicit reset of latched faults. Call it with output disabled.
 * The caller supplies the selected model's hash and physical limits; no model
 * or particular speed/height case is embedded in the firmware source. */
int RLHost_Init(RLHost_Context *context,const RLHost_Config *config);
void RLHost_SetPermission(RLHost_Context *context,int allowed,int calibrated);
void RLHost_Fault(RLHost_Context *context,RLHost_Error reason);
void RLHost_Disarm(RLHost_Context *context);
void RLHost_ReceiveBytes(RLHost_Context *context,const uint8_t *data,size_t size,uint32_t now_us);
void RLHost_Poll(RLHost_Context *context,uint32_t now_us);
/* obs is already clipped/scaled once; raw = q[4], dq[4], wheel_speed[2] in SI.
 * sample_us is the actual snapshot time, not a retimestamped old measurement.
 * Last-action fields and execution flags are written by this module. */
size_t RLHost_BuildStatus(RLHost_Context *context,const float obs[27],const float raw[10],
                         uint32_t sensor_flags,uint32_t sample_us,uint8_t *out,size_t capacity);
/* Repeats the latest fresh zero-order-held Nm request, or returns 0 and asks
 * for stop. The application's motor task owns real output and legacy control
 * exclusion. Call MarkApplied ONLY after writing the actual motor request. */
int RLHost_GetTorques(RLHost_Context *context,uint32_t now_us,RLHost_Command *command);
int RLHost_MarkApplied(RLHost_Context *context,uint32_t sequence,uint32_t now_us);

#ifndef RL_HOST_PORTABLE_TEST
/* Uses the already generated huart3 and Normal RX/TX DMA. These helpers must
 * live in DMA-accessible SRAM; their buffers are cache-line aligned. */
int RLHost_Uart3Start(RLHost_Context *context);
void RLHost_Uart3Stop(RLHost_Context *context);
void RLHost_Uart3Poll(RLHost_Context *context,uint32_t now_us);
int RLHost_Uart3SendStatus(RLHost_Context *context,const float obs[27],const float raw[10],
                          uint32_t sensor_flags,uint32_t sample_us);
/* Delegate only huart3 events from the HAL callbacks in usart.c. */
void RLHost_Uart3RxEvent(uint16_t size);
void RLHost_Uart3Error(void);
#endif
#ifdef __cplusplus
}
#endif
#endif
