#include "app_x-cube-ai.h"

#include "network.h"
#include "network_data.h"
#include "ai_platform.h"
#include "bsp_driver_sd.h"

#include "fatfs.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "cmsis_os.h"
/*
   Eğer bazı X-CUBE-AI sürümlerinde *_SIZE_BYTES yoksa
   uint8 model olduğu için SIZE ile byte sayısı aynı kabul edilir.
*/
#ifndef AI_NETWORK_IN_1_SIZE_BYTES
#define AI_NETWORK_IN_1_SIZE_BYTES AI_NETWORK_IN_1_SIZE
#endif

#ifndef AI_NETWORK_OUT_1_SIZE_BYTES
#define AI_NETWORK_OUT_1_SIZE_BYTES AI_NETWORK_OUT_1_SIZE
#endif

#define TEST_ROOT       "0:/test"
#define MAX_CLASSES     7
#define PATH_BUF_SIZE   128

/*
   1 yaparsan her dosyanın sonucunu basar.
   0 yaparsan sadece sınıf özetlerini ve confusion matrix basar.
   Büyük test setinde 0 daha mantıklı.
*/
#define PRINT_EVERY_FILE  0

/*
   SD klasör sırası:
   0:/test/00 -> flies_back_leaf
   0:/test/01 -> flies_leaf
   ...
*/
static const char* CLASS_NAMES[] = {
    "flies_back_leaf",
    "flies_leaf",
    "healthy_back_leaf",
    "healthy_leaf",
    "nutdef_leaf",
    "trips_leaf",
    "tuta_leaf",
};

#define CLASS_COUNT  (sizeof(CLASS_NAMES) / sizeof(CLASS_NAMES[0]))

AI_ALIGNED(4)
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

/* Input ayrı RAM yemesin. X-CUBE-AI activation içindeki input alanını kullanacağız. */
static ai_u8 *ai_input_data = NULL;

/* Output 7 byte civarı, kalsa sorun değil. */
AI_ALIGNED(4)
static ai_u8 ai_output_data[AI_NETWORK_OUT_1_SIZE_BYTES];

static ai_handle network = AI_HANDLE_NULL;

static ai_buffer ai_input[AI_NETWORK_IN_NUM];
static ai_buffer ai_output[AI_NETWORK_OUT_NUM];

static uint8_t ai_ready = 0;
static uint8_t test_done = 0;

static uint32_t confusion[MAX_CLASSES][MAX_CLASSES];

static void print_ai_error(const char* msg, ai_error err)
{
    printf("%s error type=%d code=%d\r\n", msg, err.type, err.code);
}

static int sd_mount_local(void)
{
    FRESULT res;
    uint8_t i;

    printf("Mounting SD card...\r\n");
    printf("SDPath = %s\r\n", SDPath);
    printf("BSP_SD_IsDetected = %u\r\n", BSP_SD_IsDetected());

    osDelay(500);

    for (i = 0; i < 5; i++)
    {
        printf("BSP_SD_Init try %u...\r\n", i);

        if (BSP_SD_Init() == MSD_OK)
        {
            printf("BSP_SD_Init OK\r\n");
            break;
        }

        printf("BSP_SD_Init failed\r\n");
        osDelay(500);
    }

    printf("Card state = %u\r\n", BSP_SD_GetCardState());

    res = f_mount(&SDFatFS, "0:", 1);

    printf("f_mount result = %d\r\n", res);

    if (res != FR_OK)
    {
        printf("SD mount failed. FRESULT=%d\r\n", res);
        return 0;
    }

    printf("SD mount OK\r\n");
    return 1;
}

static int ai_local_init(void)
{
    ai_error err;

    printf("AI init starting...\r\n");

    err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);

    if (err.type != AI_ERROR_NONE)
    {
        print_ai_error("ai_network_create", err);
        return 0;
    }

    const ai_network_params params = {
        .params = AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
        .activations = AI_NETWORK_DATA_ACTIVATIONS(activations)
    };

    if (!ai_network_init(network, &params))
    {
        err = ai_network_get_error(network);
        print_ai_error("ai_network_init", err);
        return 0;
    }

    ai_buffer *input_info;
    ai_buffer *output_info;

    input_info = ai_network_inputs_get(network, NULL);
    output_info = ai_network_outputs_get(network, NULL);

    if (input_info == NULL || output_info == NULL)
    {
        printf("ai_network_inputs_get / outputs_get failed\r\n");
        return 0;
    }

    ai_input[0] = input_info[0];
    ai_output[0] = output_info[0];

    /* Input buffer X-CUBE-AI activation alanından gelsin */
    ai_input_data = (ai_u8 *)ai_input[0].data;

    if (ai_input_data == NULL)
    {
        printf("AI input buffer NULL!\r\n");
        return 0;
    }

    /* Output küçük, ayrı buffer olarak kalabilir */
    ai_output[0].data = AI_HANDLE_PTR(ai_output_data);

    printf("AI init OK\r\n");
    printf("Input bytes : %lu\r\n", (uint32_t)AI_NETWORK_IN_1_SIZE_BYTES);
    printf("Output bytes: %lu\r\n", (uint32_t)AI_NETWORK_OUT_1_SIZE_BYTES);
    printf("Class count : %lu\r\n", (uint32_t)CLASS_COUNT);
    return 1;
}

static int is_bin_file(const char* name)
{
    size_t len = strlen(name);

    if (len < 5)
        return 0;

    const char* ext = &name[len - 4];

    if ((ext[0] == '.') &&
        (ext[1] == 'B' || ext[1] == 'b') &&
        (ext[2] == 'I' || ext[2] == 'i') &&
        (ext[3] == 'N' || ext[3] == 'n'))
    {
        return 1;
    }

    return 0;
}

static int read_bin_to_input(const char* path)
{
    FIL file;
    FRESULT res;
    UINT bytes_read = 0;

    res = f_open(&file, path, FA_READ);

    if (res != FR_OK)
    {
        printf("f_open failed: %s FRESULT=%d\r\n", path, res);
        return 0;
    }

    FSIZE_t file_size = f_size(&file);

    if ((uint32_t)file_size != (uint32_t)AI_NETWORK_IN_1_SIZE_BYTES)
    {
        printf("Wrong file size: %s size=%lu expected=%lu\r\n",
               path,
               (uint32_t)file_size,
               (uint32_t)AI_NETWORK_IN_1_SIZE_BYTES);

        f_close(&file);
        return 0;
    }

    res = f_read(&file, ai_input_data, AI_NETWORK_IN_1_SIZE_BYTES, &bytes_read);
    f_close(&file);

    if (res != FR_OK || bytes_read != AI_NETWORK_IN_1_SIZE_BYTES)
    {
        printf("f_read failed: %s FRESULT=%d read=%lu expected=%lu\r\n",
               path,
               res,
               (uint32_t)bytes_read,
               (uint32_t)AI_NETWORK_IN_1_SIZE_BYTES);

        return 0;
    }

    return 1;
}

static uint32_t argmax_u8(const ai_u8* data, uint32_t len)
{
    uint32_t max_idx = 0;
    ai_u8 max_val = data[0];

    for (uint32_t i = 1; i < len; i++)
    {
        if (data[i] > max_val)
        {
            max_val = data[i];
            max_idx = i;
        }
    }

    return max_idx;
}

static int run_one_bin(const char* path, uint32_t true_idx, uint32_t* pred_idx, uint8_t* pred_raw)
{
    (void)true_idx;

    ai_i32 batch;

    if (!read_bin_to_input(path))
        return 0;

    memset(ai_output_data, 0, sizeof(ai_output_data));

    batch = ai_network_run(network, ai_input, ai_output);

    if (batch != 1)
    {
        ai_error err = ai_network_get_error(network);
        print_ai_error("ai_network_run", err);
        return 0;
    }

    uint32_t pred = argmax_u8(ai_output_data, AI_NETWORK_OUT_1_SIZE_BYTES);

    *pred_idx = pred;
    *pred_raw = ai_output_data[pred];

    return 1;
}



static void run_all_sd_tests(void)
{
    DIR dir;
    FILINFO fno;
    FRESULT res;

    uint32_t total = 0;
    uint32_t correct = 0;
    uint32_t failed_open_dir = 0;
    uint32_t failed_files = 0;

    uint32_t class_total[MAX_CLASSES] = {0};
    uint32_t class_correct[MAX_CLASSES] = {0};

    memset(confusion, 0, sizeof(confusion));

    printf("\r\n====================================\r\n");
    printf("        SD TEST START\r\n");
    printf("====================================\r\n");

    for (uint32_t class_idx = 0; class_idx < CLASS_COUNT; class_idx++)
    {
        char class_dir[32];

        snprintf(class_dir, sizeof(class_dir), "%s/%02lu", TEST_ROOT, class_idx);

        res = f_opendir(&dir, class_dir);

        if (res != FR_OK)
        {
            printf("DIR OPEN FAILED: %s FRESULT=%d\r\n", class_dir, res);
            failed_open_dir++;
            continue;
        }
        else
        {
            printf("DIR OPEN OK: %s\r\n", class_dir);
        }

        while (1)
        {
            res = f_readdir(&dir, &fno);

            if (res != FR_OK)
            {
                failed_files++;
                break;
            }

            if (fno.fname[0] == 0)
                break;

            if (fno.fattrib & AM_DIR)
                continue;
            printf("Found file in %s: %s\r\n", class_dir, fno.fname);
            if (!is_bin_file(fno.fname))
                continue;

            char file_path[PATH_BUF_SIZE];

            snprintf(file_path, sizeof(file_path), "%s/%s", class_dir, fno.fname);

            uint32_t pred = 0;
            uint8_t pred_raw = 0;

            if (run_one_bin(file_path, class_idx, &pred, &pred_raw))
            {
                total++;
                class_total[class_idx]++;

                if (pred == class_idx)
                {
                    correct++;
                    class_correct[class_idx]++;
                }

                if (class_idx < MAX_CLASSES && pred < MAX_CLASSES)
                {
                    confusion[class_idx][pred]++;
                }
            }
            else
            {
                failed_files++;
            }
        }

        f_closedir(&dir);
    }

    printf("\r\n====================================\r\n");
    printf("        SD TEST FINISHED\r\n");
    printf("====================================\r\n");

    printf("Total files tested : %lu\r\n", total);
    printf("Correct            : %lu / %lu\r\n", correct, total);

    if (total > 0)
    {
        uint32_t acc_x100 = (correct * 10000UL) / total;

        printf("Accuracy           : %lu.%02lu %%\r\n",
               acc_x100 / 100,
               acc_x100 % 100);
    }

    printf("Failed dirs        : %lu\r\n", failed_open_dir);
    printf("Failed files       : %lu\r\n", failed_files);

    printf("\r\nPer-class results:\r\n");

    for (uint32_t i = 0; i < CLASS_COUNT && i < MAX_CLASSES; i++)
    {
        printf("%02lu %-20s : %lu / %lu",
               i,
               CLASS_NAMES[i],
               class_correct[i],
               class_total[i]);

        if (class_total[i] > 0)
        {
            uint32_t class_acc_x100 = (class_correct[i] * 10000UL) / class_total[i];

            printf("  acc=%lu.%02lu %%",
                   class_acc_x100 / 100,
                   class_acc_x100 % 100);
        }

        printf("\r\n");
    }

    printf("\r\nConfusion matrix\r\n");
    printf("Rows=true class, Cols=pred class\r\n\r\n");

    printf("      ");

    for (uint32_t j = 0; j < CLASS_COUNT && j < MAX_CLASSES; j++)
    {
        printf("%02lu ", j);
    }

    printf("\r\n");

    for (uint32_t i = 0; i < CLASS_COUNT && i < MAX_CLASSES; i++)
    {
        printf("%02lu:   ", i);

        for (uint32_t j = 0; j < CLASS_COUNT && j < MAX_CLASSES; j++)
        {
            printf("%lu ", confusion[i][j]);
        }

        printf(" | %s\r\n", CLASS_NAMES[i]);
    }

    printf("\r\nClass index list:\r\n");

    for (uint32_t i = 0; i < CLASS_COUNT; i++)
    {
        printf("%02lu -> %s\r\n", i, CLASS_NAMES[i]);
    }

    printf("\r\nTest done.\r\n");
}

void MX_X_CUBE_AI_Init(void)
{
    printf("\r\nMX_X_CUBE_AI_Init\r\n");

    if (!sd_mount_local())
    {
        ai_ready = 0;
        return;
    }

    ai_ready = ai_local_init();

    if (!ai_ready)
    {
        printf("AI init failed.\r\n");
        return;
    }

    printf("System ready.\r\n");
}

void MX_X_CUBE_AI_Process(void)
{
    if (!ai_ready)
        return;

    if (test_done)
        return;

    test_done = 1;

    run_all_sd_tests();
}
