
def get_values(filename):
    data = {}
    key = ""
    #loss1, loss2, loss_result, r2_loss_result, acc, count, cpu_time, cuda_time, cpu_memory, cuda_memory
    with open(filename, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            values = line.split(',')
            if len(values) == 1:
                key = values[0]
                data[key] = []
                continue
            data[key].append(values)

    avg_data = {}
    for key, values in data.items():
        loss1 = 0
        loss2 = 0
        loss_eval = 0
        r2 = 0
        cuda_time = 0
        cuda_memory = 0
        ee_count = 0
        acc = 0
        length = len(values)
        acc_length = 0
        for value in values:
            loss1 += float(value[0])
            if value[1] != 'None':
                loss2 += float(value[1])
            else:
                loss2 += float(value[0])
            loss_eval += float(value[2])
            r2 += float(value[3])
            if value[4] != 'nan':
                acc += float(value[4])
                acc_length += 1
            ee_count += float(value[5])
            cuda_time += float(value[7])
            cuda_memory += float(value[9])
        loss1 /= length
        loss2 /= length
        loss_eval /= length
        cuda_time /= length
        cuda_memory /= length
        ee_count /= length
        r2 /= length
        if acc_length > 0:
            acc /= acc_length

        print(key, "Loss:", loss1, loss2, loss_eval, "R2:", r2, "Acc:", acc, ee_count, "CUDA:", cuda_time, cuda_memory)


if __name__ == "__main__":
    print('rnn_size_hidden')
    get_values("../network_test/data/rnn_size_hidden")
    print('ee_ranges')
    get_values("../network_test/data/ee_ranges")
    print('lr test')
    get_values("../network_test/data/end_device_lr_test")
