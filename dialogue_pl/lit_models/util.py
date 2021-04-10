import numpy as np 


def f1_eval(logits, labels):
    def getpred(result, T1 = 0.5, T2 = 0.4) :
        # 使用阈值得到preds, result = logits
        # T2 表示如果都低于T2 那么就是 no relation, 否则选取一个最大的
        ret = []
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [36]
                else:
                    r += [maxj]
            ret.append(r)
        return ret

    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0
        
        for i in range(len(data)):
            # 每一个样本 都是[1,4,...,20] 表示有1,4,20 是1， 如果没有就是[36]
            for id in data[i]:
                if id != 36:
                    # 标签中 1 的个数
                    correct_gt += 1
                    if id in devp[i]:
                        # 预测正确
                        correct_sys += 1

            for id in devp[i]:
                if id != 36:
                    all_sys += 1

        precision = 1 if all_sys == 0 else correct_sys/all_sys
        recall = 0 if correct_gt == 0 else correct_sys/correct_gt
        f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
        return f_1

    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))

    temp_labels = []
    for l in labels:
        t = []
        for i in range(36):
            if l[i] == 1:
                t += [i]
        if len(t) == 0:
            t = [36]
        temp_labels.append(t)
    assert(len(labels) == len(logits))
    labels = temp_labels
    
    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2/100.)
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2/100.

    return bestf_1, bestT2

def _get_dataloader(mode, label_list, args, tokenizer, processor):

    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    train_features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer)
    

    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []
    for f in train_features:
        input_ids.append([])
        input_mask.append([])
        segment_ids.append([])
        for i in range(n_class):
            input_ids[-1].append(f[i].input_ids)
            input_mask[-1].append(f[i].input_mask)
            segment_ids[-1].append(f[i].segment_ids)
        label_id.append([f[0].label_id])                

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(label_id, dtype=torch.float)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=6)

    return train_dataloader