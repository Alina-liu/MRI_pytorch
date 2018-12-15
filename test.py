import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import json
from utils import AverageMeter

def calculate_test_results(output_buffer,sample_id,test_results,labels):
    outputs =torch.stack(output_buffer)
    average_score = torch.mean(outputs,dim=0)
    sorted_scores,locs = torch.topk(average_score,k=1)
    results=[]
    for i in average_score.size(0):
        results.append({
            'label':labels[i],
            'score':sorted_scores[i]
        })
    test_results['results'][sample_id] = results

def test(data_loader, model, opt, labels):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    sample_id = ''
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        inputs = Variable(inputs, volatile=True)
        outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0):
                calculate_test_results(output_buffer,sample_id,test_results,output_buffer,labels)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            sample_id = labels[j]
        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}.json'.format(
                        opt.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)
    