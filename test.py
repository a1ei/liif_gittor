import argparse
import os
import math
from functools import partial

import yaml
import jittor as jt
from tqdm import tqdm

import datasets
import models
import utils


def batched_predict(model, inp, coord, cell, bsize):
    with jt.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = jt.concat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    # inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    # inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    inp_sub = jt.array(t['sub']).float().reshape(1, -1, 1, 1)
    inp_div = jt.array(t['div']).float().reshape(1, -1, 1, 1)
    t = data_norm['gt']
    # gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    # gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    gt_sub = jt.array(t['sub']).float().reshape(1, 1, -1)
    gt_div = jt.array(t['div']).float().reshape(1, 1, -1)

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with jt.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp,
                batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/test/test-div2k-6.yaml")
    parser.add_argument('--model', default="save/test_0/epoch-best.pth")
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    jt.flags.use_cuda = 1

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    dataset.set_attrs(
        batch_size=spec['batch_size'],
        num_workers=8,
        shuffle=False
    )

    model = models.make({   #注意要根据自己的模型调整
        'name': 'liif',
        'args': {
            'encoder_spec': {
                'name': 'edsr-baseline',
                'args': {
                    'no_upsampling': True
                }
            },
            'imnet_spec': {
                'name': 'mlp',
                'args': {
                    'out_dim': 3,
                    'hidden_list': [256, 256, 256, 256]
                }
            }
        }
    })
    state_dict = jt.load(args.model)['model']
    model.load_state_dict(state_dict)

    res = eval_psnr(dataset, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
