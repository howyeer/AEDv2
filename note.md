gtbox [x0,y0,x1,y1]

<<<<<<< HEAD
'obj_ids': gt_track_ids
=======
>>>>>>> 949a4975e77cae7371b7e5c2bd319bdecffd8ea4

det_idxes = target['obj_ids'][is_det][..., None].float()
proposal = torch.cat([det_boxes, det_scores, det_labels, det_idxes], dim=1)     7个维度


```
track_instances.query_pos = pos2posemb(proposals[:, 4:5], d_model) + self.det_embed.weight
query_embed = track_instances.query_pos
tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
proposals = tgt[:, :num_proposals]
track_queries = tgt[:, num_proposals:]
```


平均forward_single_image time:0.15s, 平均get_groundingdino_output time:0.005s, 平均post_process_single_image time:0.05s losses.backward:0.15s
一个clip平均时间：1s
一个epoch平均时间：6h


报错：RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
原因：存在没有gt的clip，导致无法计算损失
解决：过滤掉每张frame都没有gt的clip
```
if not any([len(i) for i in data_dict['gt_instances']]):
            print('warning: no gt instances in this batch, skip it')
            continue
```
问题还存在：因为存在每个frame都没有匹配上的clip，导致无法计算损失

在dist.barrier()之前没经过all_reduce()，GPU依然卡在NCCL
