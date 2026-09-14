"""Isolated NDP sixth experiment; upstream source and checkpoint structure unchanged."""
import os, sys, json, time, math, traceback
from pathlib import Path
ROOT = Path('/root/autodl-tmp/NDP')
HERE = Path(__file__).resolve().parent
MODE = sys.argv.pop(1)
assert MODE in ('preflight', 'formal')
os.chdir(ROOT / 'code/ndp')
sys.path.insert(0, str(ROOT / 'code/ndp'))
import torch
import hydra
import main_panoptic
from pytorch_lightning import Callback
from omegaconf import open_dict
from loss_correction import group_factors, correct_criterion
from bn_policy import apply_group, momentum_for_group

class CorrectedModel(main_panoptic.PanopticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        self._group_size = 4
        self._bn_bases = [(name, module, float(module.momentum))
                          for name,module in self.model.backbone.named_modules()
                          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        assert len(self._bn_bases)==62
        assert sum(original==0.02 for _,_,original in self._bn_bases)==16
        assert sum(original==0.1 for _,_,original in self._bn_bases)==46
        apply_group(self._bn_bases,4)
        self._parts = {}
        self._collect = False
        self.criterion.register_forward_hook(self._criterion_hook)
        for name, module in [('loss_id', self.id_loss), ('loss_ood', self.pixel_loss),
                             ('loss_soe', self.SOE_loss)]:
            module.register_forward_hook(self._scalar_hook(name))

    def _scalar_hook(self, name):
        def hook(module, inputs, output):
            if self._collect:
                self._parts[name] = float(output.detach())
        return hook

    def _criterion_hook(self, module, inputs, output):
        fixed = correct_criterion(output, self._group_size)
        if self._collect:
            for key, value in fixed.items():
                if key in module.weight_dict:
                    family = next((p for p in ('loss_ce', 'loss_mask', 'loss_dice', 'loss_box')
                                   if key == p or key.startswith(p + '_')), key)
                    self._parts[family] = self._parts.get(family, 0.) + float(value.detach()) * module.weight_dict[key]
        return fixed

    def training_step(self, batch, batch_idx):
        count = int(self.trainer.num_training_batches)
        self._group_size, tail_scale = group_factors(batch_idx, count, 4)
        apply_group(self._bn_bases, self._group_size)
        if batch_idx==count-1:
            print('BN_TAIL '+json.dumps(dict(group_size=self._group_size,
                  momenta=sorted(set(m.momentum for _,m,_ in self._bn_bases)))),flush=True)
        self._collect = MODE == 'preflight' or batch_idx % 10 == 0 or batch_idx == count - 1
        self._parts = {}
        loss = super().training_step(batch, batch_idx) * tail_scale
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Nonfinite total loss')
        if self._collect:
            record = dict(epoch=int(self.current_epoch)+1, batch=batch_idx+1,
                          batches=count, global_step=int(self.global_step),
                          group_size=self._group_size, tail_scale=tail_scale,
                          total_loss=float(loss.detach()),
                          parts={k:v*tail_scale for k,v in self._parts.items()})
            print('LOSS_PARTS '+json.dumps(record), flush=True)
            self.log_dict({'corrected/'+k:v for k,v in record['parts'].items()},
                          on_step=True, on_epoch=False, batch_size=len(batch[1]))
        return loss

class Audit(Callback):
    def __init__(self):
        self.started = time.time()
        self.steps = 0
        self.batches = 0
        self.gradient_checks = 0
    def on_train_start(self, trainer, pl_module):
        assert trainer.world_size == 1 and str(trainer.precision) == '32'
        assert trainer.accumulate_grad_batches == 4
        assert trainer.num_training_batches == 449
        assert trainer.lr_scheduler_configs[0].scheduler.total_steps == 1130
        bn_settings=[dict(name=n,original=o,applied=m.momentum,track_running_stats=m.track_running_stats)
                     for n,m,o in pl_module._bn_bases]
        assert all(x['track_running_stats'] and abs(x['applied']-momentum_for_group(x['original'],4))<1e-12 for x in bn_settings)
        with (Path(pl_module.config.general.save_dir).parent/'bn-settings.json').open('x') as stream:
            json.dump(bn_settings,stream,indent=2)
        print('BN_SETTINGS '+json.dumps(dict(count=len(bn_settings),momenta=sorted(set(x['applied'] for x in bn_settings)))),flush=True)
        torch.cuda.reset_peak_memory_stats()
        trainer.optimizers[0].register_step_post_hook(self._post_step)
        print('TRAIN_SETTINGS '+json.dumps(dict(mode=MODE, physical_batch=2, accumulation=4,
              epochs=10, batches_per_epoch=449, total_steps=1130, precision=str(trainer.precision),
              gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), loss_correction='sum_by_group_then_tail_compensation')),flush=True)
    def _post_step(self, optimizer, args, kwargs):
        self.steps += 1
    def on_after_backward(self, trainer, pl_module):
        if MODE == 'preflight':
            for p in pl_module.parameters():
                if p.grad is not None:
                    self.gradient_checks += 1
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Nonfinite parameter gradient')
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches += 1
        if self.batches % 10 == 0 or batch_idx == 448:
            print('Epoch [%d/10] Iter [%d/449] global_step=%d total_steps=1130 loss=%.6f lr=%.9g peak_allocated_gib=%.4f' %
                  (trainer.current_epoch+1,batch_idx+1,trainer.global_step,float(outputs['loss']),
                   trainer.optimizers[0].param_groups[0]['lr'],torch.cuda.max_memory_allocated()/2**30),flush=True)

main_panoptic.PanopticSegmentation = CorrectedModel
OriginalTrainer = main_panoptic.Trainer
audit = Audit()
class AuditedTrainer(OriginalTrainer):
    def __init__(self, *args, **kwargs):
        kwargs['callbacks'] = list(kwargs.get('callbacks', [])) + [audit]
        super().__init__(*args, **kwargs)
main_panoptic.Trainer = AuditedTrainer

@hydra.main(config_path=str(HERE), config_name='baseline')
def entry(cfg):
    assert cfg.data.batch_size == 2 and cfg.trainer.accumulate_grad_batches == 4
    assert cfg.optimizer.lr == 0.0002 and cfg.general.seed == 2025
    assert cfg.general.ckpt_path is None and not Path(cfg.general.save_dir).exists()
    with open_dict(cfg.general):
        cfg.general.loss_reduction_correction = 'sum_terms_group_size; total_tail_4/group_size; v1'
        cfg.general.bn_running_stat_policy = 'backbone: m=1-(1-original_m)**(1/actual_group_size); 62 BNs; v1'
    error = None
    try:
        main_panoptic.train(cfg)
        if MODE == 'preflight':
            assert audit.steps == 10 and audit.batches == 40
        else:
            assert audit.steps == 1130 and audit.batches == 4490
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        result=dict(mode=MODE,success=error is None,error=error,optimizer_steps=audit.steps,
                    training_batches=audit.batches,gradient_checks=audit.gradient_checks,
                    elapsed_seconds=time.time()-audit.started,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        path=Path(cfg.general.save_dir).parent/(MODE+'-result.json')
        with path.open('x') as stream: json.dump(result,stream,indent=2)
        print('FINAL_RESULT '+json.dumps(result),flush=True)

if __name__ == '__main__':
    entry()
