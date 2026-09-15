"""Finite frozen inference: smoke, ID-only prototype fit, full prediction export."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, Subset
from prototype_module import PrototypeAccumulator, Moments, prototype_scores, fuse

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'v06_baseline'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


class TrainingView(Dataset):
    """Deterministic inference view of original ID training frames, with raw GT."""
    def __init__(self, scans, lut, add_distance):
        self.scans, self.lut, self.add_distance = scans, lut, add_distance

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, i):
        scan = self.scans[i]
        pts = np.fromfile(scan['filepath'], np.float32).reshape(-1, 4)
        raw = np.fromfile(scan['label_filepath'], np.uint32)
        if len(raw) != len(pts):
            raise ValueError('Raw training point/label mismatch')
        pose = np.asarray(scan['pose']).T
        xyz = pts[:, :3] @ pose[:3, :3] + pose[3, :3]
        features = np.hstack((np.zeros((len(pts), 1)), pts[:, 3:4]))
        if self.add_distance:
            features = np.hstack((features, np.linalg.norm(xyz-xyz.mean(0), axis=1)[:, None]))
        # Actual raw labels are returned separately, never fed as target to the network.
        return ({'coordinates': xyz, 'features': np.hstack((xyz, features)),
                 'labels': np.zeros((len(pts), 2), np.int32), 'num_points': [0, len(pts)],
                 'sequence': (str(scan['scene']), Path(scan['filepath']).stem)}, self.lut[raw & 65535])


class TrainingCollate:
    def __init__(self, original):
        self.original = original

    def __call__(self, records):
        if len(records) != 1:
            raise ValueError('Frozen evaluation requires batch1')
        return self.original([records[0][0]]), records[0][1]


class Engine:
    def __init__(self, config, output):
        self.cfg, self.root = config, output
        self.start = time.time()
        source = BASE/'source_v06'
        os.chdir(source); sys.path.insert(0, str(source))
        import hydra
        import MinkowskiEngine as ME
        from omegaconf import OmegaConf
        from trainer.pq_trainer import PanopticSegmentation
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed']); torch.set_num_threads(config['cpu_threads'])
        self.ME = ME
        conf = OmegaConf.load(BASE/'config_v06_baseline.yaml')
        conf.general.mode = 'test'; conf.general.save_dir = str(output)
        self.conf = conf
        self.model = PanopticSegmentation(conf)
        ckpt = torch.load(config['checkpoint'], map_location='cpu')
        self.model.load_state_dict(ckpt['state_dict'], strict=True)
        del ckpt
        self.model.requires_grad_(False); self.model.cuda().eval()
        self.captured = {}
        self.model.model.point_features_head.register_forward_hook(self.feature_hook)
        self.collate = hydra.utils.instantiate(conf.data.test_collation)
        self.test = hydra.utils.instantiate(conf.data.test_dataset)
        if len(self.test) != config['expected_evaluation_frames']:
            raise ValueError('Unexpected evaluation split')
        mapping = yaml.safe_load((source/'conf/mask4former3d.yaml').read_text())['learning_map']
        self.lut = np.full(65536, -1, np.int64)
        for k, v in mapping.items():
            if 1 <= int(v) <= conf.data.num_labels:
                self.lut[int(k)] = int(v)-1
        self.scans = yaml.safe_load(Path(config['train_database']).read_text())
        if len(self.scans) != config['expected_training_frames'] or any(str(s['scene']) not in config['training_scenes'] for s in self.scans):
            raise ValueError('Unexpected ID training split')
        self.provenance = {'checkpoint_sha256': sha(config['checkpoint']),
                           'config_sha256': sha(config['_path']),
                           'training_database_sha256': sha(config['train_database'])}
        print('V08 spatial consistency: DISABLED', flush=True)
        print('V09 ID conditional calibration: DISABLED', flush=True)
        print('Prototype module: ENABLED', flush=True)
        print('MODEL_READY', json.dumps({'parameters': sum(p.numel() for p in self.model.parameters()),
              'trainable': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
              'feature_dim': 128, 'num_classes': conf.data.num_labels, 'gpu': torch.cuda.get_device_name(),
              'gpu_count_used': 1, 'batch_size': 1, 'evaluation_frames': len(self.test)}), flush=True)

    def feature_hook(self, module, inputs, output):
        self.captured['features'] = output.F.detach()
        self.captured['coords'] = output.C.detach()

    def loader(self, dataset, training=False):
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.cfg['workers'],
                          collate_fn=TrainingCollate(self.collate) if training else self.collate)

    @torch.inference_mode()
    def forward(self, batch, semantic=False):
        data, _ = batch
        sparse = self.ME.SparseTensor(coordinates=data.coordinates, features=data.features, device='cuda')
        out = self.model(sparse, raw_coordinates=data.raw_coordinates, is_eval=True)
        if not torch.equal(self.captured['coords'].cpu(), data.coordinates.cpu()):
            raise ValueError('Sparse coordinate alignment failed')
        f, s = self.captured['features'], out['pixel_ood'].detach()
        inv = torch.as_tensor(data.inverse_maps[0], device='cuda', dtype=torch.long)
        if f.shape != (len(s), 128) or not torch.isfinite(s).all():
            raise ValueError('Unexpected feature shape/nonfinite NDP')
        sem = None
        if semantic:
            logits = out['pred_logits'].softmax(-1)[0, :, :-1]
            sem = (out['pred_masks'][0].float().sigmoid() @ logits).argmax(-1)[inv].cpu().numpy().astype(np.uint8)
        return f, s, inv, data.sequences[0], sem

    def fit(self, directory, smoke=False):
        directory.mkdir(parents=True, exist_ok=False)
        indices = np.linspace(0, len(self.scans)-1, self.cfg['smoke_frames'], dtype=int) if smoke else range(len(self.scans))
        scans = [self.scans[i] for i in indices]
        manifest = [{'scan': s, 'point_sha256': sha(s['filepath']), 'label_sha256': sha(s['label_filepath'])} for s in scans]
        save(directory/'training_manifest.json', {'frames': manifest, 'split': 'train', 'ood_used': False, **self.provenance})
        dataset = TrainingView(scans, self.lut, self.conf.data.add_distance)
        accumulator = PrototypeAccumulator(self.conf.data.num_labels, 128, 'cuda')
        for i, (batch, labels) in enumerate(self.loader(dataset, True)):
            f, s, inv, key, _ = self.forward(batch)
            accumulator.update(f, inv, torch.as_tensor(labels, device='cuda', dtype=torch.long))
            if i < 3 or (i+1) % 50 == 0:
                print('PROTOTYPE_BUILD', i+1, '/', len(scans), 'elapsed', round(time.time()-self.start, 1), flush=True)
        bank = accumulator.finish()
        if not smoke and bank['class_counts'].tolist() != self.cfg['expected_id_counts']:
            raise ValueError('Full prototype class counts differ from raw GT audit')
        bank.update({'checkpoint_path': self.cfg['checkpoint'], 'dataset': self.cfg['train_database'],
                     'split': 'train', 'normalization': 'raw feature mean then L2; L2 queries for cosine',
                     'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'class_names': list(self.conf.data.class_names),
                     'frames': len(scans), 'smoke_only': smoke, **self.provenance})
        with (directory/'prototypes.pt').open('xb') as stream:
            torch.save(bank, stream)
        reloaded = torch.load(directory/'prototypes.pt', map_location='cpu')
        assert torch.equal(bank['prototypes'], reloaded['prototypes'])
        print('[Prototype Statistics]', flush=True)
        print('num_classes =', bank['num_classes'], 'feature_dim =', bank['feature_dim'], 'prototype_shape =', list(bank['prototypes'].shape), flush=True)
        for c, name in enumerate(bank['class_names']):
            print('class', c, name, 'points =', int(bank['class_counts'][c]), 'norm_before =', float(bank['norm_before'][c]),
                  'norm_after =', float(bank['norm_after'][c]), 'valid =', bool(bank['valid_mask'][c]), flush=True)
        moments = {'S_ndp': Moments(), 'S_proto': Moments()}
        for i, (batch, labels) in enumerate(self.loader(dataset, True)):
            f, s, inv, key, _ = self.forward(batch)
            p, _, _ = prototype_scores(f, bank)
            keep = torch.as_tensor(labels >= 0, device='cuda')
            moments['S_ndp'].update(s[inv][keep].cpu().numpy())
            moments['S_proto'].update(p[inv][keep].cpu().numpy())
            if i < 3 or (i+1) % 50 == 0:
                print('ID_NORMALIZATION', i+1, '/', len(scans), 'elapsed', round(time.time()-self.start, 1), flush=True)
        stats = {k: m.result() for k, m in moments.items()}
        if any(v['std'] <= 1e-12 for v in stats.values()):
            raise ValueError('Degenerate ID normalization')
        stats.update({'split': 'train', 'ood_used': False, 'prototypes_sha256': sha(directory/'prototypes.pt'), **self.provenance})
        save(directory/'normalization.json', stats)
        print('ID_STATISTICS', json.dumps(stats), flush=True)
        return bank, stats

    def export(self, directory, bank, stats, smoke=False):
        indices = np.linspace(0, len(self.test)-1, self.cfg['smoke_frames'], dtype=int).tolist()
        dataset = Subset(self.test, indices) if smoke else self.test
        predictions = directory/'prediction'; predictions.mkdir(exist_ok=False)
        rows = []
        for i, batch in enumerate(self.loader(dataset)):
            f, s, inv, key, sem = self.forward(batch, semantic=True)
            p, nearest, sim = prototype_scores(f, bank)
            ndp, proto = s[inv].cpu().numpy(), p[inv].cpu().numpy()
            cfg = self.cfg['prototype']
            final = fuse(ndp, proto, stats, cfg['alpha'], cfg['eps'])
            if not all(np.isfinite(x).all() for x in (ndp, proto, final)) or not bool(((sim >= -1) & (sim <= 1)).all()):
                raise ValueError('Nonfinite/out-of-range scores')
            if smoke or i in indices:
                old = np.loadtxt(Path(self.cfg['baseline_predictions'])/str(key[0])/(str(key[1])+'.txt')).astype(np.float32)
                error = float(np.abs(old-ndp).max())
                if not np.array_equal(old, ndp):
                    raise ValueError('Original Round6 score changed: '+str(key)+' max_error='+str(error))
                rows.append({'frame': list(key), 'points': len(ndp), 'feature_shape': list(f.shape),
                             'label_shape': [len(inv)], 'baseline_max_abs_error': error})
            dest = predictions/str(key[0])/(str(key[1])+'.npz'); dest.parent.mkdir(exist_ok=True)
            with dest.open('xb') as stream:
                np.savez_compressed(stream, S_ndp=ndp, S_proto=proto, S_final=final,
                                    nearest_proto_class=nearest[inv].cpu().numpy().astype(np.uint8),
                                    nearest_proto_similarity=sim[inv].cpu().numpy(), prototype_ood_score=proto,
                                    semantic_prediction=sem)
            if i < 3 or (i+1) % 50 == 0:
                print('INFERENCE', i+1, '/', len(dataset), 'elapsed', round(time.time()-self.start, 1), flush=True)
        if not smoke:
            expected = {str(Path(Path(p).parent.parent.name)/(Path(p).stem+'.npz')) for scene in self.test.scenes for p in scene}
            actual = {str(p.relative_to(predictions)) for p in predictions.rglob('*.npz')}
            if expected != actual:
                raise ValueError('Prediction coverage mismatch')
        result = {'frames': len(dataset), 'coverage_exact': True, 'identity_checks': rows,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                  'seconds': time.time()-self.start, 'gpu': torch.cuda.get_device_name(), 'gpu_count_used': 1,
                  'batch_size': 1, 'alpha': self.cfg['prototype']['alpha'], **self.provenance}
        save(directory/'inference_result.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['smoke', 'fit', 'infer'])
    p.add_argument('--config', type=Path, default=HERE/'config.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); cfg = json.loads(a.config.read_text()); cfg['_path'] = str(a.config.resolve())
    if not cfg['prototype']['enabled']:
        raise ValueError('This experiment requires Prototype enabled')
    root = a.output.resolve(); engine = Engine(cfg, root)
    if a.phase == 'smoke':
        bank, stats = engine.fit(root/'smoke', smoke=True)
        engine.export(root/'smoke', bank, stats, smoke=True)
    elif a.phase == 'fit':
        if not json.loads((root/'smoke'/'smoke_result.json').read_text())['passed']:
            raise ValueError('Smoke gate missing')
        engine.fit(root/'full')
    else:
        directory = root/'full'
        bank = torch.load(directory/'prototypes.pt', map_location='cpu')
        stats = json.loads((directory/'normalization.json').read_text())
        if any(stats[k] != v for k, v in engine.provenance.items()) or stats['prototypes_sha256'] != sha(directory/'prototypes.pt'):
            raise ValueError('Fit provenance mismatch')
        engine.export(directory, bank, stats)


if __name__ == '__main__':
    main()
