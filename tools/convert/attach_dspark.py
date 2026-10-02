"""Add a fixed-window DSpark drafter to encoded target bytes without requantizing.

Linux: python -m tools.convert.attach_dspark --base model.ninfer --draft checkpoint/
       --draft-out scratch/draft.ninfer --out composed.ninfer
WSL mapped SMB drives can additionally use --windows-python /mnt/c/Python311/python.exe
for atomic Windows no-replace publication; Linux hardlinks are the default.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError, TensorObject, TensorSpec, ResourceSpec
from tools.artifact.writer import ArtifactWriter
from .model import Model
from .pipeline import convert
from .qwen3_5 import _Builder, draft_config
from .recipe import Recipe
from .sources.safetensors import SafetensorsSource


def build_draft(base, source):
    target = deepcopy(base.directory.components['text'])
    config = draft_config(source.config, target['config'], 'dspark')
    model = Model({'text': target, 'dspark': {'target': 'text', 'config': config}})
    for role, oid in target.get('resources', {}).items():
        model.resources[oid] = base.read_object(oid)
    _Builder(model).draft(source, config, target['config'], 'dspark')
    recipe = Recipe(model)
    projections = [p.name for p in model.parameters.values() if p.projection]
    recipe.assign(projections, format='q8_g32_fp16', method='grouped_absmax')
    return model, recipe


def remap(binding, names):
    result = deepcopy(binding)
    if 'object' in result:
        result['object'] = names[result['object']]
    else:
        for part in result['parts']:
            part['object'] = names[part['object']]
    return result


def compose_plan(base, donor):
    b, d = base.directory, donor.directory
    if 'dspark' in b.components or any(n.startswith('dspark/') for n in b.bindings):
        raise ArtifactError('base already contains DSpark')
    if b.components['text']['config'] != d.components['text']['config']:
        raise ArtifactError('target text configs differ')
    component = deepcopy(d.components['dspark'])
    if component['target'] != 'text':
        raise ArtifactError('DSpark must target the encoded text backbone')
    for role, oid in b.components['text'].get('resources', {}).items():
        other = d.components['text'].get('resources', {}).get(role)
        if other is None or base.read_object(oid) != donor.read_object(other):
            raise ArtifactError(f'target resource differs: {role}')
    imported = {n: value for n, value in d.bindings.items() if n.startswith('dspark/')}
    required = {oid for binding in imported.values() for oid in
                ([binding['object']] if 'object' in binding else [p['object'] for p in binding['parts']])}
    uses = [deepcopy(u) for u in d.uses if u['parameter'] in imported]
    for use in uses:
        for binding in use.get('auxiliaries', {}).values():
            required.update([binding['object']] if 'object' in binding else [p['object'] for p in binding['parts']])
    names = {oid: 'import/dspark/' + oid for oid in required}
    if set(names.values()) & set(base.by_id):
        raise ArtifactError('DSpark import object ID collision')
    objects = [obj for obj in donor.objects if obj.id in required]
    if len(objects) != len(required):
        raise ArtifactError('missing DSpark donor object')
    for use in uses:
        if 'auxiliaries' in use:
            use['auxiliaries'] = {k: remap(v, names) for k, v in use['auxiliaries'].items()}
    heads = [u for u in b.uses if u['parameter'] == 'text/output_head' and u['input'] == 'text/final_hidden']
    if len(heads) != 1:
        raise ArtifactError('base requires one unambiguous text output-head use')
    uses.append({**deepcopy(heads[0]), 'input': 'dspark/final_hidden'})
    copies = [(base, obj.id, obj.id) for obj in base.objects]
    copies.extend((donor, obj.id, names[obj.id]) for obj in objects)
    specs = [TensorSpec(new, a.by_id[old].shape, a.by_id[old].format, a.by_id[old].layout)
             if isinstance(a.by_id[old], TensorObject) else
             ResourceSpec(new, a.by_id[old].bytes, a.by_id[old].encoding)
             for a, old, new in copies]
    return specs, {**deepcopy(b.components), 'dspark': component}, {
        **deepcopy(b.bindings), **{n: remap(v, names) for n, v in imported.items()}}, [*deepcopy(list(b.uses)), *uses], copies


class WindowsPublishingWriter(ArtifactWriter):
    def __init__(self, *args, windows_python, **kwargs):
        self.windows_python = windows_python
        super().__init__(*args, **kwargs)

    def _publish_completed_file(self, source, target):
        paths = [subprocess.check_output(['wslpath', '-w', str(p.resolve())], text=True).strip()
                 for p in (source, target)]
        # On Windows os.rename is atomic within a filesystem and rejects any
        # existing destination. No shell, path interpolation, or overwrite option.
        subprocess.run([self.windows_python, '-c', 'import os,sys; os.rename(sys.argv[1],sys.argv[2])',
                        *paths], check=True)


def attach(args):
    start = time.monotonic()
    for path in (args.draft_out, args.out, Path(str(args.out) + '.attach.json')):
        if path.exists():
            raise FileExistsError(path)
    with Artifact(args.base) as base, SafetensorsSource(args.draft) as source:
        model, recipe = build_draft(base, source)
        import torch
        torch.set_num_threads(4)
        convert(model, recipe, args.draft_out, device='cpu', rows_per_chunk=128,
                provenance={'source': str(args.draft), 'source_config': deepcopy(source.config),
                            'policy': 'fixed K<=7; confidence adaptation disabled; Q8 drafter'},
                progress=lambda i, n, job: print(f'draft {i+1}/{n}: {job.spec.id}', flush=True))
        with Artifact(args.draft_out) as donor:
            specs, components, bindings, uses, copies = compose_plan(base, donor)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            required = sum(a.by_id[old].bytes for a, old, _ in copies) + 64 * 1024**2
            if shutil.disk_usage(args.out.parent).free < required:
                raise OSError('insufficient free space for composed artifact')
            options = dict(components=components, bindings=bindings, uses=uses,
                           metadata=deepcopy(base.directory.metadata),
                           provenance={**deepcopy(base.directory.provenance),
                               'dspark_import': {'base_artifact_id': base.artifact_id.hex(),
                                   'operation': 'encoded-byte-copy', 'draft': deepcopy(donor.directory.provenance)}})
            writer_class = ArtifactWriter
            if args.windows_python:
                writer_class = WindowsPublishingWriter
                options['windows_python'] = args.windows_python
            hashes = {}
            with writer_class(args.out, specs, **options) as writer:
                for index, (artifact, old, new) in enumerate(copies):
                    digest = hashlib.sha256(); offset = 0
                    for chunk in artifact.iter_object(old):
                        writer.write_region(new, offset, chunk)
                        digest.update(chunk); offset += len(chunk)
                    hashes[new] = digest.hexdigest()
                    if index % 100 == 0 or index + 1 == len(copies):
                        print(f'copy {index+1}/{len(copies)}: {new} ({offset} bytes)', flush=True)
            # Verify produced encoded bytes independently, including all target objects.
            with Artifact(args.out) as produced:
                for oid, expected in hashes.items():
                    digest = hashlib.sha256()
                    for chunk in produced.iter_object(oid): digest.update(chunk)
                    if digest.hexdigest() != expected:
                        raise ArtifactError(f'composed object changed: {oid}')
            report = {'output': str(args.out), 'seconds': time.monotonic()-start,
                      'target_encoded_bytes_unchanged': True, 'objects_sha256': hashes,
                      'draft_conversion': str(args.draft_out)+'.conversion.json'}
            Path(str(args.out)+'.attach.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'draft', 'draft-out', 'out'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--windows-python', help='WSL executable path for Windows SMB no-replace publication')
    print(json.dumps(attach(p.parse_args()), indent=2))


if __name__ == '__main__': main()
