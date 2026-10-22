"""Resolve the audited legacy generator's missing profile from immutable evidence."""

import hashlib
from pathlib import Path
import yaml

AUDITED_LEGACY_GENERATOR='4623d4146edff327512fc78f19159d74cbbca6e73ea7ab860b3790467b4428c1'


def resolve_sampling_profile(report):
    if report.get('sampling_profile'):
        return report['sampling_profile'], {'source':'explicit_generation_report'}
    if report.get('source_sha256',{}).get('generate_parc_long_stairs.py')!=AUDITED_LEGACY_GENERATOR:
        raise ValueError('Unknown legacy generator; do not guess the sampling profile')
    path=Path(report['generation_config'])
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=report['generation_config_sha256']:
        raise ValueError('Generation configuration changed')
    settings=yaml.safe_load(data)['mdm_gen']
    if (report['ddim_stride']!=settings['ddim_stride']
            or abs(report['cfg_scale']-settings['cfg_scale'])>1e-9):
        raise ValueError('Actual sampler differs from the official data profile')
    return 'official_data',dict(source='audited_legacy_generator_and_hashed_config',
        generator_sha256=AUDITED_LEGACY_GENERATOR,config_sha256=report['generation_config_sha256'],
        ddim_stride=report['ddim_stride'],cfg_scale=report['cfg_scale'],raw_report_modified=False)
