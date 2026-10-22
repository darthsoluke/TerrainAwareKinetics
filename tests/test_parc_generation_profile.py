import hashlib
import pytest
from scripts.parc_generation_profile import resolve_sampling_profile,AUDITED_LEGACY_GENERATOR


def legacy(tmp_path):
    path=tmp_path/'config.yaml'
    path.write_text('mdm_gen:\n  ddim_stride: 5\n  cfg_scale: 0.65\n')
    return dict(source_sha256={'generate_parc_long_stairs.py':AUDITED_LEGACY_GENERATOR},
        generation_config=str(path),generation_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ddim_stride=5,cfg_scale=.65)


def test_legacy_profile_requires_actual_sampler_and_hashed_config(tmp_path):
    report=legacy(tmp_path)
    profile,evidence=resolve_sampling_profile(report)
    assert profile=='official_data' and not evidence['raw_report_modified']
    assert 'sampling_profile' not in report
    report['ddim_stride']=10
    with pytest.raises(ValueError,match='Actual sampler'):resolve_sampling_profile(report)


def test_profile_does_not_guess_unknown_generator_or_changed_config(tmp_path):
    report=legacy(tmp_path)
    report['generation_config_sha256']='wrong'
    with pytest.raises(ValueError,match='configuration changed'):resolve_sampling_profile(report)
    report['source_sha256']={}
    with pytest.raises(ValueError,match='Unknown legacy'):resolve_sampling_profile(report)
