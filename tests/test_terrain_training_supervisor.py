from scripts.run_terrain_control_50k import phase_command


def test_continuation_resumes_optimizer_without_reinitializing_model():
    command=['python','train.py','--initialize','seed.pt','--reuse-loss-calibration','--steps','45000']
    assert phase_command(command,['--resume','last.pt'])==[
        'python','train.py','--reuse-loss-calibration','--steps','45000','--resume','last.pt']
    assert phase_command(command,['--stop-at','5000'])==command+['--stop-at','5000']
    assert '--initialize' in command
