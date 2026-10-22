"""Continue the verified small source set to a bounded1000 total optimizer updates."""

import json
from pathlib import Path
import subprocess
import sys


def main():
    task = Path.cwd().parent
    root = task/'joint_sdf_20261012'
    output = root/'reciprocal_source_coordinated_1000_v1'
    state = task/'joint-sdf-source-overfit-1000-state.json'
    if output.exists():
        raise ValueError('Do not duplicate a continuation')
    def status(value):
        state.write_text(json.dumps(dict(status=value, output=str(output), quality_approved=False), indent=2))
    try:
        status('training')
        subprocess.run([sys.executable,'-u','-m','scripts.pilot_joint_sdf_reciprocal_source',
            '--arm','coordinated','--updates','800','--checkpoint',str(root/'factual_prefix_aux_v1/weights.pt'),
            '--resume-from',str(root/'reciprocal_source_coordinated_v1/weights.pt'),
            '--cases',str(root/'context_pilot_data_v3/cases.pt'),
            '--accepted-recording',str(root/'reciprocal_source_coordinated_v1/final/generated_8030ae7a2a7ad640.npz'),
            '--output',str(output)],check=True)
        status('conditional_fit_review')
        subprocess.run([sys.executable,'-u','-m','scripts.review_joint_sdf_source_fit',
            '--checkpoint',str(output/'weights.pt'),'--cases',str(root/'context_pilot_data_v3/cases.pt'),
            '--output',str(output/'source_fit')],check=True)
        status('long_route_review')
        subprocess.run([sys.executable,'-u','-m','scripts.review_joint_sdf_natural_stair_time',
            '--runs','overfit='+str(output/'weights.pt'),'--cases',str(root/'context_pilot_data_v3/cases.pt'),
            '--output',str(output/'long_review')],check=True)
        status('complete')
    except Exception:
        status('failed')
        raise


if __name__=='__main__':
    main()
