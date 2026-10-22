"""Fit existing decoder head rows, then review only normal causal actor outputs."""

import json
from pathlib import Path
import subprocess
import sys


def main():
    task=Path.cwd().parent
    root=task/'joint_sdf_20261012'
    output=root/'render_head_fit_v1'
    state=task/'joint-sdf-render-head-fit-state.json'
    if output.exists():
        raise ValueError('Refusing to overwrite a completed or partial head fit')
    def status(value):
        state.write_text(json.dumps(dict(status=value,output=str(output),quality_approved=False),indent=2))
    try:
        status('fitting_and_review')
        subprocess.run([sys.executable,'-u',str(task/'fit_joint_sdf_render_head_v1.py'),
            '--checkpoint',str(root/'reciprocal_source_coordinated_1000_v1/weights.pt'),
            '--cases',str(root/'context_pilot_data_v3/cases.pt'),'--output',str(output)],check=True)
        for arm in ('root','rendered'):
            status(arm+'_conditional_and_route_review')
            subprocess.run([sys.executable,'-u',str(task/'review_joint_sdf_source_fit_logits_v2.py'),
                '--checkpoint',str(output/arm/'weights.pt'),'--cases',str(root/'context_pilot_data_v3/cases.pt'),
                '--output',str(output/arm/'source_fit')],check=True)
            subprocess.run([sys.executable,'-u','-m','scripts.review_joint_sdf_natural_stair_time',
                '--runs',arm+'='+str(output/arm/'weights.pt'),'--cases',str(root/'context_pilot_data_v3/cases.pt'),
                '--output',str(output/arm/'long_review')],check=True)
        status('complete')
    except Exception:
        status('failed')
        raise


if __name__=='__main__':
    main()
