import pytest
import torch

from motionbricks.data.terrain_training_execution import AsyncCheckpointWriter,validate_execution_migration


def test_execution_migration_allows_resource_changes_but_rejects_learning_changes():
    old={'global_batch':32,'world_size':2,'steps':45000,'feedback_depth':13,'code_sha256':{'model.py':'fixed','scripts/train_joint_terrain_v3.py':'old'}}
    new={**old,'global_batch':896,'world_size':7,'code_sha256':{'model.py':'fixed','scripts/train_joint_terrain_v3.py':'new'}}
    assert 'global_batch' in validate_execution_migration(old,new)
    with pytest.raises(ValueError,match='learning contract'):validate_execution_migration(old,{**new,'feedback_depth':2})
    with pytest.raises(ValueError,match='model/data'):validate_execution_migration(old,{**new,'code_sha256':{'model.py':'changed'}})


def test_async_checkpoint_is_a_snapshot_and_replaces_only_complete_files(tmp_path):
    path=tmp_path/'last.pt';value=torch.tensor([2.])
    writer=AsyncCheckpointWriter();writer.submit(path,{'value':value,'optimizer':{'step':torch.tensor(4000.)}})
    value.add_(9);writer.close()
    saved=torch.load(path,weights_only=True)
    assert saved['value'].item()==2 and saved['optimizer']['step']==4000
    assert not path.with_suffix('.tmp').exists()


def test_streamed_sampler_matches_original_global_draws_across_chunks_and_resume():
    from scripts.train_joint_terrain_v3 import ReplayableSampler
    weights=torch.tensor([.05,.1,.2,.4,.25],dtype=torch.double)
    count=150*1024+971
    reference=torch.multinomial(weights,count,replacement=True,generator=torch.Generator().manual_seed(2333))
    for rank in (0,3,6):
        sampler=ReplayableSampler(weights,150,1024,rank,7,61,draw_offset=971)
        actual=list(sampler);indices=list(range(971+61*1024+rank,count,7))
        assert len(sampler)==len(indices)==len(actual)
        assert [i for _,i in actual]==indices
        assert [v for v,_ in actual]==reference[indices].tolist()
