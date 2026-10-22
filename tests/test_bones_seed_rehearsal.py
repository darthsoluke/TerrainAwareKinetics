from collections import Counter
import torch
from motionbricks.data.bones_seed_rehearsal import bones_seed_controls,category_slots,frame_slot,CATEGORIES


def motion():
    value=torch.zeros(1,65,14)
    value[:,:,0]=torch.arange(65)*.05;value[:,:,4]=1
    return value


def test_bones_command_does_not_read_future_body_or_height():
    source=motion();history=source[:,:4].clone()
    first=bones_seed_controls(source[:,:33],'turn',source,history)
    source[:,:,1]=100;source[:,:,3:]=torch.randn_like(source[:,:,3:])
    second=bones_seed_controls(source[:,:33],'turn',source,history)
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    torch.testing.assert_close(first[:,:2],torch.tensor([[1.5,0.]]))


def test_bones_start_stop_and_balanced_real_windows():
    counts=Counter(c for step in range(200) for c in category_slots(step))
    assert counts==Counter({c:100 for c in CATEGORIES})
    assert [frame_slot(129,'stop',i,'clip') for i in range(3)]==[0,48,96]
    for frames in (65,129,290):
        assert all(0<=frame_slot(frames,c,i,'clip')<=frames-33 for c in CATEGORIES for i in range(30))
    source=motion()
    torch.testing.assert_close(bones_seed_controls(source[:,:33],'idle')[:,:2],torch.zeros(1,2))
    source[:,20:,0]=source[:,20,0,None]
    torch.testing.assert_close(bones_seed_controls(source[:,-33:],'stop',source)[:,:2],torch.zeros(1,2))
