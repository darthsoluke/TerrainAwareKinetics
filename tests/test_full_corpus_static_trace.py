import torch

from motionbricks.data.full_corpus_static_trace import isolate_shape_constant_uses


def test_shape_scalar_copy_preserves_other_consumers():
    graph = torch._C.parse_ir('''graph(%size : int, %data : Tensor):
      %dimension : Tensor = prim::NumToTensor(%size)
      %shared : Tensor = prim::Constant()
      %shape : Tensor = aten::mul(%dimension, %shared)
      %motion : Tensor = aten::mul(%data, %shared)
      return (%shape, %motion)
    ''')
    nodes = list(graph.nodes())
    nodes[1].t_('value', torch.tensor(4))
    original = nodes[1].output()
    assert isolate_shape_constant_uses(graph) == 1
    assert list(nodes[2].inputs())[1] != original
    assert list(nodes[3].inputs())[1] == original
    torch.testing.assert_close(list(nodes[2].inputs())[1].toIValue(), original.toIValue(), atol=0, rtol=0)


def test_arrays_and_non_shape_scalar_uses_remain_unchanged():
    graph = torch._C.parse_ir('''graph(%data : Tensor):
      %shared : Tensor = prim::Constant()
      %motion : Tensor = aten::mul(%data, %shared)
      return (%motion)
    ''')
    list(graph.nodes())[0].t_('value', torch.tensor(4))
    before = str(graph)
    assert isolate_shape_constant_uses(graph) == 0
    assert str(graph) == before
