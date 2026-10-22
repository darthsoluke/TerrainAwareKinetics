from collections import Counter

from scripts.train_unified_native_routes import balanced_schedule


def test_every_clip_window_is_visited_without_global_modulo_aliasing():
    prepared = []
    for name, entry, count in [('up', 'up', 11), ('down', 'down', 12),
                                *[(f'flat{i}', 'flat', 13) for i in range(6)]]:
        for start in range(count):
            prepared.append(dict(window=dict(label=name, start=start), case=dict(entry=entry)))
    schedule = balanced_schedule(prepared, 1000)
    assert schedule == balanced_schedule(prepared, 1000)
    seen = Counter(index for batch in schedule for index in batch)
    assert set(seen) == set(range(len(prepared)))
    for batch in schedule:
        assert [prepared[i]['case']['entry'] for i in batch] == ['up', 'down', 'flat', 'flat']
    for label in {p['window']['label'] for p in prepared}:
        counts = [seen[i] for i, p in enumerate(prepared) if p['window']['label'] == label]
        assert max(counts) - min(counts) <= 1
