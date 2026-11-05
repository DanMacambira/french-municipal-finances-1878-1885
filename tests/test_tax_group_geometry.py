from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from tax_group_geometry import associate_tax_lines


def _names():
    return [{'id': 'a', 'y_center': 100}, {'id': 'b', 'y_center': 200},
            {'id': 'c', 'y_center': 300}, {'id': 'd', 'y_center': 400}]


def _taxes():
    return [{'id': 't1', 'y_center': 100},
            {'id': 't2', 'y_center': 180}, {'id': 't3', 'y_center': 200},
            {'id': 't4', 'y_center': 220}, {'id': 't5', 'y_center': 300},
            {'id': 't6', 'y_center': 400}]


def _index(results):
    return {str(r['tax_id']): r for r in results}


def test_curly_brace_groups_multiple_entries_and_straight_rule_does_not():
    marks = [{'kind': 'straight_or_fragment_candidate', 'y_min': 80, 'y_max': 420},
             {'kind': 'curly_brace_candidate', 'y_min': 176, 'y_max': 225}]
    result = _index(associate_tax_lines(_names(), _taxes(), marks))
    for tid in ('t2', 't3', 't4'):
        assert result[tid]['commune_id'] == 'b'
        assert result[tid]['status'] == 'proposed_brace_group'
    for tid, name in [('t1', 'a'), ('t5', 'c'), ('t6', 'd')]:
        assert result[tid]['commune_id'] == name
        assert result[tid]['status'] == 'proposed_single_alignment'


def test_missing_brace_does_not_invent_multiple_entries():
    result = _index(associate_tax_lines(_names(), _taxes(),
            [{'kind': 'straight_or_fragment_candidate', 'y_min': 80, 'y_max': 420}]))
    assert result['t3']['commune_id'] is None
    assert result['t2']['commune_id'] is None
    assert result['t4']['commune_id'] is None
    assert result['t1']['commune_id'] == 'a'


def test_brace_crossing_two_commune_names_is_ambiguous():
    names = [{'id': 'one', 'y_center': 100}, {'id': 'two', 'y_center': 122}]
    taxes = [{'id': 'x', 'y_center': 95}, {'id': 'y', 'y_center': 125}]
    brace = [{'kind': 'curly_brace_candidate', 'y_min': 90, 'y_max': 135}]
    result = _index(associate_tax_lines(names, taxes, brace))
    assert result['x']['commune_id'] is None
    assert result['y']['commune_id'] is None


def test_empty_and_duplicate_guards():
    assert associate_tax_lines(_names(), [], []) == []
    try:
        associate_tax_lines([{'id':'a','y_center':0}, {'id':'a','y_center':1}], _taxes(), [])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate names must be rejected')
