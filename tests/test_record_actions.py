from dataclasses import replace

import pytest

from welog.record_actions import delete_record, plan_deletion


def record(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('{"text":"模拟消息"}\n',encoding='utf-8')
    return path


def test_delete_managed_capture_removes_only_that_session(tmp_path):
    root=tmp_path/'runs'
    source=record(root/'pointer-test'/'capture'/'messages.jsonl')
    (source.parent/'messages.sqlite3').write_bytes(b'fixture')
    (source.parent.parent/'calibration.json').write_text('{}')
    sibling=record(root/'pointer-other'/'capture'/'messages.jsonl')
    settings=root/'export-settings.json'
    settings.write_text('{}')
    exported=tmp_path/'exports'/'result.json'
    exported.parent.mkdir()
    exported.write_text('[]')
    plan=plan_deletion(source,root)
    assert plan.directory==root/'pointer-test'
    delete_record(plan)
    assert not plan.directory.exists()
    assert sibling.exists() and settings.exists() and exported.exists() and root.exists()


def test_delete_nested_record_preserves_parent_and_siblings(tmp_path):
    root=tmp_path/'runs'
    source=record(root/'offline'/'first'/'messages.jsonl')
    sibling=record(root/'offline'/'second'/'messages.jsonl')
    plan=plan_deletion(source,root)
    assert plan.directory==source.parent
    delete_record(plan)
    assert sibling.exists()


def test_root_records_only_delete_message_files(tmp_path):
    source=record(tmp_path/'messages.jsonl')
    (tmp_path/'messages.sqlite3').write_bytes(b'fixture')
    keep=tmp_path/'unrelated.txt'
    keep.write_text('keep')
    plan=plan_deletion(source,tmp_path)
    assert plan.directory is None
    delete_record(plan)
    assert not source.exists() and not (tmp_path/'messages.sqlite3').exists()
    assert keep.exists()


def test_imported_file_deletion_never_removes_its_directory(tmp_path):
    source=record(tmp_path/'external'/'messages.jsonl')
    keep=tmp_path/'external'/'keep.txt'
    keep.write_text('keep')
    plan=plan_deletion(source,tmp_path/'runs')
    assert plan.directory is None and plan.files==(source,)
    delete_record(plan)
    assert not source.exists() and keep.exists()


def test_invalid_or_changed_deletion_plan_cannot_remove_root(tmp_path):
    source=record(tmp_path/'runs'/'one'/'messages.jsonl')
    plan=plan_deletion(source,tmp_path/'runs')
    with pytest.raises(ValueError):delete_record(replace(plan,directory=plan.root))
    assert source.exists()
    source.rename(source.with_name('renamed.jsonl'))
    with pytest.raises(ValueError):delete_record(plan)
    assert source.parent.exists()
