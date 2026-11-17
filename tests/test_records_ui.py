"""Offscreen workbench integration; all messages are synthetic, desktop calls forbidden."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['QT_QPA_PLATFORM']='offscreen'

import pytest
pytest.importorskip('PySide6')
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QInputDialog
from welog.record_dialogs import MessageEditDialog, RecordInfoDialog
from welog.gui import CaptureWindow, configure_font


def pump(app,condition,seconds=8):
    deadline=time.monotonic()+seconds
    while not condition() and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.005)
    app.processEvents()
    assert condition()


@pytest.fixture
def workbench(tmp_path):
    app=QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    configure_font(app)
    def forbidden(*args):raise AssertionError('Records UI must never touch real windows')
    window=CaptureWindow(SimpleNamespace(activate=forbidden,window_at_cursor=forbidden,describe_window=forbidden),tmp_path/'runs')
    window.show()
    app.processEvents()
    yield app,window
    pump(app,lambda:not window.records_panel.busy())
    window.hide()


def make_records(path,count=3):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as file:
        for index in range(count):
            file.write(json.dumps({'message_id':str(index),'direction':'incoming' if index%2==0 else 'outgoing',
                'type':'audio' if index==1 else 'text','duration_seconds':19 if index==1 else None,
                'text':"19''" if index==1 else f'消息 {index}\n完整正文','status':'complete'})+'\n')
    return path


def test_load_alias_preview_multiformat_export_and_saved_settings(workbench,tmp_path):
    app,window=workbench
    source=make_records(tmp_path/'old'/'messages.jsonl')
    before=source.read_bytes()
    window.show_records(source)
    panel=window.records_panel
    pump(app,lambda:not panel.busy())
    assert window.pages.currentIndex()==1
    assert panel.table.rowCount()==3
    assert panel.detail.toPlainText()=='消息 0\n完整正文'
    panel.incoming.setText('好友')
    panel.outgoing.setText('本人')
    assert panel.table.item(0,0).text()=='好友'
    assert panel.table.item(1,0).text()=='本人'
    panel.table.selectRow(1)
    assert '19 秒' in panel.detail_meta.text() and '语音' in panel.detail_meta.text()
    for key,box in panel.formats.items():box.setChecked(key in {'json','txt','html'})
    destination=tmp_path/'导出到这里'
    panel.directory.setText(str(destination))
    panel.title.setText('一次对话')
    QTest.mouseClick(panel.export_button,Qt.MouseButton.LeftButton)
    pump(app,lambda:not panel.busy())
    assert panel.last_export['count']==3
    assert {Path(file).suffix for file in panel.last_export['files']}=={'.json','.txt','.html'}
    result=json.loads(next(destination.glob('*.json')).read_text(encoding='utf-8'))
    assert result['messages'][0]['sender']=='好友' and result['messages'][1]['sender']=='本人'
    assert panel.open_directory.isEnabled()
    assert source.read_bytes()==before
    settings=json.loads(panel.settings_path.read_text(encoding='utf-8'))
    assert settings['incoming_name']=='好友' and settings['directory']==str(destination)
    assert not list(tmp_path.rglob('*.png'))
    assert all(box.height()>=28 and box.text() for box in panel.formats.values())


def test_pagination_search_and_export_always_include_all_messages(workbench,tmp_path):
    app,window=workbench
    window.show_records(make_records(tmp_path/'messages.jsonl',450))
    panel=window.records_panel
    pump(app,lambda:not panel.busy())
    assert panel.table.rowCount()==200 and panel.next.isEnabled() and not panel.previous.isEnabled()
    panel.next.click()
    pump(app,lambda:not panel.busy())
    assert panel.offset==200 and panel.rows[0]['text'].startswith('消息 200')
    panel.search.setText('消息 449')
    QTest.keyClick(panel.search,Qt.Key.Key_Return)
    pump(app,lambda:not panel.busy())
    assert panel.table.rowCount()==1 and panel.page['total']==450
    for key,box in panel.formats.items():box.setChecked(key=='jsonl')
    panel.directory.setText(str(tmp_path/'all'))
    panel.start_export()
    pump(app,lambda:not panel.busy())
    assert panel.last_export['count']==450


def test_open_file_and_directory_dialogs_use_selected_paths(workbench,tmp_path,monkeypatch):
    app,window=workbench
    source=make_records(tmp_path/'messages.jsonl')
    panel=window.records_panel
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args:(str(source),'消息记录'))
    monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *args:str(tmp_path/'chosen'))
    panel.import_records()
    pump(app,lambda:not panel.busy())
    panel.choose_directory()
    assert panel.source.path==source and panel.directory.text()==str(tmp_path/'chosen')
    for box in panel.formats.values():box.setChecked(False)
    assert not panel.export_button.isEnabled()
    assert not list(window.output_root.glob('*'))


def test_recent_sessions_and_bad_input_report_errors_without_writes(workbench,tmp_path):
    app,window=workbench
    path=make_records(window.output_root/'pointer-20261116-220506-aa19c7'/'capture'/'messages.jsonl')
    window.refresh_sessions()
    assert window.sessions.count()==1
    item=window.sessions.item(0)
    assert item.data(Qt.ItemDataRole.UserRole)==str(path)
    window.show_records(item.data(Qt.ItemDataRole.UserRole))
    pump(app,lambda:not window.records_panel.busy())
    window.show_records(tmp_path/'missing.json')
    pump(app,lambda:not window.records_panel.busy())
    assert '读取失败' in window.records_panel.feedback.text()
    assert not list(tmp_path.rglob('export-settings.json'))


def test_load_requests_only_apply_the_newest_source(workbench,tmp_path):
    app,window=workbench
    first=make_records(tmp_path/'first.jsonl',500)
    second=make_records(tmp_path/'second.jsonl',2)
    window.show_records(first)
    window.show_records(second)
    pump(app,lambda:not window.records_panel.busy())
    assert window.records_panel.source.path==second
    assert len(window.records_panel.rows)==2


def test_export_error_keeps_source_and_allows_retry(workbench,tmp_path):
    app,window=workbench
    source=make_records(tmp_path/'messages.jsonl')
    window.show_records(source)
    panel=window.records_panel
    pump(app,lambda:not panel.busy())
    panel.directory.setText(str(source))
    panel.start_export()
    pump(app,lambda:not panel.busy())
    assert '导出失败' in panel.feedback.text()
    assert panel.export_button.isEnabled() and panel.last_export is None
    assert panel.source.path==source


def test_compact_window_keeps_export_controls_available(workbench,tmp_path):
    app,window=workbench
    window.resize(1160,760)
    window.show_records(make_records(tmp_path/'messages.jsonl'))
    pump(app,lambda:not window.records_panel.busy())
    panel=window.records_panel
    assert all(box.height()>=28 for box in panel.formats.values())
    assert panel.export_button.height()>=30
    assert not panel.export_button.isHidden()


def test_delete_record_cancel_then_confirm_clears_view_and_refreshes_list(workbench,tmp_path,monkeypatch):
    app,window=workbench
    source=make_records(window.output_root/'pointer-test'/'capture'/'messages.jsonl')
    sibling=make_records(window.output_root/'pointer-keep'/'capture'/'messages.jsonl')
    report=source.parent/'report.html'
    report.write_text('fixture')
    window.show_records(source)
    pump(app,lambda:not window.records_panel.busy())
    window.refresh_sessions()
    window.report_path=str(report)
    window.report_button.setEnabled(True)
    panel=window.records_panel
    assert panel.delete_button.isEnabled()
    questions=[]
    def answer_no(*args):
        questions.append(args[2])
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox,'question',answer_no)
    panel.delete_button.click()
    assert source.exists() and panel.source is not None
    assert str(source.parent.parent) in questions[0]
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.StandardButton.Yes)
    panel.delete_button.click()
    assert not source.parent.parent.exists() and sibling.exists()
    assert panel.source is None and panel.table.rowCount()==0
    assert not panel.export_button.isEnabled() and not panel.delete_button.isEnabled()
    assert window.report_path is None and not window.report_button.isEnabled()
    assert window.sessions.count()==1 and '记录已删除' in panel.feedback.text()


def test_delete_while_loading_does_not_open_confirmation(workbench,tmp_path,monkeypatch):
    app,window=workbench
    source=make_records(tmp_path/'messages.jsonl',450)
    def forbidden(*args):raise AssertionError('Busy records cannot be deleted')
    monkeypatch.setattr(QMessageBox,'question',forbidden)
    window.show_records(source)
    assert not window.records_panel.delete_button.isEnabled()
    window.delete_saved_record(source)
    assert source.exists()
    pump(app,lambda:not window.records_panel.busy())


def test_full_mode_is_default_and_direction_explains_required_start(workbench):
    app,window=workbench
    assert window.capture_mode.currentData()=='all'
    assert not window.frame_limit.isEnabled() and window.frame_limit.isHidden()
    assert '最底部' in window.start_hint.text()
    window.direction.setCurrentIndex(1)
    assert '最早一条' in window.start_hint.text()
    window.capture_mode.setCurrentIndex(1)
    assert window.frame_limit.isEnabled() and not window.frame_limit.isHidden()


def test_rename_updates_sidebar_name_and_keeps_statistics(workbench,tmp_path,monkeypatch):
    app,window=workbench
    source=make_records(window.output_root/'pointer-test'/'capture'/'messages.jsonl')
    window.show_records(source)
    panel=window.records_panel
    pump(app,lambda:not panel.busy())
    original=source.read_bytes()
    monkeypatch.setattr(QInputDialog,'getText',lambda *args,**kwargs:('对方的名字',True))
    panel.rename_button.click()
    pump(app,lambda:not panel.busy())
    assert window.sessions.item(0).text()=='对方的名字'
    assert panel.source_label.text()=='对方的名字' and panel.title.text()=='对方的名字'
    assert panel.metadata['statistics']['total']==3 and panel.metadata['statistics']['type_counts']['audio']==1
    assert source.read_bytes()==original
    info=RecordInfoDialog(panel.metadata)
    assert '总消息数：3' in info.details.toPlainText() and '语音：1' in info.details.toPlainText()
    info.hide()


def test_edit_message_and_restore_keep_selected_message(workbench,tmp_path,monkeypatch):
    app,window=workbench
    source=make_records(tmp_path/'messages.jsonl')
    window.show_records(source)
    panel=window.records_panel
    pump(app,lambda:not panel.busy())
    original=source.read_bytes()
    panel.table.selectRow(2)
    def accept_edit(dialog):
        dialog.text.setPlainText('人工修正内容')
        dialog.kind.setCurrentIndex(dialog.kind.findData('file'))
        dialog.direction.setCurrentIndex(dialog.direction.findData('outgoing'))
        return dialog.DialogCode.Accepted
    monkeypatch.setattr(MessageEditDialog,'exec',accept_edit)
    panel.edit_button.click()
    pump(app,lambda:not panel.busy())
    assert panel.table.currentRow()==2 and panel.detail.toPlainText()=='人工修正内容'
    assert panel.rows[2]['type']=='file' and panel.rows[2]['direction']=='outgoing' and panel.rows[2]['edited']
    assert panel.reset_button.isEnabled()
    panel.reset_button.click()
    pump(app,lambda:not panel.busy())
    assert panel.table.currentRow()==2 and panel.detail.toPlainText()=='消息 2\n完整正文'
    assert not panel.rows[2].get('edited') and source.read_bytes()==original


def test_image_attachment_can_be_replaced_previewed_and_exported(workbench,tmp_path,monkeypatch):
    import cv2
    import numpy as np
    app,window=workbench
    source=make_records(tmp_path/'messages.jsonl')
    window.show_records(source)
    panel=window.records_panel
    pump(app,lambda:not panel.busy())
    picture=tmp_path/'photo.png'
    ok,data=cv2.imencode('.png',np.full((30,40,3),(100,160,230),np.uint8))
    assert ok
    picture.write_bytes(data.tobytes())
    monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args:(str(picture),'图片'))
    panel.attach_button.click()
    pump(app,lambda:not panel.busy())
    assert panel.rows[0]['type']=='image' and panel.rows[0]['attachment']['name']=='photo.png'
    assert not panel.attachment_preview.isHidden() and panel.open_attachment_button.isEnabled()
    panel.directory.setText(str(tmp_path/'out'))
    for key,box in panel.formats.items():box.setChecked(key=='html')
    panel.start_export()
    pump(app,lambda:not panel.busy())
    assert panel.last_export['attachments']==1
    html=Path(panel.last_export['files'][0]).read_text(encoding='utf-8')
    assert '<img src=' in html


def test_time_group_filter_restricts_preview_but_exports_all(workbench,tmp_path):
    app,window=workbench
    source=make_records(tmp_path/'messages.jsonl')
    rows=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    for index,row in enumerate(rows):
        row['time_interval']={'start':'2025-12-02T12:41+08:00' if index==0 else '2025-12-02T14:22+08:00',
                              'end':'2025-12-02T14:22+08:00' if index==0 else None,'timezone':'UTC+08:00','status':'bounded' if index==0 else 'open_end'}
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    window.show_records(source)
    panel=window.records_panel
    pump(app,lambda:not panel.busy())
    assert panel.time_filter.count()==3
    panel.time_filter.setCurrentIndex(1)
    pump(app,lambda:not panel.busy())
    assert len(panel.rows)==1 and panel.page['total']==3
    assert '12:41' in panel.detail_meta.text() and '14:22' in panel.detail_meta.text()
    panel.directory.setText(str(tmp_path/'out'))
    panel.start_export()
    pump(app,lambda:not panel.busy())
    assert panel.last_export['count']==3


def test_legacy_file_card_updates_information_and_table_counts(workbench,tmp_path):
    app,window=workbench
    path=tmp_path/'old.jsonl'
    rows=[{'message_id':'file','direction':'incoming','type':'text','text':'第十组.pptx\nP\n30.7M 未下载\n微信电脑版','status':'complete'},
          {'message_id':'mention','direction':'incoming','type':'text','text':'[文件]汇报ppt.pptx','status':'complete'}]
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    before=path.read_bytes()
    window.show_records(path)
    panel=window.records_panel
    pump(app,lambda:not panel.busy())
    assert panel.metadata['statistics']['type_counts']=={'file':1,'text':1}
    assert panel.table.item(0,2).text()=='文件' and panel.table.item(0,1).text()=='第十组.pptx'
    info=RecordInfoDialog(panel.metadata)
    assert '文件：1' in info.details.toPlainText()
    info.hide()
    assert path.read_bytes()==before
