"""Render only WeLog's own widgets with invented messages, never the desktop."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['QT_QPA_PLATFORM']='offscreen'

from PySide6.QtWidgets import QApplication
from welog.gui import CaptureWindow, configure_font
from welog.exports import RecordSource
from welog.record_state import rename_record


def forbidden(*args):
    raise AssertionError('Preview must never query or operate desktop windows')


def main():
    output=Path('runs/gui-preview').resolve()
    output.mkdir(parents=True,exist_ok=True)
    source=output/'empty-root'/'pointer-preview'/'capture'/'messages.jsonl'
    source.parent.mkdir(parents=True,exist_ok=True)
    messages=[
        ('incoming','text','周末一起去看展吗？\n我找到了一个很有意思的展览。','complete'),
        ('outgoing','text','好呀，周六下午怎么样？','complete'),
        ('incoming','audio',"19''",'complete'),
        ('incoming','text','可以！结束后还可以去附近喝杯咖啡。','complete'),
        ('outgoing','text','那就这样说定了，我来订票。','complete'),
        ('incoming','text','对了，展览的地址是……','partial'),
        ('incoming','file','展览介绍.pdf','complete'),
    ]
    with source.open('w',encoding='utf-8') as file:
        for index,(direction,kind,text,status) in enumerate(messages):
            file.write(json.dumps({'message_id':str(index),'direction':direction,'type':kind,'text':text,'status':status,
                'time_interval':{'start':'2025-12-02T12:41+08:00' if index<3 else '2025-12-02T14:22+08:00',
                                 'end':'2025-12-02T14:22+08:00' if index<3 else None,'status':'bounded' if index<3 else 'open_end','timezone':'UTC+08:00','precision':'interval'},
                **({'duration_seconds':19} if kind=='audio' else {})},ensure_ascii=False)+'\n')
    rename_record(RecordSource(source),'林间')
    app=QApplication([])
    app.setQuitOnLastWindowClosed(False)
    configure_font(app)
    desktop=SimpleNamespace(activate=forbidden,describe_window=forbidden,window_at_cursor=forbidden,escape_pressed=forbidden)
    window=CaptureWindow(desktop,output/'empty-root')
    window.show()
    app.processEvents()
    window.grab().save(str(output/'workbench-capture.png'))
    window.records_panel.incoming.setText('林间')
    window.records_panel.outgoing.setText('我')
    window.show_records(source)
    deadline=time.monotonic()+10
    while window.records_panel.load_worker is not None and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.005)
    assert window.records_panel.source is not None
    app.processEvents()
    window.grab().save(str(output/'workbench-records.png'))
    window.hide()
    print(output)


if __name__=='__main__':main()
