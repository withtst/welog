"""Small forms for record information and manual corrections."""
from datetime import datetime
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QPlainTextEdit, QSpinBox, QTextEdit, QVBoxLayout
from .exports import TYPE_NAMES, STATUS_NAMES
from .record_state import KINDS
from .timeline import CHAT_TZ


class MessageEditDialog(QDialog):
    def __init__(self,record,parent=None):
        super().__init__(parent)
        self.setWindowTitle('编辑消息')
        self.resize(560,430)
        layout=QVBoxLayout(self)
        form=QFormLayout()
        self.direction=QComboBox()
        for key,name in [('incoming','对方'),('outgoing','我方'),('unknown','未知')]:self.direction.addItem(name,key)
        self.direction.setCurrentIndex(max(0,self.direction.findData(record['direction'])))
        self.kind=QComboBox()
        for key in KINDS:self.kind.addItem(TYPE_NAMES[key],key)
        self.kind.setCurrentIndex(max(0,self.kind.findData(record['type'])))
        self.status=QComboBox()
        for key,value in STATUS_NAMES.items():self.status.addItem(value,key)
        self.status.setCurrentIndex(max(0,self.status.findData(record['status'])))
        self.duration=QSpinBox()
        self.duration.setRange(0,86400)
        self.duration.setSuffix(' 秒')
        self.duration.setValue(record.get('duration_seconds') or 0)
        self.duration.setEnabled(self.kind.currentData()=='audio')
        self.kind.currentIndexChanged.connect(lambda:self.duration.setEnabled(self.kind.currentData()=='audio'))
        form.addRow('方向',self.direction)
        form.addRow('类型',self.kind)
        form.addRow('状态',self.status)
        form.addRow('语音时长',self.duration)
        layout.addLayout(form)
        self.text=QTextEdit()
        self.text.setPlainText(record['text'])
        layout.addWidget(self.text,1)
        buttons=QDialogButtonBox()
        buttons.addButton('保存',QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton('取消',QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def patch(self):
        return {'text':self.text.toPlainText(),'direction':self.direction.currentData(),'type':self.kind.currentData(),
                'status':self.status.currentData(),'duration_seconds':self.duration.value() if self.kind.currentData()=='audio' else None}


class RecordInfoDialog(QDialog):
    def __init__(self,metadata,parent=None):
        super().__init__(parent)
        self.setWindowTitle('记录信息')
        self.resize(540,440)
        layout=QVBoxLayout(self)
        def local(value):
            if not value:return '未知'
            try:return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(CHAT_TZ).strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:return value
        stats=metadata.get('statistics',{})
        content=[metadata['name'],'',f"采集开始：{local(metadata.get('capture_started_at'))}",
            f"采集结束：{local(metadata.get('capture_finished_at'))}",f"最近导出：{local(metadata.get('last_export_at'))}",
            '时区：UTC+08:00',f"总消息数：{stats.get('total',0)}",f"待核对：{stats.get('incomplete',0)}",f"人工修改：{stats.get('edited',0)}",'']
        content.extend(f"{TYPE_NAMES[key]}：{stats.get('type_counts',{}).get(key,0)}" for key in KINDS)
        content.append('')
        content.extend(f"{'对方' if key=='incoming' else '我方' if key=='outgoing' else '未知方向'}：{value}" for key,value in stats.get('direction_counts',{}).items())
        content.extend(['','导出历史：'])
        content.extend(f"{local(event['created_at'])} · {event.get('count',0)} 条 · {', '.join(event.get('formats',[]))}" for event in metadata.get('export_history',[]))
        self.details=QPlainTextEdit('\n'.join(content))
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        buttons=QDialogButtonBox()
        buttons.addButton('关闭',QDialogButtonBox.ButtonRole.RejectRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
