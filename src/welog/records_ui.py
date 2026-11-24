"""Read-only records workspace. File work runs off the UI thread; no desktop APIs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
    QComboBox, QFrame, QInputDialog, QLayout, QLineEdit, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from .archive import write_json
from .desktop_theme import GlassCard, label, primary
from .exports import ExportOptions, FORMATS, RecordSource, STATUS_NAMES, TYPE_NAMES, export_records, preview_records
from .record_state import describe, rebuild_intervals, rename_record, replace_attachment, reset_edit, resolve_attachment, save_edit
from .record_dialogs import MessageEditDialog, RecordInfoDialog
from .timeline import interval_brief, interval_label


class RecordLoadWorker(QThread):
    ready=Signal(dict)
    failed=Signal(str)

    def __init__(self,path,offset,search,time_filter='',parent=None):
        super().__init__(parent)
        self.path,self.offset,self.search=Path(path),offset,search
        self.time_filter=time_filter

    def run(self):
        try:
            source=RecordSource(self.path)
            warning=''
            try:rebuild_intervals(source)
            except (OSError,ValueError,sqlite3.Error,KeyError,TypeError) as error:warning=type(error).__name__
            self.ready.emit({**preview_records(source,offset=self.offset,search=self.search,time_filter=self.time_filter),
                             'path':str(source.path),'time_filter':self.time_filter,'timeline_warning':warning})
        except Exception as error:self.failed.emit(str(error))


class ExportWorker(QThread):
    ready=Signal(dict)
    progress=Signal(int)
    failed=Signal(str)

    def __init__(self,source,directory,formats,options,parent=None):
        super().__init__(parent)
        self.source,self.directory,self.formats,self.options=source,directory,formats,options

    def run(self):
        try:self.ready.emit(export_records(self.source,self.directory,self.formats,self.options,self.progress.emit))
        except Exception as error:self.failed.emit(str(error))


class MutationWorker(QThread):
    failed=Signal(str)
    def __init__(self,action,parent=None):
        super().__init__(parent)
        self.action=action
    def run(self):
        try:self.action()
        except Exception as error:self.failed.emit(str(error))


class RecordsPanel(QWidget):
    source_loaded=Signal(str)
    resume_requested=Signal(str)
    delete_requested=Signal(str)
    record_changed=Signal()

    def __init__(self,output_root:Path,parent=None):
        super().__init__(parent)
        self.output_root=output_root
        self.settings_path=output_root/'export-settings.json'
        self.source=None
        self.rows=[]
        self.page=None
        self.offset=0
        self.last_export=None
        self.load_worker=None
        self.export_worker=None
        self.edit_worker=None
        self.metadata={}
        self._focus_message_id=None
        self._pending=None
        self._generation=0
        self._settings=self.read_settings()
        layout=QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(20)

        preview=GlassCard()
        content=QVBoxLayout(preview)
        content.setContentsMargins(22,22,22,20)
        content.setSpacing(12)
        top=QHBoxLayout()
        top.addWidget(label('消息','heading'))
        top.addStretch()
        self.info_button=QPushButton('信息')
        self.info_button.clicked.connect(self.show_info)
        top.addWidget(self.info_button)
        self.rename_button=QPushButton('重命名')
        self.rename_button.clicked.connect(self.rename)
        top.addWidget(self.rename_button)
        self.delete_button=QPushButton('删除记录')
        self.delete_button.setProperty('role','danger')
        self.delete_button.clicked.connect(lambda:self.delete_requested.emit(str(self.source.path)) if self.source else None)
        top.addWidget(self.delete_button)
        self.import_button=QPushButton('打开记录')
        self.import_button.clicked.connect(self.import_records)
        top.addWidget(self.import_button)
        content.addLayout(top)
        self.resume_button = primary(QPushButton('在此记录上继续采集'))
        self.resume_button.clicked.connect(lambda: self.resume_requested.emit(str(self.source.path)) if self.source else None)
        content.addWidget(self.resume_button)
        self.source_label=label('选择或打开记录','muted')
        self.source_label.setWordWrap(True)
        content.addWidget(self.source_label)
        self.search=QLineEdit()
        self.search.setPlaceholderText('搜索消息（Enter）')
        self.search.setClearButtonEnabled(True)
        self.search.returnPressed.connect(self.search_records)
        filtering=QHBoxLayout()
        filtering.addWidget(self.search,1)
        self.time_filter=QComboBox()
        self.time_filter.setFixedWidth(210)
        self.time_filter.addItem('全部时间段','')
        self.time_filter.view().setMinimumWidth(420)
        self.time_filter.currentIndexChanged.connect(self.search_records)
        filtering.addWidget(self.time_filter)
        content.addLayout(filtering)
        self.stats=label('','muted')
        content.addWidget(self.stats)
        self.table=QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(['昵称','消息内容','类型','状态','时间区间'])
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0,85)
        self.table.setColumnWidth(2,52)
        self.table.setColumnWidth(3,64)
        self.table.setColumnWidth(4,170)
        self.table.verticalHeader().setDefaultSectionSize(48)
        self.table.itemSelectionChanged.connect(self.update_detail)
        content.addWidget(self.table,1)
        pagination=QHBoxLayout()
        self.previous=QPushButton('← 上一页')
        self.next=QPushButton('下一页 →')
        self.page_label=label('每页 200 条','muted')
        self.previous.clicked.connect(lambda:self.load_source(self.source.path,max(0,self.offset-200),self.search.text()))
        self.next.clicked.connect(lambda:self.load_source(self.source.path,self.offset+200,self.search.text()))
        pagination.addWidget(self.previous)
        pagination.addStretch()
        pagination.addWidget(self.page_label)
        pagination.addStretch()
        pagination.addWidget(self.next)
        content.addLayout(pagination)
        self.detail_meta=label('','muted')
        self.detail_meta.setWordWrap(True)
        content.addWidget(self.detail_meta)
        tools=QHBoxLayout()
        self.edit_button=QPushButton('编辑消息')
        self.edit_button.clicked.connect(self.edit_selected)
        self.attach_button=QPushButton('替换附件')
        self.attach_button.clicked.connect(self.attach_selected)
        self.open_attachment_button=QPushButton('打开附件')
        self.open_attachment_button.clicked.connect(self.open_attachment)
        self.reset_button=QPushButton('恢复原始')
        self.reset_button.clicked.connect(self.reset_selected)
        for button in [self.edit_button,self.attach_button,self.open_attachment_button,self.reset_button]:tools.addWidget(button)
        tools.addStretch()
        content.addLayout(tools)
        self.attachment_preview=label('')
        self.attachment_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attachment_preview.setMaximumHeight(130)
        self.attachment_preview.hide()
        content.addWidget(self.attachment_preview)
        self.detail=QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText('选择消息查看正文')
        self.detail.setMinimumHeight(65)
        self.detail.setMaximumHeight(110)
        content.addWidget(self.detail)
        layout.addWidget(preview,1)

        export=GlassCard()
        export.setFixedWidth(286)
        export_shell=QVBoxLayout(export)
        export_shell.setContentsMargins(1,1,1,1)
        scroll=QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('QScrollArea{background:transparent;border:0;}')
        scroll.viewport().setAutoFillBackground(False)
        form_body=QWidget()
        form_body.setAutoFillBackground(False)
        form=QVBoxLayout(form_body)
        form.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        scroll.setWidget(form_body)
        form_body.setAutoFillBackground(False)  # setWidget enables this by default.
        export_shell.addWidget(scroll)
        form.setContentsMargins(22,24,22,22)
        form.setSpacing(8)
        form.addWidget(label('导出','heading'))
        form.addSpacing(7)
        names=QGridLayout()
        names.setHorizontalSpacing(10)
        names.setVerticalSpacing(6)
        names.addWidget(label('对方昵称','muted'),0,0)
        names.addWidget(label('我方昵称','muted'),0,1)
        self.incoming=QLineEdit(str(self._settings.get('incoming_name','对方')))
        self.incoming.setMaxLength(80)
        self.incoming.setPlaceholderText('对方')
        self.incoming.setAccessibleName('对方昵称 incoming')
        names.addWidget(self.incoming,1,0)
        self.outgoing=QLineEdit(str(self._settings.get('outgoing_name','我')))
        self.outgoing.setMaxLength(80)
        self.outgoing.setPlaceholderText('我')
        self.outgoing.setAccessibleName('我方昵称 outgoing')
        names.addWidget(self.outgoing,1,1)
        form.addLayout(names)
        self.incoming.textChanged.connect(self.refresh_names)
        self.outgoing.textChanged.connect(self.refresh_names)
        form.addWidget(label('记录标题','muted'))
        self.title=QLineEdit(str(self._settings.get('title','聊天记录')))
        self.title.setMaxLength(80)
        form.addWidget(self.title)
        form.addSpacing(3)
        form.addWidget(label('格式（多选）','muted'))
        formats=QGridLayout()
        formats.setSpacing(3)
        self.formats={}
        selected=self._settings.get('formats',['txt','html'])
        for index,(key,name) in enumerate(FORMATS.items()):
            box=QCheckBox(name)
            box.setMinimumHeight(28)
            box.setChecked(key in selected)
            self.formats[key]=box
            box.toggled.connect(self.update_actions)
            formats.addWidget(box,index//2,index%2)
        form.addLayout(formats)
        form.addSpacing(4)
        form.addWidget(label('导出到','muted'))
        self.directory=QLineEdit(str(self._settings.get('directory',output_root.parent/'exports')))
        self.directory.setCursorPosition(0)
        self.directory.setToolTip('导出目录，可直接输入或点击“选择”。')
        destination=QHBoxLayout()
        destination.setSpacing(6)
        destination.addWidget(self.directory,1)
        self.browse=QPushButton('选择')
        self.browse.setAccessibleName('选择导出目录')
        self.browse.setToolTip('选择导出目录')
        self.browse.setFixedWidth(62)
        self.browse.clicked.connect(self.choose_directory)
        destination.addWidget(self.browse)
        form.addLayout(destination)
        form.addStretch(1)
        self.export_button=primary(QPushButton('导出全部消息'))
        self.export_button.setToolTip('导出该记录中的全部消息，搜索、时间筛选和分页不影响导出范围。')
        self.export_button.clicked.connect(self.start_export)
        form.addWidget(self.export_button)
        self.open_directory=QPushButton('打开导出目录')
        self.open_directory.setEnabled(False)
        self.open_directory.clicked.connect(self.open_export_directory)
        form.addWidget(self.open_directory)
        self.feedback=label('','muted')
        self.feedback.setWordWrap(True)
        self.feedback.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addWidget(self.feedback)
        layout.addWidget(export)
        self.update_actions()

    def read_settings(self):
        try:
            value=json.loads(self.settings_path.read_text(encoding='utf-8'))
            if not isinstance(value,dict):return {}
            result={key:item for key,item in value.items() if key in {'incoming_name','outgoing_name','title','directory'} and isinstance(item,str)}
            if isinstance(value.get('formats'),list):result['formats']=[key for key in value['formats'] if isinstance(key,str) and key in FORMATS]
            return result
        except (OSError,ValueError):return {}

    def options(self):
        return ExportOptions(self.incoming.text(),self.outgoing.text(),self.title.text().strip() or '聊天记录')

    def busy(self):
        return self.load_worker is not None or self.export_worker is not None or self.edit_worker is not None

    def update_actions(self):
        loading=self.load_worker is not None
        exporting=self.export_worker is not None
        available=self.source is not None and not self.busy()
        self.resume_button.setEnabled(available)
        self.export_button.setEnabled(available and any(box.isChecked() for box in self.formats.values()))
        for button in [self.delete_button,self.rename_button,self.info_button]:button.setEnabled(available)
        row=self.selected_record()
        for button in [self.edit_button,self.attach_button]:button.setEnabled(available and row is not None)
        self.reset_button.setEnabled(bool(available and row and row.get('edited')))
        self.open_attachment_button.setEnabled(bool(available and row and resolve_attachment(self.source,row)))
        self.previous.setEnabled(bool(self.source and self.offset>0 and not loading))
        self.next.setEnabled(bool(self.page and self.offset+200<self.page['matched'] and not loading))

    def import_records(self):
        path,_=QFileDialog.getOpenFileName(self,'打开消息记录',str(self.output_root),
            '消息记录 (*.sqlite3 *.db *.jsonl *.json)')
        if path:self.search.clear();self.load_source(path)

    def load_source(self,path,offset=0,search='',time_filter=None):
        if time_filter is None:
            time_filter=(self.time_filter.currentData() or '') if self.source and Path(path).resolve() in {self.source.path,self.source.path.parent} else ''
        self._generation+=1
        request=(path,offset,search,time_filter,self._generation)
        if self.load_worker is not None:
            self._pending=request
            return
        self._start_load(request)

    def _start_load(self,request):
        path,offset,search,time_filter,generation=request
        worker=RecordLoadWorker(path,offset,search,time_filter,self)
        self.load_worker=worker
        self.feedback.setText('读取中…')
        worker.ready.connect(lambda result:self._loaded(result,generation))
        worker.failed.connect(lambda message:self._load_failed(message,generation))
        worker.finished.connect(self._load_finished)
        worker.start()
        self.update_actions()

    def _loaded(self,result,generation):
        if generation!=self._generation:return
        changed=self.source is None or self.source.path!=Path(result['path'])
        self.source=RecordSource(result['path'])
        self.metadata=result['metadata']
        if changed:self.title.setText(self.metadata['name'])
        self.rows=result['rows']
        self.page=result
        self.offset=result['offset']
        path=self.source.path
        self.source_label.setText(self.metadata['name'])
        self.source_label.setToolTip(str(path))
        statistics=f"{result['total']:,} 条消息 · {result['incomplete']:,} 条待核对"
        if self.search.text().strip():statistics+=f" · 匹配 {result['matched']:,} 条"
        self.stats.setText(statistics)
        self.time_filter.blockSignals(True)
        self.time_filter.clear()
        self.time_filter.addItem('全部时间段','')
        for group,count in result['statistics']['time_groups'].items():self.time_filter.addItem(f'{group} ({count})',group)
        self.time_filter.setCurrentIndex(max(0,self.time_filter.findData(result.get('time_filter',''))))
        self.time_filter.blockSignals(False)
        self.time_filter.setToolTip(self.time_filter.currentData() or 'UTC+08:00')
        self.page_label.setText(f"{self.offset+1 if self.rows else 0}–{self.offset+len(self.rows)} / {result['matched']:,}")
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        for index,row in enumerate(self.rows):
            values=[self.options().sender(row),row['text'].replace('\n','  '),TYPE_NAMES.get(row['type'],str(row['type'])),
                    STATUS_NAMES.get(row['status'],str(row['status']))+(' *' if row.get('edited') else ''),interval_brief(row)]
            for column,value in enumerate(values):
                item=QTableWidgetItem(value)
                if column==3 and row['status']!='complete':item.setForeground(QColor('#ac7c62'))
                if column==0:item.setToolTip(value)
                if column==4:item.setToolTip(interval_label(row)+'\nUTC+08:00；区间范围，不代表精确发送时间')
                self.table.setItem(index,column,item)
        self.table.blockSignals(False)
        self.detail.clear()
        if self.rows:
            selected=next((index for index,row in enumerate(self.rows) if row['message_id']==self._focus_message_id),0)
            self.table.selectRow(selected)
        self._focus_message_id=None
        self.update_detail()
        self.feedback.setText('' if self.rows else '没有消息')
        if result.get('timeline_warning'):self.feedback.setText('时间区间未更新，消息仍可查看。')
        self.source_loaded.emit(str(path))

    def _load_failed(self,message,generation):
        if generation==self._generation:self.feedback.setText('读取失败：'+message)

    def _load_finished(self):
        worker=self.load_worker
        self.load_worker=None
        worker.deleteLater()
        if self._pending:
            request=self._pending
            self._pending=None
            self._start_load(request)
        self.update_actions()

    def search_records(self):
        if self.source:self.load_source(self.source.path,0,self.search.text())

    def selected_record(self):
        index=self.table.currentRow()
        return self.rows[index] if 0<=index<len(self.rows) else None

    def rename(self):
        if self.source is None or self.busy():return
        name,ok=QInputDialog.getText(self,'重命名记录','记录名称',text=self.metadata.get('name',''))
        if not ok:return
        try:
            rename_record(self.source,name)
            self.title.setText(name.strip())
            self.record_changed.emit()
            self.load_source(self.source.path,self.offset,self.search.text())
        except (OSError,ValueError) as error:self.feedback.setText('重命名失败：'+str(error))

    def show_info(self):
        if self.source and not self.busy():RecordInfoDialog(self.metadata,self).exec()

    def mutate(self,action):
        if self.busy():return
        row=self.selected_record()
        self._focus_message_id=row['message_id'] if row else None
        self.feedback.setText('保存中…')
        worker=MutationWorker(action,self)
        self.edit_worker=worker
        self._edit_error=None
        worker.failed.connect(self._mutation_failed)
        worker.finished.connect(self._mutation_finished)
        worker.start()
        self.update_actions()

    def _mutation_failed(self,message):self._edit_error=message

    def _mutation_finished(self):
        worker=self.edit_worker
        self.edit_worker=None
        worker.deleteLater()
        if self._edit_error:self.feedback.setText('保存失败：'+self._edit_error)
        elif self.source:
            self.record_changed.emit()
            self.load_source(self.source.path,self.offset,self.search.text())
        self.update_actions()

    def edit_selected(self):
        row=self.selected_record()
        if row is None or self.busy():return
        dialog=MessageEditDialog(row,self)
        if dialog.exec()==dialog.DialogCode.Accepted:
            source=self.source;patch=dialog.patch()
            self.mutate(lambda:save_edit(source,row,patch))

    def attach_selected(self):
        row=self.selected_record()
        if row is None or self.busy():return
        path,_=QFileDialog.getOpenFileName(self,'选择真实图片或文件','',
            '图片 (*.png *.jpg *.jpeg *.webp *.gif *.bmp);;所有文件 (*)' if row['type']=='image' else '所有文件 (*)')
        if path:
            source=self.source
            self.mutate(lambda:replace_attachment(source,row,path))

    def reset_selected(self):
        row=self.selected_record()
        if row is not None and not self.busy():
            source=self.source
            self.mutate(lambda:reset_edit(source,row))

    def open_attachment(self):
        row=self.selected_record()
        if row is not None:
            path=resolve_attachment(self.source,row)
            if path:QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def clear_source(self):
        if self.busy():raise RuntimeError('Records are in use')
        self._generation+=1
        self.source=None
        self.metadata={}
        self.rows=[]
        self.page=None
        self.offset=0
        self.last_export=None
        self.search.clear()
        self.time_filter.blockSignals(True)
        self.time_filter.clear();self.time_filter.addItem('全部时间段','')
        self.time_filter.blockSignals(False)
        self.table.setRowCount(0)
        self.source_label.setText('选择或打开记录')
        self.source_label.setToolTip('')
        self.stats.clear()
        self.detail.clear()
        self.detail_meta.clear()
        self.page_label.setText('每页 200 条')
        self.open_directory.setEnabled(False)
        self.update_actions()

    def refresh_names(self):
        options=self.options()
        for index,row in enumerate(self.rows):
            item=self.table.item(index,0)
            if item:item.setText(options.sender(row));item.setToolTip(options.sender(row))
        self.update_detail()

    def update_detail(self):
        self.attachment_preview.clear()
        self.attachment_preview.hide()
        index=self.table.currentRow()
        if not 0<=index<len(self.rows):
            self.detail_meta.clear()
            self.detail.clear()
            self.update_actions()
            return
        row=self.rows[index]
        duration=f" · {row['duration_seconds']} 秒" if row.get('duration_seconds') is not None else ''
        self.detail_meta.setText(f"{self.options().sender(row)} · {TYPE_NAMES.get(row['type'],row['type'])}{duration} · {STATUS_NAMES.get(row['status'],row['status'])}\n{interval_label(row)}")
        self.detail.setPlainText(row['text'])
        path=resolve_attachment(self.source,row)
        if path and row['type']=='image':
            QImageReader.setAllocationLimit(64)
            reader=QImageReader(str(path))
            reader.setAutoTransform(True)
            size=reader.size()
            if size.isValid():reader.setScaledSize(size.scaled(QSize(420,130),Qt.AspectRatioMode.KeepAspectRatio))
            image=reader.read()
            if not image.isNull():
                self.attachment_preview.setPixmap(QPixmap.fromImage(image))
                self.attachment_preview.show()
        elif row.get('attachment'):
            attachment=row['attachment']
            self.detail_meta.setText(self.detail_meta.text()+f"\n{attachment.get('name','附件')} · {attachment.get('size_bytes',0):,} 字节")
        self.update_actions()

    def choose_directory(self):
        path=QFileDialog.getExistingDirectory(self,'选择导出目录',self.directory.text())
        if path:self.directory.setText(path)

    def start_export(self):
        if self.source is None or self.busy():return
        if not self.directory.text().strip():self.feedback.setText('请先选择导出目录。');return
        selected=[key for key,box in self.formats.items() if box.isChecked()]
        if not selected:self.feedback.setText('请至少选择一种格式。');return
        options=self.options()
        directory=self.directory.text().strip()
        try:
            self.output_root.mkdir(parents=True,exist_ok=True)
            write_json(self.settings_path,{'incoming_name':options.incoming_name,'outgoing_name':options.outgoing_name,
                'title':options.title,'directory':directory,'formats':selected})
        except OSError:
            pass  # A read-only settings location must not prevent exporting elsewhere.
        self.feedback.setText('正在导出全部消息…')
        self.open_directory.setEnabled(False)
        worker=ExportWorker(self.source,directory,selected,options,self)
        self.export_worker=worker
        worker.ready.connect(self.export_ready)
        worker.failed.connect(lambda message:self.feedback.setText('导出失败：'+message))
        worker.progress.connect(lambda count:self.feedback.setText(f'正在导出：{count:,} 条消息…'))
        worker.finished.connect(self._export_finished)
        worker.start()
        self.update_actions()

    def export_ready(self,result):
        self.last_export=result
        self.feedback.setText(f"已导出 {result['count']:,} 条消息\n生成 {len(result['files'])} 个文件")
        if result.get('metadata_saved_to_source') is False:self.feedback.setText(self.feedback.text()+'\n源目录不可写，元信息已随文件导出。')
        self.feedback.setToolTip('\n'.join(result['files']))
        self.open_directory.setEnabled(True)
        if self.source:self.metadata=describe(self.source,(self.page or {}).get('statistics',{}))

    def _export_finished(self):
        worker=self.export_worker
        self.export_worker=None
        worker.deleteLater()
        self.update_actions()

    def open_export_directory(self):
        if self.last_export:QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_export['directory']))
