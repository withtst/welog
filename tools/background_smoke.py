"""Explicit native integration test against two OWN no-activate synthetic windows only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
import ctypes
from ctypes import wintypes

os.environ.pop('QT_QPA_PLATFORM',None)
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QImage,QPainter
from PySide6.QtWidgets import QApplication,QWidget
from welog.background import BackgroundSource
from welog.controller import run_capture
from welog.demo import make_scene
from welog.models import CaptureConfig,Rect
from welog.recognition import EmptyOcr
from welog.replay import run_replay
from welog.windows import WindowsDesktop
from welog.archive import write_json
from welog.imageio import read_image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--direction',choices=['up','down'],default='down')
    parser.add_argument('--frames',type=int,default=8,help='0: run until boundary')
    args=parser.parse_args()
    if args.frames<0:raise ValueError('frames must be nonnegative')
    desktop=WindowsDesktop()
    app=QApplication([])
    class Canvas(QWidget):
        scene=None;position=0;wheel_calls=0
        def __init__(self):
            super().__init__()
            self.setWindowTitle('WeLog background synthetic fixture')
            self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus,True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.resize(400,500);self.move(80,80)
        def paintEvent(self,event):
            if self.scene is None:return
            pixels=self.scene[self.position:self.position+self.pixel_height]
            qimage=QImage(pixels.data,pixels.shape[1],pixels.shape[0],pixels.strides[0],QImage.Format.Format_BGR888)
            painter=QPainter(self)
            painter.scale(1/self.devicePixelRatioF(),1/self.devicePixelRatioF())
            painter.drawImage(0,0,qimage)
        def wheelEvent(self,event):
            self.wheel_calls+=1
            self.position=max(0,min(1600,self.position-round(event.angleDelta().y()/120)*80))
            self.update();event.accept()
    canvas=Canvas();canvas.show();app.processEvents()
    hwnd=int(canvas.winId())
    # Make this OWN fixture rectangular so pixel equality doesn't include Win11's rounded-corner mask.
    dwm=ctypes.WinDLL('dwmapi')
    dwm.DwmSetWindowAttribute.argtypes=[wintypes.HWND,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD]
    preference=ctypes.c_int(1)
    dwm.DwmSetWindowAttribute(hwnd,33,ctypes.byref(preference),ctypes.sizeof(preference))
    client=desktop.client_rect(hwnd)
    canvas.pixel_height=client.height
    canvas.scene=make_scene(client.height+1600,client.width,seed=81)
    canvas.position=1600 if args.direction=='up' else 0
    initial_position=canvas.position
    cover=QWidget();cover.setWindowTitle('WeLog synthetic occluder')
    cover.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus,True)
    cover.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    cover.setStyleSheet('background:#eab2c5')
    cover.resize(canvas.width()+80,canvas.height()+100);cover.move(canvas.x()-30,canvas.y()-30)
    cover.show();cover.lower();canvas.lower();app.processEvents()
    timer=QTimer();timer.setInterval(60);timer.timeout.connect(canvas.update);timer.start()
    def forbidden(*args,**kwargs):raise AssertionError('Foreground activation and global input are forbidden')
    desktop.activate=desktop.scroll=forbidden
    desktop.api.SetForegroundWindow=desktop.api.SetCursorPos=desktop.api.SendInput=forbidden
    result={};errors=[];offset_errors=[];pixel_errors=[]
    def progress(row):
        if row['accepted']:
            offset_errors.append(row['scroll_offset_px']-(canvas.position-initial_position))
            frame=read_image(args.output/'capture'/'frames'/f"{row['frame_id']}.png")
            expected=canvas.scene[canvas.position:canvas.position+client.height]
            pixel_errors.append(int(np.max(np.abs(frame.astype(np.int16)-expected.astype(np.int16)))))
    def worker():
        try:
            source=BackgroundSource(desktop,hwnd,Rect(0,0,client.width,client.height))
            result.update(run_capture(source,args.output/'capture',config=CaptureConfig(capture_backend='background',
                direction=args.direction,max_frames=args.frames,sample_interval_s=.03,stable_samples=2,settle_delay_s=.15,
                debug_save_images=True,debug_image_policy='all',boundary_wait_s=0),ocr_backend=EmptyOcr(),
                progress=progress))
        except Exception as error:errors.append(type(error).__name__+': '+str(error))
    thread=threading.Thread(target=worker);thread.start()
    deadline=time.monotonic()+40
    while thread.is_alive() and time.monotonic()<deadline:
        app.processEvents();time.sleep(.005)
    if thread.is_alive():raise RuntimeError('Native background test timed out')
    timer.stop();cover.hide();canvas.hide();thread.join()
    if errors:raise RuntimeError(errors)
    report={'capture':result,'wheel_calls':canvas.wheel_calls,'actual_position_px':canvas.position,
        'actual_scroll_px':canvas.position-initial_position,'direction':args.direction,
        'offset_errors_px':offset_errors,'frame_pixel_errors':pixel_errors,'only_synthetic_windows':True,'forbidden_input_calls':0}
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'native-validation.json',report)
    expected_status='boundary_reached' if args.frames==0 else 'limit_reached'
    if result['status']!=expected_status or not canvas.wheel_calls or any(offset_errors) or any(pixel_errors):raise RuntimeError(report)
    replay=run_replay(args.output/'capture',args.output/'replay')
    assert replay['accepted']==result['accepted'] and replay['source_checkpoint_difference_px']==0
    print(json.dumps({'status':result['status'],'accepted':result['accepted'],'actual_position_px':canvas.position,
        'offset_error_px':max(map(abs,offset_errors)),'replay_accepted':replay['accepted']}))


if __name__=='__main__':main()
