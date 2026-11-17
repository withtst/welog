"""Explicit opt-in image retention; the normal session never imports an image writer."""
import hashlib
from .session import MemorySession
from .imageio import write_image


class DebugImageLimit(RuntimeError):
    pass


class DebugSession(MemorySession):
    def __init__(self,path,metadata,max_bytes):
        super().__init__(path,{**metadata,'kind':'live_debug','storage_mode':'debug_images',
            'debug_save_images':True,'debug_max_bytes':max_bytes})
        self.max_bytes=max_bytes
        self.image_bytes=0
        self.image_count=0
        (self.path/'frames').mkdir()

    def capture(self,image,action):
        relative=f'frames/{self.counter:06d}.png'
        target=self.path/relative
        if self.image_bytes>=self.max_bytes:raise DebugImageLimit('Debug image quota reached')
        write_image(target,image)
        size=target.stat().st_size
        if self.image_bytes+size>self.max_bytes:
            target.unlink()
            raise DebugImageLimit('Debug image quota reached')
        with target.open('rb') as file:checksum=hashlib.file_digest(file,'sha256').hexdigest()
        frame=self.frame(image.shape,action,{'image':relative,'sha256':checksum})
        self.image_bytes+=size
        self.image_count+=1
        return frame
