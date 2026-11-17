"""Explicit record deletion. Resolve and validate the complete directory before removal."""
from dataclasses import dataclass
from pathlib import Path
import shutil

from .exports import RecordSource


@dataclass(frozen=True)
class RecordDeletion:
    source: Path
    root: Path
    directory: Path | None
    files: tuple[Path,...]
    asset_directories: tuple[Path,...]=()

    @property
    def target(self):
        return self.directory or self.source


def plan_deletion(path,output_root):
    selected=RecordSource(path)
    source=selected.path
    root=Path(output_root).resolve()
    managed=source.name in {'messages.sqlite3','messages.jsonl'} and source.is_relative_to(root)
    if managed and source.parent!=root:
        directory=source.parent
        if directory.name=='capture' and directory.parent.parent==root and directory.parent.name.startswith('pointer-'):
            directory=directory.parent
        # Never delete the configured root, or anything outside that exact root.
        if directory==root or not directory.is_relative_to(root):raise ValueError('记录目录超出采集目录')
        return RecordDeletion(source,root,directory,())
    files=[source]
    if managed:
        files=[path for path in [root/'messages.sqlite3',root/'messages.jsonl'] if path.is_file()]
    from .record_state import asset_root,state_path
    state=state_path(selected)
    if state.is_file():files.append(state)
    assets=asset_root(selected)
    for file in tuple(files):
        if file.suffix in {'.sqlite3','.db'}:
            files.extend(candidate for suffix in ['-wal','-shm','-journal'] if (candidate:=file.with_name(file.name+suffix)).is_file())
    return RecordDeletion(source,root,None,tuple(files),(assets,) if assets.is_dir() else ())


def delete_record(plan:RecordDeletion):
    current=plan_deletion(plan.source,plan.root)
    if current!=plan:raise ValueError('记录位置或内容已变化，请重新选择')
    for directory in plan.asset_directories:
        if directory.resolve()!=directory or directory.parent!=plan.source.parent:raise ValueError('非法附件目录')
    if plan.directory:
        target=plan.directory.resolve()
        if target!=plan.directory or target==plan.root or not target.is_relative_to(plan.root):
            raise ValueError('非法记录目录')
        shutil.rmtree(target)
    else:
        for path in plan.files:path.unlink(missing_ok=True)
        for directory in plan.asset_directories:shutil.rmtree(directory)
