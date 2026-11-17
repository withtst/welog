"""Small deterministic rules shared by normal OCR, seam OCR and stored-result repair."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import replace


_NOTICE_TOKEN = r"(?<!\d)\d+\s*[+＋]?\s*条\s*新\s*消\s*息"
_NOTICE = re.compile(_NOTICE_TOKEN)
_NOTICE_LINE = re.compile(r"(?m)^[ \t]*" + _NOTICE_TOKEN + r"[ \t]*(?:\r?\n|$)")
_DURATION = re.compile(r'''(?P<seconds>\d+)\s*(?:['‘’′ʼ]\s*['‘’′ʼ]|["“”″])''')
_BRACKETS = {'[':']','【':'】','(':')'}


def clean_notice(text: str) -> tuple[str, bool]:
    """Only numbered unread-message notices are removed; other text is preserved."""
    if not _NOTICE.search(text):
        return text, False
    cleaned = _NOTICE_LINE.sub('', text)
    cleaned = _NOTICE.sub('', cleaned).strip()
    return cleaned, True


def audio_duration(text: str) -> int | None:
    """A whole message must be a seconds marker; a duration inside prose is not audio."""
    value = unicodedata.normalize('NFKC', text).strip()
    if value and value[0] in _BRACKETS and value.endswith(_BRACKETS[value[0]]):
        value = value[1:-1].strip()
    match = _DURATION.fullmatch(value)
    return int(match['seconds']) if match else None


def clean_lines(lines):
    output = []
    for line in lines:
        text, changed = clean_notice(line.text)
        if text.strip():
            output.append(replace(line, text=text, notice_removed=line.notice_removed or changed))
    return output


def sanitize_bubble(bubble) -> bool:
    """False means the candidate consisted only of an unread overlay, not an OCR failure."""
    if bubble.kind not in {'text','audio'}:
        return True
    affected = any(line.notice_removed or clean_notice(line.text)[1] for line in bubble.lines)
    bubble.lines = clean_lines(bubble.lines)
    if affected and not bubble.lines:
        return False
    if audio_duration('\n'.join(line.text for line in bubble.lines)) is not None:
        bubble.kind = 'audio'
    bubble.ocr_status = 'ok' if bubble.lines else 'failed'
    return True


def classify_stored_message(connection, identity):
    row = connection.execute('SELECT kind FROM messages WHERE id=?',(identity,)).fetchone()
    if row is None or row['kind'] not in {'text','audio'}:
        return
    text = '\n'.join(line['text'] for line in connection.execute(
        'SELECT text FROM lines WHERE message_id=? ORDER BY top,x1',(identity,)))
    if audio_duration(text) is not None:
        connection.execute("UPDATE messages SET kind='audio' WHERE id=?",(identity,))


def sanitize_store(store) -> dict:
    """Repair a COPY of existing structured results, without needing or writing images."""
    db = store.connection
    stats = {'cleaned_lines':0,'removed_messages':0,'classified_audio':0}
    for message in db.execute('SELECT id,kind FROM messages').fetchall():
        if message['kind'] not in {'text','audio'}:
            continue
        identity = message['id'];affected = False
        for line in db.execute('SELECT id,text FROM lines WHERE message_id=?',(identity,)).fetchall():
            text, changed = clean_notice(line['text'])
            if not changed:
                continue
            affected = True;stats['cleaned_lines'] += 1
            if text.strip():db.execute('UPDATE lines SET text=? WHERE id=?',(text,line['id']))
            else:db.execute('DELETE FROM lines WHERE id=?',(line['id'],))
        if affected and not db.execute('SELECT 1 FROM lines WHERE message_id=?',(identity,)).fetchone():
            db.execute('DELETE FROM observations WHERE message_id=?',(identity,))
            db.execute('DELETE FROM messages WHERE id=?',(identity,))
            stats['removed_messages'] += 1
            continue
        classify_stored_message(db,identity)
        updated = db.execute('SELECT kind FROM messages WHERE id=?',(identity,)).fetchone()
        if message['kind']!='audio' and updated['kind']=='audio':stats['classified_audio'] += 1
    db.commit()
    return stats
