"""Replace text in a file while preserving its line endings.
usage: crlf_edit.py FILE OLD_FILE NEW_FILE  (OLD/NEW given with \n endings)"""
import sys
path, old_f, new_f = sys.argv[1:4]
raw = open(path, 'rb').read().decode('utf-8')
crlf = '\r\n' in raw
text = raw.replace('\r\n', '\n')
old = open(old_f, encoding='utf-8').read()
new = open(new_f, encoding='utf-8').read()
cnt = text.count(old)
if cnt != 1:
    sys.exit(f"{path}: old text found {cnt} times")
text = text.replace(old, new)
if crlf:
    text = text.replace('\n', '\r\n')
open(path, 'wb').write(text.encode('utf-8'))
print(f"edited {path} (crlf={crlf})")
