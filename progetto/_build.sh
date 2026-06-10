#!/bin/zsh
set -e
cd /Users/davspiri/Documents/GitHub/Metodi-ottimizzazione-Big-Data/progetto
SK="/Users/davspiri/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/6600c774-3f60-4cd2-8fd7-e38498974507/64cda4e8-cd64-40bf-b750-5cbe51ff8399/skills/docx"
python3 _pre.py
pandoc documentazione_pp.tex -f latex -t docx --highlight-style=tango --toc --toc-depth=2 -o _raw.docx
rm -rf _unpacked
python3 "$SK/scripts/office/unpack.py" _raw.docx _unpacked/ >/dev/null
python3 _post.py
python3 "$SK/scripts/office/pack.py" _unpacked/ documentazione.docx --original _raw.docx 2>&1 | tail -6
echo "=== build done ==="
