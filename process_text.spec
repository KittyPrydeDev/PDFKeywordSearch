# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
import os
import ntpath
import PyQt5


a = Analysis(['process_text.py'],
             pathex=['C:\\Users\\john.DEV\\source\\repos\\pdfkeywordsearching', os.path.join(ntpath.dirname(PyQt5.__file__), 'Qt', 'bin')],
             binaries=[('C:\\Users\\john.DEV\\Anaconda3\\envs\\pdfkeywordsearching\\Lib\\site-packages\\pandas\\_libs\\window\concrt140.dll', '.'), ('C:\\Users\\john.DEV\\Anaconda3\\envs\\pdfkeywordsearching\\Lib\\site-packages\\pandas\\_libs\\window\\msvcp140.dll', '.') ],
             datas=[('tika-server-1.24.jar', '.'), ('tika-server-1.24.jar.md5', '.'), ('C:\\Users\\john.DEV\\Anaconda3\\envs\pdfkeywordsearching\\Lib\\site-packages\\langdetect', 'langdetect'), ('C:\\Users\\john.DEV\\Anaconda3\envs\pdfkeywordsearching\\Lib\\site-packages\\en_core_web_sm', 'en_core_web_sm'), ('font\\','font\\')],
             hiddenimports=['pkg_resources.py2_warn', 'srsly.msgpack.util', 'cymem.cymem', 'thinc.linalg', 'thinc.neural._aligned_alloc', 'thinc.extra.search', 'preshed.maps', 'murmurhash.mrmr', 'cytoolz.utils', 'cytoolz._signatures', 'spacy.strings', 'spacy.morphology', 'spacy.lexeme', 'spacy.tokens', 'spacy.gold', 'spacy.tokens.underscore', 'spacy.tokens.morphanalysis',  'spacy.parts_of_speech', 'dill', 'spacy.tokens.printers', 'spacy.tokens._retokenize', 'spacy.syntax', 'spacy.syntax.stateclass', 'spacy.syntax.transition_system', 'spacy.syntax.nonproj', 'spacy.syntax.nn_parser', 'spacy.syntax.arc_eager', 'spacy.syntax._beam_utils', 'spacy.syntax.ner',  'spacy.vocab', 'spacy.lemmatizer', 'spacy._ml', 'spacy.lang.en',  'blis', 'blis.py', 'spacy.matcher._schemas', 'spacy._align', 'spacy.syntax._parser_model', 'spacy.kb'],
             hookspath=['.'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='processtext',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               hiddenimports=a.hiddenimports,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='processtext')
