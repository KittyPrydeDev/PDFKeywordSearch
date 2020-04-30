# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['processtext.py'],
             pathex=['C:\\Users\\Lisa\\Documents\\GitHub\\pdfkeywordsearching'],
             binaries=[],
             datas=[('tika-server-1.24.jar', '.'), ('tika-server-1.24.jar.md5', '.'), ('C:\ProgramData\Anaconda3\envs\pdfkeywordsearching\Lib\site-packages\langdetect', 'langdetect')],
             hiddenimports=['pkg_resources.py2_warn', 'srsly'],
             hookspath=[],
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
               strip=False,
               upx=True,
               upx_exclude=[],
               name='processtext')
