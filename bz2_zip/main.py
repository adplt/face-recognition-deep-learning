import bz2
import shutil

zipfile = bz2.BZ2File('./00001/00001_930831_fa_a.ppm.bz2')

with bz2.open('./00001/00001_930831_fa_a.ppm.bz2') as fh, open(newname, 'wb') as fw:
    shutil.copyfileobj(fh, fw)
