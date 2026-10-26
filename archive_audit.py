
import csv, hashlib, os, sys, zipfile
from pathlib import Path
from collections import Counter

IMG={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
ANNOT={'.csv','.txt','.json','.xml','.yaml','.yml','.pbtxt'}

def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def inventory(path):
    with zipfile.ZipFile(path) as z:
        files=[n for n in z.namelist() if not n.endswith('/') and not n.startswith('__MACOSX/')]
        imgs=[n for n in files if Path(n).suffix.lower() in IMG]
        docs=[n for n in files if Path(n).suffix.lower() in ANNOT]
        byfmt=Counter(Path(n).suffix.lower() for n in imgs)
        byfolder=Counter(str(Path(n).parent) for n in imgs)
        splitcounts=Counter()
        for n in imgs:
            parts=[p.lower() for p in Path(n).parts]
            split=next((p for p in parts if p in ('train','valid','val','validation','test')),'unspecified')
            splitcounts[split]+=1
        return imgs,docs,byfmt,byfolder,splitcounts

def main():
    base=Path(sys.argv[1] if len(sys.argv)>1 else '.').expanduser()
    if not base.exists():sys.exit(f'Folder missing: {base}')
    paths=sorted(base.rglob('*.zip'))
    if not paths:sys.exit('No ZIP archives found in folder')
    rows=[];folderrows=[];annotrows=[]
    for p in paths:
        try:
            imgs,docs,fmt,folders,splits=inventory(p)
            rows.append({'archive':p.name,'sha256':checksum(p),'images':len(imgs),'image_formats':'; '.join(f'{k}:{v}' for k,v in fmt.items()),'splits':'; '.join(f'{k}:{v}' for k,v in splits.items()),'annotation_files':len(docs),'status':'ZIP scanned; label correctness, licence and source still require review'})
            for folder,count in sorted(folders.items()):folderrows.append({'archive':p.name,'folder':folder,'image_count':count,'note':'Folder names may be classes OR split folders; verify manually'})
            for item in docs:annotrows.append({'archive':p.name,'annotation_or_metadata_file':item})
            print(f'{p.name}: {len(imgs)} images, {len(docs)} annotation/metadata candidates')
        except Exception as e:print(f'ERROR reading {p.name}: {e}')
    out=base/'F01_audit_results';out.mkdir(exist_ok=True)
    for name,data,fields in [('archive_summary.csv',rows,['archive','sha256','images','image_formats','splits','annotation_files','status']),('image_folder_counts.csv',folderrows,['archive','folder','image_count','note']),('annotation_file_list.csv',annotrows,['archive','annotation_or_metadata_file'])]:
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    print('Saved audit reports to',out)
    print('NOTE: scanned image counts do not confirm unique original images or correct disease labels.')
if __name__=='__main__':main()
