import numpy as np, nibabel as nib, os, nilearn, json, struct
from scipy import ndimage as ndi
from skimage import measure
import trimesh, fast_simplification
from PIL import Image
d=os.path.join(os.path.dirname(nilearn.__file__),'datasets','data')
L=lambda n: nib.load(f'{d}/mni_icbm152_{n}_tal_nlin_sym_09a_converted.nii.gz')
t1=L('t1'); gm=L('gm').get_fdata(); wm=L('wm').get_fdata(); T=t1.get_fdata(); aff=t1.affine
print('shape',T.shape,'codes',nib.aff2axcodes(aff),'zooms',t1.header.get_zooms(), T.min(),T.max(), gm.max(), wm.max())
tissue=(gm+wm)/max(gm.max(),1e-9) if gm.max()>1.5 else gm+wm
tissue=ndi.gaussian_filter(tissue,0.7)
mask=tissue>0.5
lab,n=ndi.label(mask); sizes=ndi.sum(mask,lab,range(1,n+1)); mask=lab==(1+int(np.argmax(sizes)))
field=np.where(ndi.binary_dilation(mask,iterations=2),tissue,0)
v,f,_,_=measure.marching_cubes(field,0.5)
vw=nib.affines.apply_affine(aff,v)   # RAS mm
m=trimesh.Trimesh(vw,f,process=True)
m=max(m.split(only_watertight=False),key=lambda c:len(c.faces))
print('raw',len(m.vertices),len(m.faces))
vs,fs=fast_simplification.simplify(m.vertices.astype(np.float32),m.faces.astype(np.int32),target_count=140000)
m=trimesh.Trimesh(vs,fs,process=True); trimesh.smoothing.filter_taubin(m,lamb=0.5,nu=-0.53,iterations=8)
if m.volume<0: m.invert()
print('final',len(m.vertices),len(m.faces),'bounds',m.bounds.tolist())
# sulcal shade: depth below a smoothed hull -> per-vertex 0..1
hull=ndi.binary_closing(mask,iterations=12); dist=ndi.distance_transform_edt(hull)
ijk=nib.affines.apply_affine(np.linalg.inv(aff),m.vertices)
depth=ndi.map_coordinates(dist,ijk.T,order=1,mode='nearest')
shade=np.clip(1-depth/9.0,0,1)
lo=m.bounds[0]; hi=m.bounds[1]
q=np.round((m.vertices-lo)/(hi-lo)*65535).astype('<u2')
out='docs/assets/'
with open(out+'brain.bin','wb') as fh:
    fh.write(struct.pack('<4sII6f',b'BRN1',len(q),len(m.faces),*lo,*hi))
    fh.write(q.tobytes()); fh.write(np.round(shade*255).astype('u1').tobytes())
    pad=(-fh.tell())%4; fh.write(b'\0'*pad)
    fh.write(m.faces.astype('<u4').tobytes())
print('brain.bin',os.path.getsize(out+'brain.bin'))
# slice atlas: brain-masked T1 at 1.5 mm, axial slices, anterior up, patient right on screen right (as in the repo QC)
brainmask=ndi.binary_fill_holes(ndi.binary_closing(mask,iterations=3))
Tm=np.where(ndi.gaussian_filter(brainmask.astype(float),1)>0.5,T,0)
fg=Tm[Tm>0]; p1,p99=np.percentile(fg,[1,99]); print('window',p1,p99)
step=1.5
x=np.arange(lo[0]-4,hi[0]+4,step); y=np.arange(lo[1]-4,hi[1]+4,step); z=np.arange(lo[2]-2,hi[2]+2,step)
X,Y,Z=np.meshgrid(x,y,z,indexing='ij'); ijk=nib.affines.apply_affine(np.linalg.inv(aff),np.stack([X,Y,Z],-1))
vol=ndi.map_coordinates(Tm,ijk.transpose(3,0,1,2),order=1,mode='constant')
vol=np.clip((vol-p1)/(p99-p1),0,1); vol[vol<0.02]=0
fgc=(vol>0).sum(axis=(0,1))
print('vol',vol.shape,'fg per z min/max',fgc.min(),fgc.max())
nx,ny,nz=vol.shape; cols=12; rows=int(np.ceil(nz/cols))
atlas=np.zeros((rows*ny,cols*nx),np.uint8)
for k in range(nz):
    sl=np.flipud(vol[:,:,k].T)  # rows: anterior at top; columns: +x (patient right) to screen right
    r,c=divmod(k,cols); atlas[r*ny:(r+1)*ny,c*nx:(c+1)*nx]=np.round(sl*255)
Image.fromarray(atlas).save(out+'mni_slices.png',optimize=True)
meta=dict(nx=nx,ny=ny,nz=nz,cols=cols,step=step,x0=float(x[0]),y0=float(y[0]),z0=float(z[0]),fg=[int(a) for a in fgc],meshLo=[float(a) for a in lo],meshHi=[float(a) for a in hi])
json.dump(meta,open('explorer/aggregates/volmeta.json','w'))
print('png',os.path.getsize(out+'mni_slices.png'),meta['nx'],meta['ny'],meta['nz'])
