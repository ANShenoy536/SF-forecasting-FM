# Pre-processing

This requires the SHARP `Bp`, `Br`, `Bt`, `magnetogram`, `conf_disambig`, `bitmap` segments and if you require the `Bp_err`, `Br_err` and `Bt_err` segments.

## 1. Generate the FMs for vertical current, unsigned current helicity and photospheric magnetic free energy density 
```bash
python FieldMapGenerator.py <base_dir> <dest_dir>
```

- `base_dir` is the path to the source or base directory containing the SHARP segments should be structured as follows: \
&emsp; <img width="295" height="105" alt="image" src="https://github.com/user-attachments/assets/a5c3f4f6-5091-42f0-9714-9eced10ddadd" />
- `dest_dir` is the path to the directory where the generated FMs will be saved.
## 2. Pre-processing all the files
This follows the following pipeline: 
<img width="4160" height="1320" alt="preprocessing_flow_paper2" src="https://github.com/user-attachments/assets/294e5914-fc54-4c03-b4ae-ab726f6caa85" />

The pre-processing pipeline is housed in the jupyter noteook `PadCrop_Norm.ipynb`. However, it requires the centroids of the bitmaps from `Get_Bitmap_Centroids.py`. Therefore, run it first.
```bash
python Get_Bitmap_Centroids.py <bit_dir> <csv_dest>
```
- `bit_dir` : string : is the path to the directory containing all the SHARP bitmaps
- `csv_dest` : string : is the path to the output csv file listing the centroids of the bitmaps
## 3. Generating the FMs for the unsigned magnetic flux
This file generates the FM for the unsigned magnetic flux from the pre-processed  $B_\phi$, $B_r$, $B_\theta$ magnetograms using \
&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; $$USFLUX = \sqrt{\frac{B_\phi^2 + B_r^2 + B_\theta^2}{3}}$$
\
Thus, the generated FMs need not be pre-processed again.
```bash
python Gen_USFLUX_FM.py <ffcsv> <src_dir>`
```
- `ffcsv` : string : is the path to the FiveFoldCV.csv file.
- `src_dir` : string : is the path to the directory containing the pre-processed Bp, Br and Bt FMs. It is also where the pre-processed USFLUX directory will be created.

