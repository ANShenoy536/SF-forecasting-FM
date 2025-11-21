# Pre-processing

This requires the SHARP $Bp$, $Br$, $Bt$, $magnetograms$, $conf_disambig$, $bitmap$ segments and if you require the $Bp_err$, $Br_err$ and $Bt_err$ segments.

## 1. Generate the FMs for vertical current, unsigned current helicity and photospheric magnetic free energy density 
### (and their error files if required)

`python FieldMapGenerator.py <base_dir path> <dest_dir path>`

The source or base directory containing the SHARP segments should be structured as follows:
base_dir
|--> NOAA_AR numbers
      |--> T_RECs of indiv. examples
            |-->SHARP segments

dest_dir is where the generated FMs will be saved.
\
\
## 2. Pre-processing all the files
This follows the following pipeline: 
<img width="4160" height="1320" alt="preprocessing_flow_paper2" src="https://github.com/user-attachments/assets/294e5914-fc54-4c03-b4ae-ab726f6caa85" />

The pre-processing pipeline in housed in the jupyter noteook `PadCrop_Norm.ipynb`. However, it requires the centroids of the bitmaps from `Get_Bitmap_Centroids.py`. Therefore, run it first.

`python Get_Bitmap_Centroids.py <bit_dir path> <csv_dest path>`
## 3. Generating the FMs for the unsigned magnetic flux
This file generates the FM for the unsigned magnetic flux from the pre-processed  $B_\phi$, $B_r$, $B_\theta$ magnetograms using \ 

\begin{equation} 
USFLUX = \sqrt{\frac{B_\phi^2 + B_r^2 + B_\theta^2}{3}}
\end{equation}
\
Thus, the generated FMs need not be pre-processed again.
`python Gen_USFLUX_FM.py <src_dir path>
src_dir is the directory containing the pre-processed Bp, Br and Bt FMs. It is also where the pre-processed USFLUX directory will be created.
