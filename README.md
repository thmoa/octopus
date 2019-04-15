# Octopus

This repository contains code corresponding to the paper **Learning to Reconstruct People in Clothing from a Single RGB Camera**.

## Installation

Download and install DIRT: https://github.com/pmh47/dirt.

Download the neutral SMPL model from http://smplify.is.tue.mpg.de/ and place it in the `assets` folder.
```
cp <path_to_smplify>/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl assets/neutral_smpl.pkl
```

Download pre-trained model weights from [here](https://drive.google.com/open?id=1_CwZo4i48t1TxIlIuUX3JDo6K7QdYI5r) and place them in the `weights` folder.

```
unzip <downloads_folder>/octopus_weights.hdf5.zip -d weights
```


## Usage

We provide scripts and sample data for single subject (`infer_single.py` ) and batch processing (`infer_batch.py`).
Both scripts output usage information when executed without parameters.

### Quick start

We provide sample scripts for both modes:

```
bash run_demo.sh
bash run_batch_demo.sh
```

## Data preparation

If you want to process your own data, some pre-processing steps are needed:

1. Crop your images to 1080x1080.
2. Run [PGN semantic segmentation](https://github.com/Engineering-Course/CIHP_PGN) on your images.
3. Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) body_25 and face keypoint detection on your images.

Semantic segmentation and OpenPose keypoints form the input to Octopus. See `data` folder for sample data.

## Texture

The following code may be used to stitch a texture for the reconstruction: https://github.com/thmoa/semantic_human_texture_stitching


## Citation

This repository contains code corresponding to:

T. Alldieck, M. Magnor, B. L. Bhatnagar, C. Theobalt, and G. Pons-Moll.
**Learning to Reconstruct People in Clothing from a Single RGB Camera**. In
*IEEE Conference on Computer Vision and Pattern Recognition*, 2019.

Please cite as:

```
@inproceedings{alldieck19cvpr,
    title = {Learning to Reconstruct People in Clothing from a Single {RGB} Camera},
    author = {Alldieck, Thiemo and Magnor, Marcus and Bhatnagar, Bharat Lal and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {jun},
    year = {2019},
}
```


## License

Copyright (c) 2019 Thiemo Alldieck, Technische Universität Braunschweig, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the **Learning to Reconstruct People in Clothing from a Single RGB Camera** paper in documents and papers that report on research using this Software.
