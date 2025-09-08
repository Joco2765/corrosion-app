1. ##### **Overview** 



This Streamlit app segments corrosion areas in images, image batches (ZIP), and videos.

It uses an HRNet-W48 backbone (via timm) with a lightweight segmentation head, optimized for fast single-class inference.

TorchDynamo/Inductor are disabled to avoid any compilation step and keep startup simple on CPU or GPU.



##### **2. Features**



###### Three input modes:



* Single Image (JPG/PNG)
* Batch ZIP of images (JPG/PNG) with a results ZIP and a summary CSV
* Video (MP4/AVI/MOV/MKV) with downscale + frame-skipping for speed



###### Strict checkpoint loading:



* Cleans “module.” prefixes (DataParallel)
* Accepts top-level dicts like {state\_dict: ...} or {model: ...}
* Fails fast if state\_dict keys do not match the model (clear error)



###### Fast pre/post-processing:



* Optional long-side resize and /32 padding before the network
* Thresholding, morphology (closing/opening), hole filling
* Red overlay with adjustable alpha



###### Downloadable outputs:



* Binary mask PNG and overlay PNG
* ZIP bundle for batch mode (masks, overlays, summary.csv)
* Annotated video (MP4 if available, falls back to AVI automatically)



###### GPU-friendly without compilation:



* Optional AMP (float16) on CUDA if available
* TorchDynamo/Inductor explicitly disabled



##### **3. Sidebar Settings (what they do)**



* **Checkpoint (.pth):** Path to your PyTorch checkpoint file on the machine running the app.
* **Images:** max long side (px): Downscale the longer side before inference. Lower = faster, potentially less detail.
* **Binary threshold:** Probability threshold (0–1) to binarize the mask. Typical start: 0.50.
* **Morphology — Closing:** Fills small gaps in predicted areas.
* **Morphology — Opening:** Removes small noisy blobs.
* **Fill holes:** Fills cavities fully enclosed by the mask.
* **Overlay alpha:** Red overlay transparency on top of the input.
* **Video:** max long side (px): Downscale video frames before inference (helps memory and speed).
* **Video:** process every Nth frame: Frame skipping. Only infer every Nth frame, reuse last mask otherwise.



#### **4. Inputs and Outputs**



###### Single Image:



* Input: One JPG/PNG.
* Output: Mask PNG (binary), Overlay PNG (red = predicted corrosion).



###### Batch ZIP:



* Input: A .zip containing JPG/PNG images.
* Output: A ZIP named “predictions.zip” 



###### Video:



* Input: MP4/AVI/MOV/MKV.
* Output: Annotated video (MP4 if encoder is available; fallback to AVI/XVID).







