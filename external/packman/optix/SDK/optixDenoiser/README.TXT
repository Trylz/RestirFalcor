denoise all AOVs:
optixDenoiser_exp -a data/albedo.exr \
                  -A data/diffuse.exr -A data/glossy.exr -A data/specular.exr \
                  -o result.exr data/beauty.exr

When the RGB pixels in the output files result_diffuse_denoised.exr, result_glossy_denoised.exr
and result_specular_denoised.exr are added (composited), the resulting image should look very close
to result.exr (the denoised beauty layer).

optixDenoiser -k enables the inference model used for AOVs, even if no AOVs are given. The beauty
layer is denoised with this model.
This model preserves colors much better and might remove low frequency noise better compared to the
non-AOV inference model.
---------------------------------------------------------------------------------------------------
temporal denoising (using direct prediction, temporal kernel prediction not supported):
optixDenoiser_exp -F 1-20 -a motiondata/soane-BSDF-+++.exr \
                          -n motiondata/soane-Normal-+++.exr \
                          -f motiondata/soane-Flow-+++.exr \
                          -o result-+++.exr motiondata/soane-Beauty-+++.exr
It is recommended to add -e -8.6 to the command line when creating a clip with ffmpeg (exposure
control). A clip can be created with ffmpeg -framerate 20 -i result-%03d.exr soane.mp4

To get a clip from the noisy beauty images first multiply the images by pow(2, -8.6) for example
with oiitool (part of the OpenImageIO library):
oiiotool --frames 1-20 --cmul 2.577163e-3 motiondata/soane-Beauty-%03d.exr -o noisy%03d.exr
ffmpeg -framerate 20 -i noisy%03d.exr noisy.mp4
---------------------------------------------------------------------------------------------------
The class OptixDenoiser has a method 'getFlowResults' which can be used to verify flow vectors.
All data have to be set up exactly as for denoising, but when calling 'getFlowResults' instead of
'getResults', flow vectors are applied at each pixel to the (noisy) input images and written back
to the layer result. If the command line option -z is set, only flow is applied, no denoising.
