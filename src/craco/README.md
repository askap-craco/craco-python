pipeline runs with grid lut generated with new way, runs without hang
confirmed that the lookup table generated here is the same as the one, runs without hang
backup before get more numbers from pickle file, runs without hang

sometimes is interesting, which we need to run the test_pipeline_ref.py first, it is the version without anything added by my

pipeline_short.pickle, nt = 256, nuv is about 200
pipeline_long.pickle,  nt = 16, nuv is about 6,000
pipeline.pickle,       nt = 256, nuv is about 6,000

binary_container_1.xclbin was compiled with MAX_NSMP_UV=4800, we can not run it with 6,000 UVs, grid will hang if we do so

we also need nt > 16 as boxcar history needs that to work 