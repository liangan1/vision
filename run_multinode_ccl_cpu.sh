torch_ccl_path=$(python -c "import torch; import intel_pytorch_extension as ipex; import os;  print(os.path.abspath(os.path.dirname(ipex.__file__)))")
source $torch_ccl_path/../env/setvars.sh
#python  -m intel_pytorch_extension.launch --distributed --hostfile hostfile  --nnodes 8 --no_python pkill python 
python  -m intel_pytorch_extension.launch --distributed --hostfile hostfile  --nnodes 1 references/classification/train.py --lr 0.1 --model resnet50 --ipex --dnnl --mix-precision --data-path /lustre/dataset/imagenet/img_raw/  -b 16 -j 48  --dist-backend=ccl --mpi-launcher  
