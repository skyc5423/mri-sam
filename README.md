# mri-sam
segment anything model applied to mr images

First, download sam-hq git repo
git clone https://github.com/SysCV/sam-hq.git

1. conda activate {your_env}
2. cd sam-hq 
3. pip install -e .
4. pip install pytorch==1.10.0 torchvision==0.11.2
5. pip install opencv-python pycocotools matplotlib onnxruntime onnx

and open file 'sam-hq/segment_anything/build_sam.py'
modify line 106 as
state_dict = torch.load(f, map_location=torch.device('cpu'))

you can download model pth at https://github.com/SysCV/sam-hq
