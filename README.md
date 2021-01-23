# PCT-tensorflow
Tensorflow implementation of PCT: Point Cloud Transformer.  
Paper link: https://arxiv.org/pdf/2012.09688.pdf  

The code is based on pointnet, pointnet2 and PCT_Pytorch

pointnet:     https://github.com/charlesq34/pointnet  

pointnet2:    https://github.com/charlesq34/pointnet2  

PCT_Pytorch:  https://github.com/uyzhang/PCT_Pytorch  

# classification
The data used in point cloud cls is ModelNet40 and split as pointnet.  

After making tf_ops, downloading the modelnet40 and put it in datasets/modelnet40_ply_hdf5_2048.  
Using train.py to train the PCT model, and testing via test.py.  

The OA of my test results is 89%+, not as SOTA as paper demonstrated and other repo.  

I doubt there exist some errors in my implementation.  

If there exist any errors, please contact me via committing issue or liufeng@radi.ac.cn

