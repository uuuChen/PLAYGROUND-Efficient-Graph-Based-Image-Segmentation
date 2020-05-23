## PLAYGROUND-Efficient-Graph-Based-Image-Segmentation
> This is a demo of [Efficient Graph-Based Image Segmentation](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf). It only
differs from the paper that "Nearest Neighbor Graphs" is not applied.

## Usage
```
python main.py -ifp ./images/sample.png  --k 2000 --seg-by-rgb
python main.py -ifp ./images/sample2.png --k 2000 --seg-by-rgb
python main.py -ifp ./images/sample3.png --k 2000 --seg-by-rgb
python main.py -ifp ./images/sample4.png --k 2000 --seg-by-rgb
```
* ifp       : image-file-path, the path of the image to be segmented
* k         : if k is larger, the image will be divided coarser, else if k is smaller, the image will be divided finer
* seg-by-rgb: whether to use rgb three channels to segment the image

## Results
### sample.png
![](https://i.imgur.com/gpP2s85.png)
### sample2.png
![](https://i.imgur.com/0ebNasO.png)
### sample3.png
![](https://i.imgur.com/dL2ZXXD.png)
### sample4.png
![](https://i.imgur.com/j3CjK2z.png)
