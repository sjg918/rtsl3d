# rtsl3d
구현해야할목록 : 

dataloader
"rgb" : 위의 100pixel 잘라내기.
"lidar" : 카메라 차원으로 변환이 필요함.
"target" : 어떻게 해야할지 감도안옴.

network
"cosine similarity cost volume" : 해놨음.
"Roi" : 생성된 latent space에 있는 lidar point들을 voxel에 맞춰 우겨 넣어야함. 아마 numba로 구현할듯?
"sparse 3d conv" : spconv가 pip로 풀렸기 때문에 쉬워졌음. 하지만 역시 해본적없음.
"head" : 아마 rts3d를 가져다쓰면될듯?

loss
"keypoint heatmap" : rtm3d에 있음. 클래스 확률. 깊이. 방향.
"3d iou" : point r cnn것 사용. rtm3d에 있음.


지구멸망기원

![mosaic](https://github.com/sjg918/rtsl3d/blob/main/image.png?raw=true)
