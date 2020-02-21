# README

## Requirements

> 1. tqdm 
> 2. PIL, scipy, numpy, logging
> 3. pytorch, torchvision
> 4. geopy, utm

-> 전부 conda로 설치 가능함

## 파일 배치도

![](https://i.imgur.com/cBc8Xcu.png)

- Tokyo247이라는 폴더 안 쪽에 PosNeg를 다운받아서 직접 넣어줘야 함(preexplored GT)

## 데이터 배치도 + 코드에서 수정해야할 부분 

- 데이터는 기본적으로 홈에 넣어줘야 함(홈에 넣고 싶지 않다면 아래 참고)
- 여기서 data_path_from_home이라는 변수의 값을 data의 path에 해당하도록 넣어주면 됨(dataset.py의 Line 14)


![](https://i.imgur.com/bbwpZRo.png)

- data_path 변수를 아무튼 데이터 경로로 갈 수 있도록만 하면 됨. 


아래는 이미지 데이터 이미지 구조도이다

    tokyoTimeMachine --------- images
                        |----- tokyoTM_train.mat
                        |----- tokyoTM_val.mat




## Dataloader와 Dataset 사용법

### Dataloader
는 dataloader.py를 import하고 사용하면 된다. 

```py    
trainLoader = TokyoValTrainLoader(mode, batch_size=4, collate_fn=make_batch, GT_from_file=True)
valLoader = TokyoValDataLoader(mode, batch_size=4, collate_fn=make_batch, GT_from_file=True)
```

mode = 'db' 또는 'query'를 필요에따라 넣고 사용하면 된다. 
batch_size는 편의에 따라 변경하면 될 것 같다. 
또한 query의 경우 db dataloader를 input으로 넣어줄 경우 pos, neg가 자동으로 계산된다.

아래는 코드의 예시이다.

![](https://i.imgur.com/D0C2On3.png)


한 번 출력해보면 dictionary가 return 되는데, 아래의 key별로 다음의 값을 저장한다. 

- filename -> 실제 파일명
- utm_coordinate -> utm으로 변환되어 있는 coordination
- timestamp -> 사진이 찍힌 날짜
- image -> Resize, ToTensor가 적용된 이미지

mode = 'query'로 설정하는 경우에는 

- pos : positive 이미지(db set에서의 index로 저장되어 있음)
- neg : negative 이미지(db set에서의 index로 저장되어 있음)

```python    
for data in trainLoader:
    print(data)
```

위와 같은 형태로 사용할 수 있다. 

### Dataset

DBLoader에서 dbset을 직접 가져올 수도 있는데, 이는 

![](https://i.imgur.com/xQffNAb.png)

아래와 같이 사용하여 index로 이미지에 접근할 수도 있다. 

## 구체적인 구현 

1. positive의 추출 과정은 전체 탐색에는 시간이 너무 오래 걸려(한 이미지당 10분 정도 소요), 4000개의 랜덤 이미지를 탐색하는 과정을 positive가 5개가 넘을 때까지 반복하였다. (보통은 3,4번의 iteration 안에 마무리 되었음)
2. negative의 경우 한 번의 탐색으로도 충분히 많은 양을 찾을 수 있었다. positive와 같은 iteration만큼 추출하되, 마지막에 그 중에 랜덤 샘플링을 하여(20개) 정리
3. 매번 GT 탐색을 하기에는 너무 오래걸려 별도의 제공된 파일(PosNeg(1GB)) 를 다운하여 별도의 과정 없이 바로 GT(pos, neg)를 가져올 수 있도록 하였다. 
4. Tokyo247의 경우 test query인데, 따로 쓸 일이 없다고 하셔서 구현하다가 멈췄다. 