# Time-series-forecasting

## 0. 폴더 설명
<pre>
project
├─ data: 데이터
|  ├─ inference_result: 추론 결과
│  ├─ processed_data: 전처리된 데이터
│  │  ├─ node_link
│  │  └─ traffic
│  └─ raw_data: 전처리 안 된 데이터
│      ├─ node_link
│      └─ traffic
├─ model: 신경망 모델 코드 및 weight값
├─ preprocess: 데이터 전처리 코드
├─ util: 기타 코드(데이터 다운, 압축해제 등)
└─ visualize: 시각화 코드(시간 남으면)
</pre>

## 1. 초기 세팅
- data 디렉토리 생성 및 표준노드링크 정보 다운로드
- 방법: initialize.py 실행

## 2. 데이터 다운로드
### 현재 download_traffic_data.py는 2023년 09월 전체 데이터를 다운 받음.
- 교통상황 데이터 다운로드 및 압축 해제
- 방법: download_traffic_data.py 실행

## 3. 데이터 전처리
- 표준노드링크 파일로부터, 사용할 노드와 링크에 대한 정보 추출
- 교통상황 데이터 전처리
- 방법:  
    1. preprocess/get_daegu_node_link.ipynb 실행
    2. preprocess/select_daegu_node_link.ipynb 실행
    3. preprocess/process_all_traffic_data.ipynb 실행

## 4. 모델 설계
- model 폴더에 각자 설계

## 5. 모델 훈련
### 현재 train.ipynb는 2023년 09월 데이터로 훈련함.
- 방법:
    1. train.ipynb 실행

## 6. 모델 테스트
### 현재 inference.ipynb는 2023년 10월 데이터로 추론함.
- 방법:
    1. inference.ipynb 실행