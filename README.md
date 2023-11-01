# Time-series-forecasting

## 0. 폴더 설명
<pre>
project
├─ data: 데이터
│  ├─ processed_data: 전처리된 데이터
│  │  ├─ node_link
│  │  └─ traffic
│  └─ raw_data: 전처리 안 된 데이터
│      ├─ node_link
│      └─ traffic
├─ model: 신경망 모델
├─ preprocess: 데이터 전처리 코드
├─ util: 기타 코드(데이터 다운, 압축해제 등)
└─ visualize: 시각화 코드(시간 남으면)
</pre>

## 1. 초기 세팅
- data 디렉토리 생성 및 표준노드링크 정보 다운로드
- 방법: initialize.py 실행

## 2. 데이터 다운로드
### 현재 download_traffic_data.py는 2023년 09월 10일 데이터만 다운 받음. 추후 수정 예정
- 교통상황 데이터 다운로드 및 압축 해제
- 방법: download_traffic_data.py 실행

## 3. 데이터 전처리
- 표준노드링크 파일로부터, 사용할 노드와 링크에 대한 정보 추출
- 교통상황 데이터 전처리
- 방법:  
    1. preprocess/get_daegu_node_link.ipynb 실행
    2. preprocess/select_node_link.ipynb 실행
    3. preprocess/process_traffic_data.ipynb 실행

## 4. 모델 설계, 훈련/테스트
### 현재 train_test.ipynb는 2023년 09월 10일 데이터로 훈련하고, 2023년 09월 11일 데이터로 5분 뒤 도로 속도 예측 수행.  
- 설계한 모델은 model 폴더에 작성
- 모델 훈련 및 테스트는 아래 방법 참고
- 방법:
    1. train_test.ipynb 실행