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
├─ preprocess: 데이터 전처리 코드
├─ util: 기타 코드(데이터 다운, 압축해제 등)
└─ visualize: 시각화 코드(시간 남으면)
</pre>

## 기본 데이터로 수행하는 경우, 1~3은 생략 가능

## 1. 초기 세팅 (데이터 추가 및 변경 시 실행)
- data 디렉토리 생성 및 표준노드링크 정보 다운로드
- 방법: initialize.py 실행

## 2. 데이터 다운로드 (데이터 추가하려는 경우 실행)
### 현재 download_traffic_data.py는 2023년 09월 전체 데이터를 다운 받음. target_list를 수정하면 다른 데이터 다운 가능
- 교통상황 데이터 다운로드 및 압축 해제
- 방법: download_traffic_data.py 실행

## 3. 데이터 전처리 (데이터 추가 시 실행)
- 표준노드링크 파일로부터, 사용할 노드와 링크에 대한 정보 추출
- 교통상황 데이터 전처리
- 방법:  
    1. preprocess/get_daegu_node_link.ipynb 실행
    2. preprocess/select_daegu_node_link.ipynb 실행
    3. preprocess/process_all_traffic_data.ipynb 실행

## 4. 모델 훈련 및 추론 수행
- 개발의 용이성을 위해 훈련과 추론을 한 파일 내에서 수행
- 방법: TPN.ipynb 실행
