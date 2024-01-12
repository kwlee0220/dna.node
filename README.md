# dna.node

## 설치
DNANode를 설치하고 사용하기 위해서는 다음이 준비되어 있어야 한다.
- Conda: Conda 이외의 가상 환경 플랫폼을 사용할 수 있으나, 본 문서는 conda를 기준으로 설치하는 것을 가정한다.
본 문서에서는 dna.node이름의 가상 환경을 사용하는 것을 가정한다. 상황에 따라 다른 이름의 가상 환경 이름을 사용하여도 무방하다.
- Nvidia GPU: DNANode에서 제공하는 일부 명령어의 경우에는 Nvidia GPU를 사용한다. 또한 본 설치 가이드에서는
Nvidia driver는 이미 설치되어 있는 것을 가정한다.
- Docker가 설치되고, 실행 중인 것을 가정한다.

### 1. Python 가상 환경 생성
본 문서에서는 가상 환경 관리를 위해 [Anaconda](https://www.anaconda.com/)를 사용하는 것을 가정한다.
```
conda create -n dna.node python=3.10
conda activate dna.node
```
Pytorch 버전 호환성을 위해 python 3.10 버전을 설치한다.
Python 3.10 이후 버전을 사용하면 'cu113'용 Pytorch를 설치할 수 없다.

### 2. Pytorch 설치 및 CUDA 동작 확인
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

python -c "import torch; print(torch.cuda.is_available())"
True
```
위 python 수행을 통해 True가 화면에 출력되어야 한다. 그렇지 않다면 Nvidia GPU, driver, 또는 pytorch 중 하나 이상에 오류가 있다는 의미이다.

### 3. Pip를 통한 dna.node 설치
```
pip install dna.node
```

### 4. FFMPEG 설치
일부 RTSP URL을 사용하는 카메라 소스를 지원하기 위해서는 [ffmpeg 라이브러리](https://ffmpeg.org/download.html)를 설치할 필요가 있다.
설치된 ffmeg bin 디렉토리를 system path에 등록시킨다.


### 5. DNANode 소스 다운로드
DNANode github에서 [dna.node](https://github.com/kwlee0220/dna.node.git)를 clone하고,
생성된 디렉토리로 이동한다.
소스를 사용하지 않더라도 DNANode 수행에 필요한 설정 정보나 docker compose 설정 파일을 사용하기 위해서 download하는 것을 권장한다.
```
git clone https://github.com/kwlee0220/dna.node.git
cd dna.node
```

### 6. DNA infra 수행
```
cd dna.node/docker/compose
docker compose -f docker-compose-infra.yml -p dna-infra up -d
```
도커 수행이 시작되면, 웹을 통해 [KafkaGUI](http://localhost:8080)에 접속하여 동작을 확인한다.

### 7. 기본 동작 확인
아래의 명령을 수행하여 동작을 확인한다.

#### 7.1 Sample event 데이터를 Kafka topic들 (node-tracks, track-features)에 적재시킨다.
```
dna_replay data/samples/etri_04_event.pickle --progress --max_wait_ms 1000
```
KafkaGUI를 사용하여 두 topic에 데이터가 적재되었음을 확인한다.

#### 7.2 Kafka topic에 저장된 이벤트를 화면에 출력한다.
```
dna_print_events --topics node-tracks track-features --kafka_offset earliest --timeout 1000
```


## 주요 command 명령어들

### 1. `dna_node`: 차량 추적 및 이벤트 발송
`dna_node`는 주어진 카메라에서 영상을 획득하여 영상에 등장하는 차량의 위치를 지속적으로 추적하고, 해당 정보를
지정된 Kafka topic들에 저장시킨다. 기본적인 사용방법은 아래와 같다.
```
dna_node <설정 파일 경로> [options]
```
설정 파일은 YAML 형식을 사용하며 예제는 [conf.yaml](conf/sample.yaml)를 참조하면 된다.

옵션 | 인자  | 설명
-------------|------------|---------
-h, --help   |            | 본 프로그램의 도움말을 출력하고 종료시킨다.
--camera     | \<number> \| \<video-path> \| \<rstp-url>    | 영상 획득에 사용할 카메라를 지정.</br> * \<number> USB등으로 연결된 지역 카메라 번호. (예: 0)</br> * \<video-path>: 비디오 파일 경로. (예: `data/test.mp4`)</br> * \<rstp-url>: RSTP 영상 URL. (예: `rtsp://admin:password@129.254.99.99:558/LiveChannel/0/media.smp`)
--init_ts    | 0 \| open \| realtime \| YYYYMMDDThhmmss | 획득된 영상 frame에 부여될 timestamp 지정.</br>- 0: 첫번째 frame부터 FPS로 계산하는 timestamp를 부여. 예를들어 FPS가 10인 경우에는 100, 200, 300, ... 가 차례대로 frame에 부여된다.</br> - open: 카메라 개방이 완료되는 시점을 기준으로 timestamp가 FPS으로 계산된 시간만큼 증가하며 부여됨.</br> - realtime: 카메라에서 영상을 획득한 실제 시각을 기준으로 timestamp가 부여됨.</br> - YYYYMMDDThhmmss: 주어진 시각을 기준으로 FPS로 계산된 시간만큼 증가하여 timestamp가 부여.
--begin_frame  | \<number> | 분석에 사용할 첫번째 영상의 프레임 번호.
--end_frame    | \<number> | 분석에 사용할 종료 영상의 프레임 번호. **지정된 번호에 해당하는 프레임을 포함되지 않는다.**
--sync         |            | 영상의 각 프레임을 FPS에 맞추어서 획득함. 본 option은 동영상에서 영상을 획득하는 경우에 의미가 있음.
--show         | [\<width>x\<hight>] | 획득된 영상의 출력 여부. Parameter를 통해 영상 크기가 지정될 수 있으며 별도 지정이 없는 경우에는 카메라에서 획득된 영상 크기로 화면에 출력됨.
--title  | \<spec> | 영상이 화면에 출력되는 경우, 화면에 표시될 프레임 상태 정보를 지정. 사용할 수 있는 정보는 다음과 같다. </br> • date: 영상 획득 날짜.</br> • time: 영상 획득 시각.</br> • ts: 영상 획득 timestamp.</br> • frame: 획득 영상의 프레임 번호.</br> • fps: 측정된 fps.</br> 한 번에 여러 정보를 표시하기 위해서는 위 정보를 '+' 기호를 분리자로 결합하여 지정한다. 예를 들어 `frame+ts`의 경우에는 프레임 번호와 FPS가 화면에 표시된다.
--output_video | \<file-path> | 처리된 영상이 저장될 동영상 파일 경로.
--crf | 'opencv' \| 'ffmpeg' \| 'lossless' | 저장될 영상의 품질. 'opencv' \< 'ffmpeg' \< 'lossless' 순서대로 영상 품질은 향상된다. 본 option을 사용하지 않는 경우는 'opencv'를 가정한다.
--output | \<file-path> | 영상 분석으로 생성된 이벤트가 저장될 파일 경로. 이벤트들을 pickle 형식으로 저장된다.
--kafka_brokers| \<접속 정보> | 영상 분석으로 생성된 이벤트를 전송할 Kafka 접속 정보. 별도로 지정하지 않으면 `localhost:9092`를 설정된다.
--silent_kafka| | 생성된 이벤트를 Kafka로 전송하지 않도록 막음.
--progress| | 수행 진척 상황 표시 여부
--logger | \<file-path> | 프로그램 수행 중 출력되는 log 메시지를 위한 설정 정보 파일 경로명. 별도로 지정하지 않는 경우 `conf/logger.yaml` 경로를 사용함.

#### 예제 1
아래 명령은 RSTP url `rtsp://admin:password@129.254.99.99:558/LiveChannel/0/media.smp`에서 발송하는 카메라 영상 정보를
사용하여 차량 위치 추적 프로그램을 수행한다. 획득되는 영상 프레임에 부여하는 timestamp는 카메라 개방 시간을 기준으로 FPS를 기준으로
순차 증가하며 부여된다. 분석된 결과로 발생된 이벤트들은 설정에 따라 Kafka topic에 전송된다.
이때 사용하는 설정은 `conf/etri_01.yaml` 파일에서 읽어온다.
```
dna_node conf/etri_01.yaml --camera rtsp://admin:password@129.254.99.99:558/LiveChannel/0/media.smp --init_ts open
```

#### 예제 2
아래 명령은 동영상 `data/samples/test4.mp4`에서 영상을 읽어 차량 위치 추적 프로그램을 수행한다. 영상 프레임은 1번 프레임부터 999번 프레임까지만
사용하게 되며, 각 프레임은 2023년 11월 10일 오전 9시 3분을 기준으로 FPS에 의해 계산된 시간만큼 추가해가며 timestamp가 부연된다.
분석된 결과로 발생된 이벤트들은 Kakfa topic에 전송되지 **않고**, 대신 `output/etri_04_events.pickle` 파일에 저장된다.
분석 결과 영상은 화면에 표시되고, 화면에는 프레임 번호, 계산된 FPS가 출력된다. 또한 영상 처리의 진행 정도가 화면에 출력된다. 분석 설정은 `conf/etri_04.yaml` 파일을 사용한다.
```
dna_node conf/etri_04.yaml --camera data/samples/test4.mp4 --end_frame 1000 --init_ts 20231110T090300 --show --title frame+ts+fps --progress --silent_kafka --output output/etri_04_events.pickle
```


### 2. `dna_node_server`: DNA node 서버 프로그램
`dna_node_server`는 외부에서 Redis를 통해 들어오는 분석 요청을 수행하는 작업을 수행한다.
```
dna_node_server [options]
```

옵션 | 인자  | 설명
----------------|-------------|---------
-h, --help      |             | 본 프로그램의 도움말을 출력하고 종료시킨다.
--conf_root     | \<dir-path> | 본 프로그램이 사용할 분석 설정 파일들이 저장될 디렉토리를 지정함.
--redis         | \<url>      | 분석 요청 전달에 사용할 Redis 서버 URL.
--req_channel   | \<name>     | Redis 서버 내 PubSub 채널 이름.
--sync          |            | 영상의 각 프레임을 FPS에 맞추어서 획득함. 본 option은 동영상에서 영상을 획득하는 경우에 의미가 있음.
--show          | [\<width>x\<hight>] | 획득된 영상의 출력 여부. Parameter를 통해 영상 크기가 지정될 수 있으며 별도 지정이 없는 경우에는 카메라에서 획득된 영상 크기로 화면에 출력됨.
--logger | \<file-path> | 프로그램 수행 중 출력되는 log 메시지를 위한 설정 정보 파일 경로명. 별도로 지정하지 않는 경우 `conf/logger.yaml` 경로를 사용함.
#### 예제
```
dna_node_server --conf_root conf --redis redis://localhost:6379?db=0 --req_channel track-requests --show
```


### 3. `dna_show`: 영상 출력
`dna_show`는 주어진 카메라에서 영상을 획득하여 화면에 출력하는 작업을 수행한다.
```
dna_show [options]
```
옵션 | 인자  | 설명
----------------|------------|---------
-h, --help      |            | 본 프로그램의 도움말을 출력하고 종료시킨다.
--camera     | \<number></br> \| \<video-path></br> \| \<rstp-url>    | 영상 획득에 사용할 카메라를 지정.</br> * \<number> USB등으로 연결된 지역 카메라 번호. (예: 0)</br> * \<video-path>: 비디오 파일 경로. (예: `data/test.mp4`)</br> * \<rstp-url>: RSTP 영상 URL. (예: `rtsp://admin:password@129.254.99.99:558/LiveChannel/0/media.smp`)
--init_ts    | 0 \| open \| realtime</br> \| YYYYMMDDThhmmss | 획득된 영상 frame에 부여될 timestamp 지정.</br>- 0: 첫번째 frame부터 FPS로 계산하는 timestamp를 부여. 예를들어 FPS가 10인 경우에는 100, 200, 300, ... 가 차례대로 frame에 부여된다.</br> - open: 카메라 개방이 완료되는 시점을 기준으로 timestamp가 FPS으로 계산된 시간만큼 증가하며 부여됨.</br> - realtime: 카메라에서 영상을 획득한 실제 시각을 기준으로 timestamp가 부여됨.</br> - YYYYMMDDThhmmss: 주어진 시각을 기준으로 FPS로 계산된 시간만큼 증가하여 timestamp가 부여.
--begin_frame  | \<number> | 분석에 사용할 첫번째 영상의 프레임 번호.
--end_frame    | \<number> | 분석에 사용할 종료 영상의 프레임 번호. **지정된 번호에 해당하는 프레임을 포함되지 않는다.**
--sync         |            | 영상의 각 프레임을 FPS에 맞추어서 획득함. 본 option은 동영상에서 영상을 획득하는 경우에 의미가 있음.
--show         | [\<width>x\<hight>] | 획득된 영상의 출력 여부. Parameter를 통해 영상 크기가 지정될 수 있으며 별도 지정이 없는 경우에는 카메라에서 획득된 영상 크기로 화면에 출력됨.
--title  | \<spec> | 영상이 화면에 출력되는 경우, 화면에 표시될 프레임 상태 정보를 지정. 사용할 수 있는 정보는 다음과 같다. </br> * date: 영상 획득 날짜.</br> * time: 영상 획득 시각.</br> * ts: 영상 획득 timestamp.</br> * frame: 획득 영상의 프레임 번호.</br> * fps: 측정된 fps.</br> 한 번에 여러 정보를 표시하기 위해서는 위 정보를 '+' 기호를 분리자로 결합하여 지정한다. 예를 들어 `frame+ts`의 경우에는 프레임 번호와 FPS가 화면에 표시된다.
--output_video | \<file-path> | 처리된 영상이 저장될 동영상 파일 경로.
--crf | 'opencv' \| 'ffmpeg' \| 'lossless' | 저장될 영상의 품질. 'opencv' \< 'ffmpeg' \< 'lossless' 순서대로 영상 품질은 향상된다. 본 option을 사용하지 않는 경우는 'opencv'를 가정한다.
--progress| | 수행 진척 상황 표시 여부
--logger | \<file-path> | 프로그램 수행 중 출력되는 log 메시지를 위한 설정 정보 파일 경로명. 별도로 지정하지 않는 경우 `conf/logger.yaml` 경로를 사용함.
#### 예제
```
* dna_show --camera rtsp://admin:password@129.254.99.99:558/PlaybackChannel/4/media.smp/start=20231030T090000&end=20231030T090500 --init_ts 20231030T090000
* dna_show --camera 0 --init_ts runtime --show --title frame+ts+fps --progress
```


### 4. `dna_print_events`: 이벤트 출력
`dna_print_events`는 Kafka topic에 저장된 이벤트를 화면에 출력한다.
```
dna_print_events [options]
```
옵션 | 인자  | 설명
----------------|-------------|---------
-h, --help      |             | 본 프로그램의 도움말을 출력하고 종료시킨다.
--kafka_brokers | \<접속 정보> | 영상 분석으로 생성된 이벤트를 전송할 Kafka 접속 정보. 별도로 지정하지 않으면 `localhost:9092`를 설정된다.
--kafka_offset  | latest \| earliest \| none | Kafka topic의 시작 offset
--topics        | \<topic-name>, ... | 출력할 이벤트를 보유한 topic 이름 (리스트)
--poll_timeout  | \<milli-secs> | Kafka poll timeout을 설정한다. 별도로 지정하지 않는 경우는 1000로 설정된다.
--initial_poll_timeout | \<milli-secs> | Initial poll timeout를 설정한다. 일반적으로 초기 Kafka 접속 시간은 길어지기 때문에 별도로 지정할 필요가 있다. 별도로 지정하지 않는 경우는 5000로 설정된다.
--timeout       | \<milli-secs> | Timeout를 설정한다. Kafka에서 주어진 기간 내에 이벤트를 획득하지 못하는 경우 프로그램을 종료시킨다.
--drop_poll_timeout |         | Poll timeout이 발생하는 경우, 해당 정보를 화면에 출력하지 않는다.
--type          | node-track</br>\| global-track</br>\| track-feature</br>\| json-event | 출력할 event 타입.
--filter | \<expr>        | 이벤트 필터 표현식


### 5. `dna_export_topic`: Kafka topic 이벤트 export
`dna_export_topic`는 Kafka topic에 저장된 이벤트를 지정된 파일로 export 시킨다.
```
dna_export_topic <file> [options]
```
옵션 | 인자  | 설명
----------------|-------------|---------
-h, --help      |             | 본 프로그램의 도움말을 출력하고 종료시킨다.
--kafka_brokers | \<접속 정보> | 영상 분석으로 생성된 이벤트를 전송할 Kafka 접속 정보. 별도로 지정하지 않으면 `localhost:9092`를 설정된다.
--kafka_offset  | latest \| earliest \| none | Kafka topic의 시작 offset
--topics        | \<topic-name>, ... | 출력할 이벤트를 보유한 topic 이름 (리스트)
--poll_timeout  | \<milli-secs> | Kafka poll timeout을 설정한다. 별도로 지정하지 않는 경우는 1000로 설정된다.
--initial_poll_timeout | \<milli-secs> | Initial poll timeout를 설정한다. 일반적으로 초기 Kafka 접속 시간은 길어지기 때문에 별도로 지정할 필요가 있다. 별도로 지정하지 않는 경우는 5000로 설정된다.
--timeout       | \<milli-secs> | Timeout를 설정한다. Kafka에서 주어진 기간 내에 이벤트를 획득하지 못하는 경우 프로그램을 종료시킨다.
--progress| | 수행 진척 상황 표시 여부
--logger | \<file-path> | 프로그램 수행 중 출력되는 log 메시지를 위한 설정 정보 파일 경로명. 별도로 지정하지 않는 경우 `conf/logger.yaml` 경로를 사용함.


### 6. `dna_import_topic`: Kafka topic 이벤트 import
`dna_import_topic`는 주어진 파일에 기록된 event를 지정된 Kafka topic에 import 시킨다.
```
dna_import_topic <file> [options]
```
옵션 | 인자  | 설명
----------------|-------------|---------
-h, --help      |             | 본 프로그램의 도움말을 출력하고 종료시킨다.
--kafka_brokers | \<접속 정보> | 영상 분석으로 생성된 이벤트를 전송할 Kafka 접속 정보. 별도로 지정하지 않으면 `localhost:9092`를 설정된다.
--topic        | \<topic-name> | 출력할 이벤트를 보유한 topic 이름 (리스트)
--progress| | 수행 진척 상황 표시 여부
--logger | \<file-path> | 프로그램 수행 중 출력되는 log 메시지를 위한 설정 정보 파일 경로명. 별도로 지정하지 않는 경우 `conf/logger.yaml` 경로를 사용함.


### 7. `dna_replay`: 노드 이벤트 replay
`dna_replay`는 주어진 파일에 기록된 event를 지정된 Kafka topic에 import 시킨다.
```
dna_replay <file> [options]
```
옵션 | 인자  | 설명
----------------|-------------|---------
-h, --help      |             | 본 프로그램의 도움말을 출력하고 종료시킨다.
--kafka_brokers | \<접속 정보> | 이벤트를 전송할 Kafka 접속 정보. 별도로 지정하지 않으면 `localhost:9092`를 설정된다.
--sync         |            | 이벤트내에 포함된 timestamp에 동기화시켜 해당 이벤트를 전송시킨다..
--progress| | 수행 진척 상황 표시 여부
--logger | \<file-path> | 프로그램 수행 중 출력되는 log 메시지를 위한 설정 정보 파일 경로명. 별도로 지정하지 않는 경우 `conf/logger.yaml` 경로를 사용함.


### 8. `dna_show_gtracks`: 전역 위치 애니메이션
`dna_show_gtracks`는 Kafka topic 'global-tracks'에 저장된 global track 이벤트를 활용해 항공 지도에 차량의 위치 정보를
animation 효과로 가시화한다.
```
dna_show_gtracks <file> [options]
```
옵션 | 인자  | 설명
----------------|-------------|---------
-h, --help      |             | 본 프로그램의 도움말을 출력하고 종료시킨다.
--kafka_brokers | \<접속 정보> | 영상 분석으로 생성된 이벤트를 전송할 Kafka 접속 정보. 별도로 지정하지 않으면 `localhost:9092`를 설정된다.
--kafka_offset  | latest \| earliest \| none | Kafka topic의 시작 offset
--topics        | \<topic-name>, ... | 출력할 이벤트를 보유한 topic 이름 (리스트)
--poll_timeout  | \<milli-secs> | Kafka poll timeout을 설정한다. 별도로 지정하지 않는 경우는 1000로 설정된다.
--initial_poll_timeout | \<milli-secs> | Initial poll timeout를 설정한다. 일반적으로 초기 Kafka 접속 시간은 길어지기 때문에 별도로 지정할 필요가 있다. 별도로 지정하지 않는 경우는 5000로 설정된다.
--timeout       | \<milli-secs> | Timeout를 설정한다. Kafka에서 주어진 기간 내에 이벤트를 획득하지 못하는 경우 프로그램을 종료시킨다.
--show_support  |  | 노드별 차량 위치 정보를 통해 단일 전역 위치가 생성된 경우, 해당 노드별 차량 위치 표시 여부.
--sync          |  | 전역 차량 애니메이션 시간 동기화 여부.
--output_video  | \<path>  | 애니메이션 영상을 활용한 동영상 생성 여부.
--no_show       |  | 애니메이션 화면 출력 중지 여부.
--progress| | 수행 진척 상황 표시 여부
--logger | \<file-path> | 프로그램 수행 중 출력되는 log 메시지를 위한 설정 정보 파일 경로명. 별도로 지정하지 않는 경우 `conf/logger.yaml` 경로를 사용함.


### 9. `dna_show_mc_locations`: NodeTrack 이벤트 기반 다중 물체 위치 display
`dna_show_mc_locations`는 주어진 NodeTrack 이벤트 파일을 사용하여 다중 물체의 위치를 지도에 출력한다.
```
dna_show_mc_locations <track-files> [options]
```
옵션 | 인자  | 설명
----------------|-------------|---------
-h, --help      |             | 본 프로그램의 도움말을 출력하고 종료시킨다.
--offsets       | \<num>,\<num>,...,\<num> | 노드 이벤트의 상대 오프셋
--start_frame   | \<num> | 시작 프레임 번호
--interactive   |             | 차량 위치 정보 출력 애니메이션 중지
--logger        | \<file-path> | 프로그램 수행 중 출력되는 log 메시지를 위한 설정 정보 파일 경로명. 별도로 지정하지 않는 경우 `conf/logger.yaml` 경로를 사용함.