# dna.node

## Installation
DNANode를 설치하고 사용하기 위해서는 다음이 준비되어 있어야 한다.
- Conda: Conda 이외의 가상 환경 플랫폼을 사용할 수 있으나, 본 문서는 conda를 기준으로 설치하는 것을 가정한다.
본 문서에서는 dna.node이름의 가상 환경을 사용하는 것을 가정한다. 상황에 따라 다른 이름의 가상 환경 이름을 사용하여도 무방하다.
- Nvidia GPU: DNANode에서 제공하는 일부 명령어의 경우에는 Nvidia GPU를 사용한다. 또한 본 설치 가이드에서는
Nvidia driver는 이미 설치되어 있는 것을 가정한다.

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

### 3. DNANode 소스 다운로드
DNANode github에서 [dna.node](https://github.com/kwlee0220/dna.node.git)를 clone하고,
생성된 디렉토리로 이동한다.
소스를 사용하지 않더라도 설정 정보나 docker compose 설정 파일을 사용하기 위해서 download하는 것을 권장한다.
```
git clone https://github.com/kwlee0220/dna.node.git
cd dna.node
```

### 4. Pip를 통한 dna.node 설치
```
pip install dna.node
```


## Installation from GitHub

### 2. PYTHONPATH 환경 변수 등록
시스템 환경 변수 'PYTHONPATH'에 clone된 디렉토리를 추가한다.
```
(Linux 환경)
set PYTHONPATH=.../dna.node
(Windows 환경)
setx PYTHONPATH %cd%    # Fork된 현 디렉토리인 것을 가정함.
```

### 3. 기본 동작 확인
아래의 명령을 수행하여 동작을 확인한다. 
```
cd <설치된 (clone된) 디렉토리>
python scripts\dna_print_events.py --help
```