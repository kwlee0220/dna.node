# dna.node

# Installation
DNANode를 설치하고 사용하기 위해서는 다음이 준비되어 있어야 한다.
- Conda: Conda 이외의 가상 환경 플랫폼을 사용할 수 있으나, 본 문서는 conda를 기준으로 설치하는 것을 가정한다. 본 문서에서는 `dna.node`이름의 가상 환경을 사용하는 것을 가정한다. 상황에 따라 다른 이름의 가상 환경 이름을 사용하여도 무방하다.
- Nvidia GPU: DNANode에서 제공하는 일부 명령어의 경우에는 Nvidia GPU를 사용한다. 또한 본 설치 가이드에서는 Nvidia driver는 이미 설치되어 있는 것을 가정한다.

## Installation from GitHub
1. DNANode github에서 [dna.node](https://github.com/kwlee0220/dna.node.git)를 fork함.
```
git clone https://github.com/kwlee0220/dna.node.git
cd dna.node
```

2. 가상 환경 생성
```
conda create -n dna.node python=3.10
conda activate dna.node
```
Pytorch 버전 호환성을 위해 python 3.10 버전을 가정한다. Python 3.10 이후 버전을 사용하면 'cu113'용 Pytorch를 설치할 수 없다.

3. Pytorch 설치
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
python -c "import torch; print(torch.cuda.is_available())"
```
위 python 수행을 통해 True가 화면에 출력되어야 한다. 그렇지 않다면 Nvidia GPU, driver, 또는 pytorch 중 하나 이상에 오류가 있다는 의미이다.

4. 기타 활용 module들 설치
```
pip install -r requirements.txt
```