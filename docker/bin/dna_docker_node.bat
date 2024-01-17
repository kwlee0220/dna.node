@ECHO OFF

@REM localhost에 해당하는 local ip 주소를 환경변수 LOCAL_IP에 저장
for /F "tokens=2 delims=:" %%i in ('"ipconfig | findstr IP | findstr 192."') do SET LOCAL_IP=%%i
set LOCAL_IP=%LOCAL_IP: =%

docker run -it --name dna-node ^
			--rm ^
			--gpus all ^
			-e DNA_NODE_FFMPEG_PATH=/opt/conda/bin/ffmpeg ^
			-e DISPLAY=%LOCAL_IP%:0.0 ^
			--network=dna_server_net ^
			-v ./dna.node:/dna.node -v ./torch_hub:/root/.cache/torch/hub ^
			kwlee0220/dna-node ^
			dna_node conf/etri_04.yaml --camera test.mp4 --kafka_brokers localhost:9092 --init_ts runtime %*