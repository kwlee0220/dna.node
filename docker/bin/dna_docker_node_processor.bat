@ECHO OFF

@REM localhost에 해당하는 local ip 주소를 환경변수 LOCAL_IP에 저장
for /F "tokens=2 delims=:" %%i in ('"ipconfig | findstr IP | findstr 192."') do SET LOCAL_IP=%%i
set LOCAL_IP=%LOCAL_IP: =%

docker run -it --name dna-node-processor ^
			--rm ^
			--gpus all ^
			-e DNA_NODE_RABBITMQ_HOST=rabbitmq:5672 ^
			-e DNA_NODE_KAFKA_BROKERS=kafka02:19092 ^
			-e DNA_NODE_CONF_ROOT=conf ^
			-e DISPLAY=%LOCAL_IP%:0.0 ^
			--network=dna_server_net ^
			-v .:/dna.node ^
			kwlee0220/dna-node %*