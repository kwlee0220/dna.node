@ECHO OFF

@REM localhost에 해당하는 local ip 주소를 환경변수 LOCAL_IP에 저장
for /F "tokens=2 delims=:" %%i in ('"ipconfig | findstr IP | findstr 192."') do SET LOCAL_IP=%%i
set LOCAL_IP=%LOCAL_IP: =%

docker run -it --name dna_tool --rm ^
			-e DISPLAY=%LOCAL_IP%:0.0 ^
			--network=host ^
			-v %cd%\dna.node:/dna.node ^
			kwlee0220/dna-node %*