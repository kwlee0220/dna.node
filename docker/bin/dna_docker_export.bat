@ECHO OFF

@REM localhost에 해당하는 local ip 주소를 환경변수 LOCAL_IP에 저장
for /F "tokens=2 delims=:" %%i in ('"ipconfig | findstr IP | findstr 192."') do SET LOCAL_IP=%%i
set LOCAL_IP=%LOCAL_IP: =%

docker run -it --rm --network host --name dna-export kwlee0220/dna-export %*