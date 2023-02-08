 
 @ECHO OFF
 SET current_path=%~dp0
 docker run --gpus all -p 8888:8888 --memory=24g -v %current_path%:/app/project/ -ti project-dev bash