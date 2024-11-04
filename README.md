# DataMineTurfGrass-Patholagy-Database

Firstly you will need to install Docker on your local system by following the steps on the website below:

https://docs.docker.com/get-started/get-docker/

After fully following the documentation guide, you should have a Docker application on your system

curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

bash standalone_embed.sh start

docker pull milvusdb/milvus:v2.4.3-hotfix


docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.4.3-hotfix

milvis download:

wget https://github.com/milvus-io/milvus/releases/download/v2.1.4/milvus-standalone-docker-compose.yml -O docker-compose.yml

