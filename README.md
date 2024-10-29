# DataMineTurfGrass-Patholagy-Database

Firstly you will need to install Docker on your local system by following the steps on the website below:

https://docs.docker.com/get-started/get-docker/

After that you should have a Docker application. Open the Docker application and run the following commands in the terminal on Docker:

curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

bash standalone_embed.sh start

docker pull milvusdb/milvus:v2.4.3-hotfix


docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.4.3-hotfix

milvis download:

wget https://github.com/milvus-io/milvus/releases/download/v2.1.4/milvus-standalone-docker-compose.yml -O docker-compose.yml

docker run --hostname=cd4b2c2515e8 --env=ETCD_USE_EMBED=true --env=ETCD_DATA_DIR=/var/lib/milvus/etcd --env=ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml --env=COMMON_STORAGETYPE=local --env=PATH=/milvus/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --env=LD_LIBRARY_PATH=/milvus/lib::/usr/lib --env=LD_PRELOAD=/milvus/lib/libjemalloc.so --env=MALLOC_CONF=background_thread:true --volume=/Users/Hruthin1/volumes/milvus:/var/lib/milvus --volume=/Users/Hruthin1/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml --volume=/Users/Hruthin1/user.yaml:/milvus/configs/user.yaml --network=bridge --workdir=/milvus/ -p 19530:19530 -p 2379:2379 -p 9091:9091 --restart=no --label='org.opencontainers.image.ref.name=ubuntu' --label='org.opencontainers.image.version=22.04' --runtime=runc -d milvusdb/milvus:v2.4.13-hotfix

