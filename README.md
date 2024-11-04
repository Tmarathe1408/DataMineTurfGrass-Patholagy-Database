# DataMineTurfGrass-Patholagy-Database

Firstly you will need to install Docker on your local system by following the steps on the website below:

  https://docs.docker.com/get-started/get-docker/

After fully following the documentation guide, you should have a Docker application on your system

The next step is to clone the directory on VS Code. After getting the files on a directory, open up the docker application on your system. 

At this step you should have docker open and VS Code with all the files in one repository/directory. Now You need to open the terminal in VS Code and run the following command:

    docker-compose up -d

If this does not work try the following command to see if any of them work: 

    sudo docker-compose up -d

    docker compose up -d

    sudo docker compose up -d

After the command goes through, on the docker application you should see a new container. You can now start the ports and connection to the database by turning it on using docker. 

Now you want to make sure you have all the imports used in the test.py file installed on your system. If you do not have pip, install pip. This will allow you to install python moduls necessary for the vector database

Also make sure you are in the directory with all the files. 

    # Install pymilvus for Milvus interaction
    pip install pymilvus

    # Install sentence-transformers for generating embeddings
    pip install sentence-transformers

    # Install SQLite3 (comes pre-installed with Python, but you can install additional SQLite tools if needed)
    # This usually doesn't require installation, but just in case:
    pip install pysqlite3

    # Install numpy for numerical operations
    pip install numpy

Possible errors will have solutions below. These will be added during lab. 





