# python-docker

Download docker from https://www.docker.com/

In a console navigate to your desired folder and clone the project with the following command:

`git clone https://github.com/kikigogo9/qcqp-project.git`

In the command line go to the project folder:

`cd qcqp-project`

Now you can run docker here:

`docker-compose up -d`

The image will be built in a few minutes.

To access the image remotely we can use:

`docker exec -it CONTAINER_ID sh`

Which will launch an interactive shell from within the docker.
We can get the `CONTAINER_ID` by checking out the currently running containers with:

`docker ps`


# git

To create a new branch use:

`git branch branch_name`

To switch branches use:

`git checkout branch_name`

You can see your changed files by running:

`git status`

To add a file to the commit use:

`git add file_path/filename`

To add every changed file use:

`git add .`

Commiting files after adding them to the commit:

`git commit -m "commit message"`

To push the commit use:

`git push`

Pulling commits on a recently modified branch:

`git pull`