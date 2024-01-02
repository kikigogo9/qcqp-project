# TODO

- [ ] Add product feature map - Nick
- [ ] Add hardware efficient ansatz - Willow
- [ ] Add a cost function - Kristian
- [ ] Add loss function - Nick
- [ ] Add boundary handling method - Willow
- [ ] Combine into a single runner, to handle simple ODEs - Kristian
## BONUS

- [ ] Add Chebyshev feature maps
- [ ] Extend the runner to handle nonlinear ODEs
<!--- 
- [ ] (for unchecked checkbox)
- [x] (for checked checkbox)
--->
# Resources
- https://pennylane.ai/qml/glossary/quantum_differentiable_programming/ (*)
- https://doi.org/10.1103/PhysRevA.103.052416 / https://arxiv.org/pdf/2011.10395.pdf (**)
- https://doi.org/10.1103/PhysRevA.104.052417 / https://arxiv.org/pdf/2108.01218.pdf (***)
- https://arxiv.org/abs/2306.17026 (****)
- https://arxiv.org/abs/2308.01827

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

Committing files after adding them to the commit:

`git commit -m "commit message"`

To push the commit use:

`git push`

Pulling commits on a recently modified branch:

`git pull`