FROM continuumio/anaconda3
COPY . /hongxiang/app
WORKDIR /hongxiang/app
RUN apt-get update 
RUN apt-get -y install vim
RUN apt-get -y install dos2unix
RUN apt-get -y install cmake
RUN apt-get -y install libx11-dev
RUN apt-get -y install flex
WORKDIR ./dhsvm/no_sat_dump/
RUN make -f makefile_for_binary
WORKDIR /hongxiang/app
RUN apt-get -y install gfortran
RUN apt-get install -y --no-install-recommends build-essential r-base
RUN mkdir -p ./Rlibrary && chmod 777 ./Rlibrary
RUN R -e "install.packages('lmom', dependencies=TRUE, repos='http://cran.rstudio.com/', lib='./Rlibrary')"
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]

