# FROM nvidia/cuda:11.4.0-devel-ubuntu20.04
FROM ubuntu:20.04

# Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# japanise setting
RUN apt-get update && apt-get install -y locales
RUN localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

# install utility
RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    git \
    vim \
    python3-pip

# install python package
WORKDIR /root/requirements
ADD requirements.txt /root/requirements
RUN pip3 install -r requirements.txt

# install fish
RUN apt-get update && apt-get install -y fish
SHELL ["/usr/bin/fish", "-c"]

# install fish plagin
RUN curl https://git.io/fisher --create-dirs -sLo ~/.config/fish/functions/fisher.fish
RUN fisher install jethrokuan/z
RUN curl -L https://get.oh-my.fish > install
RUN fish install --path=~/.local/share/omf --config=~/.config/omf
RUN omf install flash

WORKDIR /root/workspace
