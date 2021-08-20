FROM python:3.6
COPY ./requirements.txt /nishika/requirements.txt
WORKDIR /nishika
RUN pip3 install -r /nishika/requirements.txt