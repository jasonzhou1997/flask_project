FROM python:3
COPY . .
RUN python3 -m pip install -r requirements.txt
EXPOSE 8080
CMD flask run --host 0.0.0.0 --port 8080
