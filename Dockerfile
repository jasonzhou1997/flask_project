FROM python:3
COPY . .
RUN python3 -m pip install -r requirements.txt
CMD flask run --port 8080
