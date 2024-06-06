FROM python:3.9

WORKDIR /wsgi

COPY Pipfile Pipfile.lock /wsgi/

RUN pip install pipenv

RUN pipenv install --deploy --ignore-pipfile

COPY . /wsgi

EXPOSE 3008

CMD ["pipenv", "run", "flask", "run", "--host=0.0.0.0", "--port=3008"]
