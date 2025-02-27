FROM python:3.9

WORKDIR /wsgi

COPY Pipfile Pipfile.lock /wsgi/

RUN pip install pipenv && pipenv install --deploy --ignore-pipfile

RUN pipenv run pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pipenv run python -m spacy download en_core_web_sm

COPY . /wsgi

EXPOSE 3008

CMD ["pipenv", "run", "flask", "run", "--host=0.0.0.0", "--port=3008"]
